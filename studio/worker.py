"""Isolated one-engine process for previews, training and evaluation."""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
import time
import traceback
import hashlib
import importlib.metadata
import math
import numpy as np
import torch
import gymnasium as gym
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure, KVWriter
from stable_baselines3.common.monitor import Monitor
from studio.visualization import StudioEnv
from rl.vec_env import ResearchVecEnv
from rl.algorithms import create_model, load_model, REGISTRY
from studio.config import environment_config, atomic_json


class Recorder(KVWriter):
    def __init__(self, directory):
        self.directory = directory
        self.file = (directory/'metrics.jsonl').open('a', buffering=1)
        self.start = time.monotonic()

    def write(self, key_values, key_excluded=None, step=0):
        values = {str(k):float(v) for k,v in key_values.items() if isinstance(v,(int,float,np.number,bool))}
        if not all(math.isfinite(v) for v in values.values()):
            raise FloatingPointError('Nonfinite scalar metric; training halted')
        if values:
            self.file.write(json.dumps(dict(step=int(step), elapsed=time.monotonic()-self.start, values=values), allow_nan=False)+'\n')

    def close(self):
        if not self.file.closed:
            self.file.close()


class ViewEnv(gym.Wrapper):
    def __init__(self, env, directory, interval, overview=False):
        super().__init__(env)
        self.directory, self.interval, self.last_frame = directory, interval, 0
        self.overview = overview

    def frame(self, force=False):
        if not force and time.monotonic()-self.last_frame < self.interval:
            return
        if self.env.unwrapped.view_mode != 'topdown':
            pixels = self.env.unwrapped.camera_frame()
        else:
            options = {"scaling": 10}
            if self.overview:
                x0,x1,y0,y1 = self.env.current_map.road_network.get_bounding_box()
                options.update(camera_position=((x0+x1)/2,(y0+y1)/2), scaling=700/max(x1-x0,y1-y0,1))
            pixels = self.env.render(mode='topdown', window=False, screen_size=(800,800), film_size=(1600,1600),
                                     num_stack=3, draw_target_vehicle_trajectory=False, **options)
        tmp = self.directory/'frame.tmp.jpg'
        Image.fromarray(np.asarray(pixels).astype(np.uint8)).save(tmp, quality=82)
        tmp.replace(self.directory/'frame.jpg')
        self.last_frame = time.monotonic()

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.frame(True)
        return result

    def step(self, action):
        result = self.env.step(action)
        self.frame(force=result[2] or result[3])
        return result


def save_checkpoint(model, directory, name, c, env_config):
    path = directory/(name+'.zip')
    tmp = directory/(name+'.tmp.zip')
    model.save(tmp)
    tmp.replace(path)
    atomic_json(path.with_suffix('.json'), dict(algorithm=c['algorithm'], model_file=path.name, step=model.num_timesteps,
               settings=c, environment_config=env_config, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


class TrainingCallback(BaseCallback):
    def __init__(self, directory, c, env_config, recorder):
        super().__init__()
        self.directory, self.c, self.env_config, self.recorder = directory,c,env_config,recorder
        self.episodes=[]
        self.step_costs=[]
        self.last_status=0

    def _on_step(self):
        info = self.locals['infos'][0]
        self.step_costs.append(float(info['cost']))
        if 'episode_safety' in info:
            row=dict(episode=len(self.episodes)+1, step=self.num_timesteps, **info['episode_safety'])
            self.episodes.append(row)
            with (self.directory/'episodes.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False)+'\n')
            vals={'episode/'+k:float(v) for k,v in row.items() if isinstance(v,(float,int,bool)) and v is not None}
            recent=self.episodes[-20:]
            for k in ('success','collision','out_of_road','episode_reward','episode_risk_sum','route_completion'):
                vals['rolling20/'+k]=float(np.mean([r[k] for r in recent]))
            for k,v in vals.items():
                self.logger.record(k,v)
            self.logger.dump(self.num_timesteps)
        if self.num_timesteps % self.c['log_freq'] == 0:
            self.logger.record('step/cost', info['cost'])
            self.logger.record('step/mean_cost', np.mean(self.step_costs))
            self.step_costs.clear()
            for key in ('vehicle_risk','road_risk','min_ttc','min_distance','road_clearance','active_traffic_count'):
                if info.get(key) is not None:
                    self.logger.record('step/'+key, info[key])
            self.logger.dump(self.num_timesteps)
        if time.monotonic()-self.last_status>.5:
            atomic_json(self.directory/'status.json',dict(state='running',kind='train',step=self.num_timesteps,total=self.c['timesteps'],
                episodes=len(self.episodes), scene_seed=info['scene_seed'], cost=info['cost'], device=str(self.model.device)))
            self.last_status=time.monotonic()
        if self.num_timesteps % self.c['checkpoint_freq'] == 0:
            save_checkpoint(self.model,self.directory,f'checkpoint_{self.num_timesteps}',self.c,self.env_config)
        return not (self.directory/'STOP').exists()


def train(directory,c,env_config):
    env=ResearchVecEnv([lambda: Monitor(ViewEnv(StudioEnv(research_config=env_config, view_mode=c.get('view_mode','topdown')),directory,c['frame_interval']))])
    recorder=Recorder(directory)
    logger=configure(str(directory/'tensorboard'), ['tensorboard','csv'])
    logger.output_formats.append(recorder)
    model=None
    try:
        model=create_model(c,env)
        model.set_logger(logger)
        cb=TrainingCallback(directory,c,env_config,recorder)
        model.learn(total_timesteps=c['timesteps'],callback=cb,log_interval=1)
        logger.dump(model.num_timesteps)
        save_checkpoint(model,directory,'model',c,env_config)
        state='stopped' if (directory/'STOP').exists() else 'completed'
        atomic_json(directory/'summary.json',dict(timesteps=model.num_timesteps,episodes=len(cb.episodes),algorithm=c['algorithm'],
                   dataset=env_config.get('dataset'), cost_used_for_optimization=REGISTRY[c['algorithm']].uses_cost,device=str(model.device)))
        return dict(state=state,kind='train',step=model.num_timesteps,total=c['timesteps'],episodes=len(cb.episodes))
    finally:
        env.close()
        logger.close()


def empirical_cvar(values, tail):
    ordered=np.sort(np.asarray(values,dtype=float))[::-1]
    n=len(ordered)*tail
    whole=int(n)
    total=ordered[:whole].sum()
    if whole<len(ordered):
        total+=(n-whole)*ordered[whole]
    return float(total/n)


def evaluate(directory,c,env_config,model_path):
    from scripts.evaluate_env import summarize
    model=load_model(c['algorithm'],model_path,device=c['device'])
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(c['seed'], using_cuda=str(model.device).startswith('cuda'))
    env=ViewEnv(StudioEnv(research_config=env_config, view_mode=c.get('view_mode','topdown')),directory,c['frame_interval'])
    recorder=Recorder(directory)
    logger=configure(str(directory/'tensorboard'),['tensorboard','csv'])
    logger.output_formats.append(recorder)
    rows=[];steps=0
    try:
        for episode in range(c['episodes']):
            if (directory/'STOP').exists(): break
            obs,_=env.reset(seed=c['seed_start']+episode)
            discounted=0.;power=1.
            while True:
                action,_=model.predict(obs,deterministic=c['deterministic'])
                obs,reward,terminated,truncated,info=env.step(action)
                steps+=1
                discounted+=power*info['cost'];power*=c['gamma_cost']
                if steps % c['log_freq']==0:
                    for k in ('cost','vehicle_risk','road_risk','min_ttc','min_distance','active_traffic_count'):
                        if info.get(k) is not None: logger.record('step/'+k,info[k])
                    logger.dump(steps)
                if steps % 10==0:
                    atomic_json(directory/'status.json',dict(state='running',kind='eval',step=steps,episodes=len(rows),total=c['episodes'],scene_seed=info['scene_seed'],cost=info['cost']))
                if terminated or truncated:
                    row=dict(episode=episode+1,step=steps,discounted_cost=discounted,**info['episode_safety'])
                    rows.append(row)
                    with (directory/'episodes.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                    for k,v in row.items():
                        if isinstance(v,(int,float,bool)) and v is not None:logger.record('episode/'+k,v)
                    summary=summarize(rows)
                    for k,v in summary.items():
                        if isinstance(v,(int,float)):logger.record('evaluation/'+k,v)
                    logger.record('evaluation/discounted_cost_cvar',empirical_cvar([r['discounted_cost'] for r in rows],c['tail_fraction']))
                    logger.dump(steps)
                    break
                if (directory/'STOP').exists():break
            if (directory/'STOP').exists():break
        summary=summarize(rows) if rows else {'episodes':0}
        summary.update(dataset=env_config.get('dataset'), discounted_cost_cvar=empirical_cvar([r['discounted_cost'] for r in rows],c['tail_fraction']) if rows else None,
                       tail_fraction=c['tail_fraction'],gamma_cost=c['gamma_cost'],effective_tail_samples=len(rows)*c['tail_fraction'],
                       incomplete_episode_discarded=(directory/'STOP').exists(),device=str(model.device))
        atomic_json(directory/'summary.json',summary)
        return dict(state='stopped' if (directory/'STOP').exists() else 'completed',kind='eval',step=steps,episodes=len(rows),total=c['episodes'])
    finally:
        env.close();logger.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);args=p.parse_args()
    directory=args.job.resolve();job=json.loads((directory/'request.json').read_text());c=job['settings'];kind=job['kind']
    torch.set_num_threads(1)
    atomic_json(directory/'status.json',dict(state='starting',kind=kind,step=0))
    try:
        cfg=environment_config(c,kind)
        atomic_json(directory/'environment_config.json',cfg)
        if cfg.get('dataset'):
            from datasets.importer import load_clip
            clip,manifest=load_clip(cfg['dataset']['id'])
            atomic_json(directory/'dataset_manifest.json',manifest)
            atomic_json(directory/'dataset_clip.json',clip)
        metadata=dict(versions={k:importlib.metadata.version(k) for k in ('metadrive-simulator','panda3d','stable-baselines3','torch','gymnasium','numpy')},
            source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ('env','rl','studio','datasets') for f in (ROOT/folder).glob('*.py')},
            model_sha256=hashlib.sha256(Path(job['model_path']).read_bytes()).hexdigest() if job.get('model_path') else None)
        atomic_json(directory/'runtime_metadata.json',metadata)
        if kind=='preview':
            env=ViewEnv(StudioEnv(research_config=cfg, view_mode=c.get('view_mode','topdown')),directory,.1,overview=True)
            try:env.reset(seed=c['seed_start'])
            finally:env.close()
            result=dict(state='completed',kind=kind,step=0)
        elif kind=='train':result=train(directory,c,cfg)
        else:result=evaluate(directory,c,cfg,job['model_path'])
        atomic_json(directory/'status.json',result)
    except Exception as e:
        traceback.print_exc()
        atomic_json(directory/'status.json',dict(state='failed',kind=kind,error=str(e)))
        raise


if __name__=='__main__':main()
