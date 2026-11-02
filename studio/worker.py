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
import subprocess
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

    def pixels(self):
        if self.env.unwrapped.view_mode != 'topdown':
            return np.asarray(self.env.unwrapped.camera_frame()).astype(np.uint8)
        options = {"scaling": 10}
        if self.overview:
            x0,x1,y0,y1 = self.env.current_map.road_network.get_bounding_box()
            options.update(camera_position=((x0+x1)/2,(y0+y1)/2), scaling=700/max(x1-x0,y1-y0,1))
        # The preview panel is displayed at <500 CSS pixels; a large offscreen
        # film buffer adds seconds without making the road layout more legible.
        return np.asarray(self.env.render(mode='topdown', window=False, screen_size=(480,480), film_size=(640,640),
                                     num_stack=1, draw_target_vehicle_trajectory=False, **options)).astype(np.uint8)

    def frame(self, force=False):
        if not force and time.monotonic()-self.last_frame < self.interval:
            return
        pixels = self.pixels()
        if self.overview:
            rgb=np.asarray(pixels).astype(np.uint8)
            mask=np.any(rgb<238,axis=2)
            ys,xs=np.where(mask)
            if len(xs):
                dense_rows=np.flatnonzero(mask.sum(axis=1)>max(80,int(rgb.shape[1]*.17)))
                if len(dense_rows):ys=dense_rows
                pad=max(18,int(min(rgb.shape[:2])*.035))
                x0=max(0,int(xs.min())-pad);x1=min(rgb.shape[1],int(xs.max())+pad+1)
                y0=max(0,int(ys.min())-pad);y1=min(rgb.shape[0],int(ys.max())+pad+1)
                pixels=rgb[y0:y1,x0:x1]
        tmp = self.directory/'frame.tmp.jpg'
        Image.fromarray(np.asarray(pixels).astype(np.uint8)).save(tmp, quality=82)
        tmp.replace(self.directory/'frame.jpg')
        self.last_frame = time.monotonic()

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        return result


def render_srairl_frame(directory, env, info):
    """Capture the shared MetaDrive world, with a schematic fallback for CLI runs."""
    from PIL import ImageDraw
    if env.world is not None:
        if env.world.view_mode=='topdown':
            pixels=env.world.render(mode='topdown',window=False,screen_size=(960,720),film_size=(1600,1200),
                num_stack=3,draw_target_vehicle_trajectory=False,scaling=16)
        else:
            pixels=env.world.camera_frame(overview=env.world.view_mode=='bird3d')
        image=Image.fromarray(np.asarray(pixels).astype(np.uint8));draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,image.width,34),fill=(25,31,39))
        mode=getattr(env,'dynamics_mode','legacy_kinematic');response=getattr(getattr(env.world.engine,'traffic_manager',None),'reactive',False)
        if getattr(env,'scene_source',None)=='procedural':
            caption=f'MetaDrive {env.scenario_type} · 2 SR-AIRL policy cars + {info.get("traffic_vehicle_count",0)} traffic vehicles'
        else:caption=f'MetaDrive · {mode} · reactive replay {response} · ELCV (blue) · RFV (orange)'
        draw.text((12,10),caption,fill='white')
        return np.asarray(image.convert('RGB'))
    tracks=env.scenario.tracks
    t=float(info['time'])
    points=[]
    for track in tracks.values():
        states=np.asarray(track['states'],dtype=float)
        if states[0,0]-1e-6<=t<=states[-1,0]+1e-6:
            points.append((np.interp(t,states[:,0],states[:,1]),np.interp(t,states[:,0],states[:,2])))
    if not points:return
    xlo=min(p[0] for p in points)-20;xhi=max(p[0] for p in points)+20
    ylo=min(p[1] for p in points)-12;yhi=max(p[1] for p in points)+12
    width,height=960,560;image=Image.new('RGB',(width,height),(91,125,72));draw=ImageDraw.Draw(image)
    left,right,top,bottom=54,width-24,72,height-44
    def xy(x,y):return (left+(x-xlo)/max(xhi-xlo,1)*(right-left),bottom-(y-ylo)/max(yhi-ylo,1)*(bottom-top))
    road_top=xy(xlo,env.scenario.lane_num*env.scenario.lane_width)[1]
    road_bottom=xy(xlo,0)[1]
    draw.rectangle((left,road_top,right,road_bottom),fill=(75,78,80))
    for lane in range(1,env.scenario.lane_num):
        y=xy(xlo,lane*env.scenario.lane_width)[1]
        for x in range(left,right,36):draw.line((x,y,min(x+20,right),y),fill=(224,224,206),width=2)
    palette={'elcv':(238,70,58),'rfv':(53,130,241),'rlv_current':(255,201,56),'rlv_target':(179,98,218)}
    role_ids={'elcv':env.scenario.elcv_id,'rfv':env.scenario.rfv_id,
              'rlv_current':env.scenario.rlv_current_id,'rlv_target':env.scenario.rlv_target_id}
    for ident,track in tracks.items():
        if ident in (env.scenario.elcv_id,env.scenario.rfv_id):continue
        raw=env._sample(track,t)
        if raw is not None:
            x,y,_,_=raw;px,py=xy(x,y);draw.ellipse((px-5,py-5,px+5,py+5),fill=(185,187,180),outline='white')
    for role,ident in role_ids.items():
        if ident is None:continue
        if role in ('elcv','rfv') and ident in env.vehicles:
            vehicle=env.vehicles[ident];x,y=vehicle['lon'],vehicle['lat']
        else:
            raw=env._sample(tracks[ident],t)
            if raw is None:continue
            x,y=raw[0],raw[1]
        px,py=xy(x,y);color=palette[role]
        draw.rounded_rectangle((px-11,py-6,px+11,py+6),radius=3,fill=color,outline='white',width=2)
        draw.text((px+13,py-8),role.upper(),fill='white',stroke_width=1,stroke_fill='black')
    draw.rectangle((0,0,width,56),fill=(25,31,39));draw.text((20,16),f'SR-AIRL · NGSIM · {env.dynamics_mode}',fill='white')
    draw.text((20,height-30),f"t={t:.1f}s   tracking reward={info['tracking_reward']:.4f}   collision proxy={info['controlled_collision_proxy']}",fill='white')
    return np.asarray(image.convert('RGB'))


def attach_srairl_frame(directory,env,interval=.4):
    last=[0.]
    def update(active,info):
        now=time.monotonic()
        if info.get('done') or now-last[0]>=interval:
            render_srairl_frame(directory,active,info);last[0]=now
    env.frame_callback=update

    def step(self, action):
        result = self.env.step(action)
        return result


class EpisodeVideo:
    """Write evaluation frames directly to an MP4 without sending them to the browser."""
    def __init__(self, directory, episode, fps=10):
        import cv2
        self.cv2=cv2
        self.path=directory/'videos'/f'episode_{episode:04d}.mp4'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.writer=None;self.fps=fps

    def write(self, rgb):
        frame=np.asarray(rgb,dtype=np.uint8)
        if frame.ndim!=3 or frame.shape[2]<3:return
        frame=np.ascontiguousarray(frame[:,:,:3])
        if self.writer is None:
            h,w=frame.shape[:2]
            self.writer=self.cv2.VideoWriter(str(self.path),self.cv2.VideoWriter_fourcc(*'mp4v'),self.fps,(w,h))
            if not self.writer.isOpened():raise RuntimeError('无法创建评估 MP4 视频文件')
        self.writer.write(self.cv2.cvtColor(frame,self.cv2.COLOR_RGB2BGR))

    def close(self):
        if self.writer is not None:self.writer.release()


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
    env=ResearchVecEnv([lambda: Monitor(StudioEnv(research_config=env_config, view_mode=c.get('view_mode','topdown')))])
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
    video=None
    try:
        for episode in range(c['episodes']):
            if (directory/'STOP').exists(): break
            video=EpisodeVideo(directory,episode+1)
            obs,_=env.reset(seed=c['seed_start']+episode)
            video.write(env.pixels())
            discounted=0.;power=1.
            while True:
                action,_=model.predict(obs,deterministic=c['deterministic'])
                obs,reward,terminated,truncated,info=env.step(action)
                video.write(env.pixels())
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
                    video.close();video=None
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
        if video is not None:video.close()
        env.close();logger.close()


def build_srairl_scenarios(c, required_split):
    from env.core.types import InteractionScenario
    from datasets.srairl.interaction_validator import validate_interaction_scenario
    from datasets.importer import load_clip
    scenarios=[]
    for row in c['srairl_scenarios']:
        clip,manifest=load_clip(row['dataset_id'])
        if manifest['split']!=required_split:
            raise ValueError(f"SR-AIRL requires {required_split} clips, got {manifest['split']}")
        scenario=InteractionScenario.from_clip(clip,manifest,**row['roles'])
        scenario_validation=validate_interaction_scenario(scenario)
        if not scenario_validation['valid']:
            raise ValueError(f"invalid SR-AIRL interaction clip {scenario.scenario_id}: {scenario_validation['errors']}")
        scenarios.append(scenario)
    return scenarios


def create_srairl_metadrive_world(scenario,split,view_mode='topdown',reactive_background=True,dynamics_mode='legacy_kinematic'):
    """Create the shared MetaDrive road/world for synchronized legacy actors."""
    from env.research_config import load_config
    from studio.visualization import StudioEnv
    role_ids={scenario.elcv_id,scenario.rfv_id}
    duration=min(float(scenario.tracks[key]['states'][-1][0]) for key in
                 (scenario.elcv_id,scenario.rfv_id,scenario.rlv_current_id,scenario.rlv_target_id))
    tracks=[]
    for ident,track in scenario.tracks.items():
        tracks.append({**track,'id':str(ident),'ego':str(ident)==str(scenario.elcv_id)})
    clip={'schema_version':1,'source':scenario.source,'ego_id':scenario.elcv_id,
          'duration':duration,'lane_num':scenario.lane_num,'lane_width':scenario.lane_width,
          'road_length':scenario.road_length,'tracks':tracks,'geometry':'NGSIM recorded straight map'}
    cfg=load_config('configs/env/train.yaml' if split=='train' else 'configs/env/eval_id.yaml')
    e=cfg['environment'];e.update(map='',lane_num=scenario.lane_num,lane_width=scenario.lane_width,
        exit_length=max(float(scenario.road_length),80.),traffic_density=0.,horizon=max(100,int(round(duration/.1))+10))
    cfg['scenario'].update(type='straight',split=split,num_scenarios=1)
    # SR-AIRL ends on the shared clip boundary; a MetaDrive contact should be
    # measured without prematurely terminating the legacy reference rollout.
    cfg['termination'].update(terminate_on_collision=False,terminate_on_out_of_road=False)
    cfg['dataset']={'id':scenario.scenario_id,'source':scenario.source,'clip_sha256':'canonical-interaction-scenario'}
    if dynamics_mode=='metadrive_physics':
        from env.tasks.srairl_metadrive_world import SRAIRLMetaDriveWorld
        return SRAIRLMetaDriveWorld(scenario,clip,view_mode,reactive_background)
    return StudioEnv(research_config=cfg,view_mode=view_mode,dataset_clip=clip,controlled_track_ids=role_ids,
                     reactive_background=reactive_background)


def load_srairl_initial_state_samples(c):
    if c.get('initialization_protocol')!='ngsim_distribution':return []
    from datasets.importer import load_clip
    from datasets.srairl.interaction_validator import extract_initial_state_distribution
    from env.core.types import InteractionScenario
    samples=[]
    for item in c.get('srairl_scenarios',[]):
        clip,manifest=load_clip(item['dataset_id'])
        source=InteractionScenario.from_clip(clip,manifest,**item['roles'])
        for sample in extract_initial_state_distribution(source,samples=5,window_s=2.):
            samples.append({**sample,'dataset_id':item['dataset_id'],
                'recording_group':manifest.get('recording_group'),'clip_sha256':manifest.get('clip_sha256')})
    if not samples:raise ValueError('Selected NGSIM clip has no supported pre-change initial states')
    return samples


def build_srairl_envs(scenarios,c,config):
    from env.tasks.srairl_env import SRAIRLEnv
    if c.get('scene_source','ngsim')=='procedural':
        from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
        from env.tasks.srairl_interaction import SRAIRLInteractionScenario
        initial_state_samples=load_srairl_initial_state_samples(c)
        return [SRAIRLProceduralEnv(c['scenario'],c['lane_num'],c['traffic_density'],c.get('scene_seed',c['seed']),
            c['horizon'],c.get('srairl_view','topdown'),config=config,traffic_seed=c.get('traffic_seed'),
            interaction_scenario=SRAIRLInteractionScenario(**c.get('interaction_scenario',{})),
            rfv_response=c.get('rfv_response','policy'),initial_state_samples=initial_state_samples)]
    envs=[]
    for scenario in scenarios:
        world=create_srairl_metadrive_world(scenario,'train' if scenario.split=='train' else 'eval',c.get('srairl_view','topdown'),
                                            c.get('srairl_reactive_background',True),c.get('dynamics_mode','legacy_kinematic')) if c.get('metadrive_world',True) else None
        envs.append(SRAIRLEnv(scenario,config=config,dynamics_mode=c.get('dynamics_mode','legacy_kinematic'),metadrive_world=world))
    return envs


def save_srairl_input_manifest(directory,c):
    rows=[]
    if c.get('srairl_scenarios'):
        from datasets.importer import load_clip
        from env.core.types import InteractionScenario
        from datasets.srairl.interaction_validator import validate_interaction_scenario
        for item in c['srairl_scenarios']:
            clip,manifest=load_clip(item['dataset_id'])
            scenario=InteractionScenario.from_clip(clip,manifest,**item['roles'])
            validation=validate_interaction_scenario(scenario)
            row={'dataset_id':item['dataset_id'],'clip_sha256':manifest['clip_sha256'],
                         'source_sha256':manifest.get('source_sha256'), 'source':manifest['source'],
                         'split':manifest['split'],'recording_group':manifest.get('recording_group'),
                         'roles':item['roles'],'interaction_validation':validation}
            if c.get('initialization_protocol')=='ngsim_distribution':
                from datasets.srairl.interaction_validator import extract_initial_state_distribution
                row['initial_state_distribution']=extract_initial_state_distribution(scenario,samples=5,window_s=2.)
            rows.append(row)
    atomic_json(directory/'dataset_manifest.json',{'task':'SR-AIRL','dynamics_mode':c.get('dynamics_mode','legacy_kinematic'),'metadrive_world':bool(c.get('metadrive_world',True)),
        'scene_seed':c.get('scene_seed',c.get('seed')),'traffic_seed':c.get('traffic_seed'),'rl_seed':c.get('rl_seed',c.get('seed')),
        'reactive_background':bool(c.get('srairl_reactive_background',False)),
        'scene_source':c.get('scene_source','ngsim'),'scenario':c.get('scenario','straight'),
        'lane_num':c.get('lane_num'),'traffic_density':c.get('traffic_density'),
        'initialization_protocol':c.get('initialization_protocol','canonical'),
        'interaction_scenario':c.get('interaction_scenario'),'rfv_response':c.get('rfv_response','policy'),
        'expert_dataset_id':c.get('expert_dataset_id'),
        'expert_ngsim_scenarios':rows})


def train_srairl(directory,c):
    from datasets.srairl.expert_builder import ExpertDatasetBuilder
    from env.tasks.srairl_env import SRAIRLEnv
    from rl.srairl.config import SRAIRLConfig
    from rl.srairl.trainer import SRAIRLTrainer,SRAIRLTrainingConfig
    config=SRAIRLConfig.from_mapping()
    save_srairl_input_manifest(directory,c)
    scenarios=build_srairl_scenarios(c,'train')
    envs=build_srairl_envs(scenarios,c,config)
    if c.get('expert_dataset_id'):
        from datasets.srairl.expert_builder import load_transitions
        expert,_=load_transitions(ROOT/c['expert_dataset_id'])
    else:
        expert=[transition for scene in scenarios for transition in ExpertDatasetBuilder(config=config).build(scene)]
    training=SRAIRLTrainingConfig(episodes=c['episodes'],ppo_epochs=c['ppo_epochs'],seed=c.get('rl_seed',c['seed']),device=c['device'])
    trainer=SRAIRLTrainer(envs,expert,config=config,training_config=training,output_dir=directory)
    recorder=Recorder(directory)
    try:
        def report(row):
            step=int(row['global_steps'])
            values={f"train/{key}":float(value) for key,value in row.items()
                    if key not in ('episode','global_steps') and isinstance(value,(int,float,np.number,bool))}
            recorder.write(values,step=step)
            episode_row={key:(int(value) if key=='episode' else float(value) if isinstance(value,(int,float,np.number,bool)) else value)
                         for key,value in row.items()}
            with (directory/'episodes.jsonl').open('a') as f:f.write(json.dumps(episode_row,allow_nan=False)+'\n')
            atomic_json(directory/'status.json',dict(state='running',kind='train',step=int(row['episode'])+1,
                total=c['episodes'],episodes=int(row['episode'])+1,global_steps=step,device=str(trainer.device),
                dynamics_mode=c.get('dynamics_mode','legacy_kinematic')))
            return not (directory/'STOP').exists()
        trainer.train(callback=report)
        trainer.save()
        saved=json.loads((directory/'config.json').read_text());saved['dynamics_mode']=c.get('dynamics_mode','legacy_kinematic')
        saved['scene_seed']=c.get('scene_seed',c.get('seed'));saved['traffic_seed']=c.get('traffic_seed')
        saved['rl_seed']=c.get('rl_seed',c.get('seed'))
        saved['reactive_background']=bool(c.get('srairl_reactive_background',False))
        saved['scene_source']=c.get('scene_source','ngsim');saved['scenario']=c.get('scenario','straight')
        saved['lane_num']=c.get('lane_num');saved['traffic_density']=c.get('traffic_density');saved['horizon']=c.get('horizon')
        saved['interaction_scenario']=c.get('interaction_scenario');saved['rfv_response']=c.get('rfv_response','policy')
        saved['initialization_protocol']=c.get('initialization_protocol','canonical')
        saved['initial_state_samples']=[{k:v for k,v in row.items() if k in ('time_s','elcv_rfv_gap_m',
            'rlv_current_gap_m','rlv_target_gap_m','elcv_speed_mps','rfv_speed_mps',
            'rlv_current_speed_mps','rlv_target_speed_mps','source_lane','target_lane','dataset_id',
            'recording_group','clip_sha256')} for row in getattr(envs[0],'initial_state_samples',[])]
        atomic_json(directory/'config.json',saved)
        state='stopped' if (directory/'STOP').exists() else 'completed'
        summary=dict(algorithm='SR-AIRL',episodes=trainer.episode,global_steps=trainer.global_steps,
                     scenarios=[s.scenario_id for s in scenarios],expert_transitions=len(expert),
                     expert_source='MetaDrive saved demonstration' if c.get('expert_dataset_id') else 'NGSIM role-bound clips',
                     expert_dataset_id=c.get('expert_dataset_id'),
                     scene_source=c.get('scene_source','ngsim'),scenario=c.get('scenario','straight'),
                     lane_num=c.get('lane_num'),traffic_density=c.get('traffic_density'),
                     initialization_protocol=c.get('initialization_protocol','canonical'),
                     initial_state_sample_count=len(getattr(envs[0],'initial_state_samples',[])),
                     background_traffic_count=(envs[0]._last_info.get('traffic_vehicle_count') if hasattr(envs[0],'_last_info') else None),
                     dynamics_mode=c.get('dynamics_mode','legacy_kinematic'),metadrive_world=bool(c.get('metadrive_world',True)),
                     reactive_background=bool(c.get('srairl_reactive_background',False)),checkpoint=str(directory.relative_to(ROOT)))
        atomic_json(directory/'summary.json',summary)
        return dict(state=state,kind='train',step=trainer.episode,total=c['episodes'],episodes=trainer.episode,
                    global_steps=trainer.global_steps,device=str(trainer.device))
    finally:
        recorder.close()
        for env in envs:env.close()


def collect_srairl_expert(directory,c):
    """Generate reusable MetaDrive teacher trajectories for SR-AIRL training."""
    from datasets.srairl.expert_builder import save_transitions
    from datasets.srairl.metadrive_expert import collect_metadrive_expert
    from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
    from env.tasks.srairl_interaction import SRAIRLInteractionScenario
    env=SRAIRLProceduralEnv(c['scenario'],c['lane_num'],c['traffic_density'],c.get('scene_seed',c['seed']),
        c['horizon'],c['srairl_view'],traffic_seed=c.get('traffic_seed'),
        interaction_scenario=SRAIRLInteractionScenario(**c.get('interaction_scenario',{})))
    recorder=Recorder(directory)
    try:
        def report(state):
            if 'episode_step' in state:
                info=env._last_info
                recorder.write({'demo/speed_elcv_mps':float(env.world.role_vehicle('elcv').speed),
                    'demo/speed_rfv_mps':float(env.world.role_vehicle('rfv').speed),
                    'demo/distance_travelled_m':float(info.get('distance_travelled_m',0.)),
                    'demo/collision':float(info.get('metadrive_primary_collision',False)),
                    'demo/offroad':float(info.get('metadrive_out_of_road',False)),
                    'demo/lane_change_started':float(info.get('lane_change_started',False)),
                    'demo/lane_change_completed':float(info.get('lane_change_completed',False)),
                    'demo/target_gap_m':float(info.get('target_gap_m',0.)),
                    'demo/delta_v_rfv_mps':float(info.get('delta_v_rfv_mps',0.))},
                    step=int(state.get('global_steps',0)))
            atomic_json(directory/'status.json',state)
            return not (directory/'STOP').exists()
        transitions,episodes=collect_metadrive_expert(env,c['demo_episodes'],c['seed'],report)
        dataset_path=directory/'expert_transitions.npz'
        saved_dataset=ROOT/'datasets'/'srairl'/'metadrive'/f"{directory.name}.npz"
        pair_count=save_transitions(saved_dataset,transitions,dict(source='MetaDrive',scene_source='procedural',
            scenario=c['scenario'],lane_num=c['lane_num'],traffic_density=c['traffic_density'],
            horizon=c['horizon'],seed=c['seed'],episodes=len(episodes),controller=c['expert_controller'],
            interaction_scenario=c.get('interaction_scenario'),
            state_definition='SRAIRLObservationAdapter 16D frame stacked HISTORY_LEN times; action=[longitudinal_acceleration,lateral_acceleration] in m/s^2',
            dynamics_mode='metadrive_physics',background_traffic='MetaDrive IDM traffic manager',
            interaction_template='ELCV moves from source lane into the gap between RFV and RLV_target; RFV follows its leader and may respond'))
        import shutil
        shutil.copy2(saved_dataset,dataset_path)
        with (directory/'episodes.jsonl').open('a') as f:
            for row in episodes:f.write(json.dumps(row,allow_nan=False)+'\n')
        summary=dict(task='srairl_expert_collection',source='MetaDrive',scene_source='procedural',
            scenario=c['scenario'],lane_num=c['lane_num'],traffic_density=c['traffic_density'],
            horizon=c['horizon'],seed=c['seed'],episodes=len(episodes),transition_pairs=pair_count,
            interaction_scenario=c.get('interaction_scenario'),
            transitions=len(transitions),expert_controller=c['expert_controller'],
            controller_interpretation='Rule-based ELCV gap acceptance and RFV headway-response teacher; a scripted demonstration source, not an optimal or human expert.',
            background_traffic='MetaDrive IDM traffic manager',dataset_file='expert_transitions.npz',
            reusable_dataset=str(saved_dataset.relative_to(ROOT)),
            dynamics_mode='metadrive_physics')
        atomic_json(directory/'summary.json',summary)
        state='stopped' if (directory/'STOP').exists() else 'completed'
        return dict(state=state,kind='train',step=len(episodes),total=c['demo_episodes'],episodes=len(episodes),
            global_steps=pair_count,device='cpu')
    finally:
        recorder.close();env.close()


def evaluate_srairl_procedural(directory,c,env,policies,training):
    """Evaluate the learned ELCV/RFV pair on a generated MetaDrive map."""
    device=policies[0].device
    recorder=Recorder(directory);steps=0;episode_return=0.;min_distance=float('inf');min_ttc=float('inf')
    collision=False;out_of_road=False;done=False;obs,_=env.reset(seed=c['seed']);info=env._last_info
    lane_change_started=False;lane_change_completed=False;rfv_delta_v=0.;rfv_min_accel=0.
    gap_start=float(env._initial_gap);gap_final=gap_start;lane_change_time=None
    video=EpisodeVideo(directory,1)
    try:
        video.write(render_srairl_frame(directory,env,info))
        while not done and not (directory/'STOP').exists():
            state=torch.as_tensor(obs['elcv'],dtype=torch.float32,device=device).reshape(1,-1)
            with torch.no_grad():
                action_e=policies[0].policy.actor(state).squeeze(0).cpu().numpy()
                action_f=policies[1].policy.actor(state).squeeze(0).cpu().numpy()
            obs,info=env.step({'elcv':action_e*training.action_scale,'rfv':action_f*training.action_scale})
            video.write(render_srairl_frame(directory,env,info))
            steps+=1;episode_return+=float(info['tracking_reward'])
            if info['minimum_distance_m'] is not None:min_distance=min(min_distance,float(info['minimum_distance_m']))
            if info.get('minimum_ttc_s') is not None:min_ttc=min(min_ttc,float(info['minimum_ttc_s']))
            collision|=bool(info['metadrive_primary_collision']);out_of_road|=bool(info['metadrive_out_of_road'])
            lane_change_started|=bool(info['lane_change_started']);lane_change_completed|=bool(info['lane_change_completed'])
            rfv_delta_v=float(np.linalg.norm(env.world.role_vehicle('rfv').velocity)-env._rfv_initial_speed)
            rfv_min_accel=min(rfv_min_accel,float(info['rfv_min_accel_mps2']))
            gap_final=float(info['target_gap_m']);lane_change_time=info.get('time_to_lane_change')
            done=bool(info['done'])
            recorder.write({'step/progress_reward':float(info['tracking_reward']),
                'step/minimum_vehicle_distance_m':float(info['minimum_distance_m'] or 0.),
                'step/distance_travelled_m':float(info['distance_travelled_m']),
                'step/collision':float(info['metadrive_primary_collision']),
                'step/out_of_road':float(info['metadrive_out_of_road']),
                'step/lane_change_started':float(info['lane_change_started']),
                'step/lane_change_completed':float(info['lane_change_completed']),
                'step/target_gap_m':float(info['target_gap_m']),
                'step/rfv_delta_v_mps':float(rfv_delta_v),
                **({'step/min_ttc_s':float(info['minimum_ttc_s'])} if info.get('minimum_ttc_s') is not None else {})},step=steps)
            atomic_json(directory/'status.json',dict(state='running',kind='eval',step=steps,total=c['horizon'],
                episodes=0,global_steps=steps,dynamics_mode='metadrive_physics',device=str(device),
                scene_source='procedural',scenario=c['scenario'],traffic_vehicle_count=info['traffic_vehicle_count']))
        completed=not (directory/'STOP').exists()
        row=dict(episode=1,scenario_id=f"metadrive-{c['scenario']}",scene_source='procedural',
            map=c['scenario'],steps=steps,progress_return=episode_return,
            distance_travelled_m=float(info['distance_travelled_m']),minimum_distance_m=None if not np.isfinite(min_distance) else min_distance,
            minimum_ttc_s=None if not np.isfinite(min_ttc) else min_ttc,
            collision=bool(collision),out_of_road=bool(out_of_road),
            survival_success=bool(completed and not collision and not out_of_road),
            success=bool(completed and info.get('merge_success',False) and not collision and not out_of_road),
            route_completion=float(info.get('route_completion',0.)),
            initialization_protocol=info.get('initialization_protocol','canonical'),
            initial_state_time_s=info.get('initial_state_time_s'),
            initial_state_dataset_id=info.get('initial_state_dataset_id'),
            traffic_vehicle_count=int(info['traffic_vehicle_count']),lane_num=c['lane_num'],
            traffic_density=c['traffic_density'],dynamics_mode='metadrive_physics',
            lane_change_started=lane_change_started,lane_change_completed=lane_change_completed,
            merge_success=bool(info.get('merge_success',False) and not collision and not out_of_road),
            time_to_lane_change_s=lane_change_time,initial_target_gap_m=gap_start,
            final_target_gap_m=gap_final,target_gap_change_m=gap_final-gap_start,
            delta_v_rfv_mps=rfv_delta_v,rfv_min_accel_mps2=rfv_min_accel,
            rfv_response=c.get('rfv_response','policy'))
        if completed:
            with (directory/'episodes.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            recorder.write({'evaluation/progress_return':episode_return,'evaluation/distance_travelled_m':row['distance_travelled_m'],
                'evaluation/collision_rate':float(collision),'evaluation/out_of_road_rate':float(out_of_road),
                'evaluation/survival_success_rate':float(row['survival_success']),
                'evaluation/success_rate':float(row['success']),
                'evaluation/route_completion':float(row['route_completion']),
                'evaluation/lane_change_success_rate':float(row['merge_success']),
                'evaluation/merge_success_rate':float(row['merge_success']),
                'evaluation/target_gap_change_m':float(row['target_gap_change_m']),
                'evaluation/delta_v_rfv_mps':float(row['delta_v_rfv_mps']),
                **({'evaluation/minimum_distance_m':row['minimum_distance_m']} if row['minimum_distance_m'] is not None else {}),
                **({'evaluation/minimum_ttc_s':row['minimum_ttc_s']} if row['minimum_ttc_s'] is not None else {})},step=steps)
        calibration_path=ROOT/'outputs/calibration/srairl_action_adapter_v2/summary.json'
        calibration=json.loads(calibration_path.read_text()) if calibration_path.is_file() else None
        calibration_passed=bool(calibration and calibration.get('equivalence_status')=='passed')
        summary=dict(algorithm='SR-AIRL',episodes=int(completed),global_steps=steps,
            scene_source='procedural',scenario=c['scenario'],lane_num=c['lane_num'],traffic_density=c['traffic_density'],
            initialization_protocol=c.get('initialization_protocol','canonical'),
            initial_state_time_s=row.get('initial_state_time_s'),
            initial_state_dataset_id=row.get('initial_state_dataset_id'),
            background_traffic_count=row['traffic_vehicle_count'],dynamics_mode='metadrive_physics',
            progress_return=episode_return if completed else None,distance_travelled_m=row['distance_travelled_m'] if completed else None,
            collision_rate=float(collision) if completed else None,out_of_road_rate=float(out_of_road) if completed else None,
            survival_success_rate=float(row['survival_success']) if completed else None,
            success_rate=float(row['success']) if completed else None,
            route_completion=float(row['route_completion']) if completed else None,
            minimum_distance_m=row['minimum_distance_m'] if completed else None,
            minimum_ttc_s=row['minimum_ttc_s'] if completed else None,
            lane_change_started=lane_change_started,lane_change_completed=lane_change_completed,
            lane_change_success_rate=float(row['merge_success']) if completed else None,
            merge_success_rate=float(row['merge_success']) if completed else None,
            time_to_lane_change_s=lane_change_time,initial_target_gap_m=gap_start,final_target_gap_m=gap_final,
            target_gap_change_m=gap_final-gap_start,delta_v_rfv_mps=rfv_delta_v,rfv_min_accel_mps2=rfv_min_accel,
            rfv_response=c.get('rfv_response','policy'),action_calibration=calibration,
            merge_route=(dict(elcv_initial_road='ramp: 1r1_2_ → 1r1_3_',
                rfv_initial_road='mainline: >>> → 1r0_0_',target_mainline_lane_index=2,
                downstream_route_shared=bool(info.get('merge_route_valid'))) if c['scenario']=='merge' else None),
            quantitative_transfer_evaluation_allowed=calibration_passed,
            experiment_status='quantitative' if calibration_passed else 'exploratory_only',
            interpretation=('Four-role lane-change interaction evaluation: ELCV, RFV, RLV_current and RLV_target are explicit; '
                'the two leader trajectories are fixed-speed and only the ELCV/RFV are policy controlled. '
                +('Action calibration passed.' if calibration_passed else 'Action calibration failed or is missing; MetaDrive physics results are exploratory only.')))
        atomic_json(directory/'summary.json',summary)
        return dict(state='stopped' if (directory/'STOP').exists() else 'completed',kind='eval',step=steps,
                    total=c['horizon'],episodes=int(completed),global_steps=steps,device=str(device))
    finally:
        video.close();recorder.close();env.close()


def evaluate_srairl(directory,c):
    from env.tasks.srairl_env import SRAIRLEnv
    from rl.srairl.config import SRAIRLConfig
    from rl.srairl.trainer import SRAIRLTrainingConfig,SRAIRLTrainer
    path=Path(c['model_path']).resolve()
    if not path.is_dir() or path.parent!=(ROOT/'models').resolve() and (ROOT/'models'/'srairl').resolve() not in path.parents:
        raise ValueError('SR-AIRL model path is outside models/')
    saved=json.loads((path/'config.json').read_text())
    saved_dynamics=saved.get('dynamics_mode','legacy_kinematic')
    if saved_dynamics!=c.get('dynamics_mode',saved_dynamics):
        raise ValueError(f"评估动力学 {c.get('dynamics_mode')} 与模型训练动力学 {saved_dynamics} 不一致")
    config=SRAIRLConfig.from_mapping(saved['srairl'])
    train_cfg_values=dict(saved['training']);train_cfg_values['device']=c['device']
    training=SRAIRLTrainingConfig(**train_cfg_values)
    policies=SRAIRLTrainer.load_policies(path,config,training,device=c['device'] if c['device']!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if c.get('scene_source')=='procedural':
        from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
        from env.tasks.srairl_interaction import SRAIRLInteractionScenario
        env=SRAIRLProceduralEnv(c['scenario'],c['lane_num'],c['traffic_density'],c.get('scene_seed',c['seed']),
            c['horizon'],c.get('srairl_view','topdown'),config=config,traffic_seed=c.get('traffic_seed'),
            interaction_scenario=SRAIRLInteractionScenario(**c.get('interaction_scenario',{})),
            rfv_response=c.get('rfv_response','policy'),
            initial_state_samples=load_srairl_initial_state_samples(c))
        save_srairl_input_manifest(directory,c);return evaluate_srairl_procedural(directory,c,env,policies,training)
    scenarios=build_srairl_scenarios(c,'eval')
    save_srairl_input_manifest(directory,c)
    scenario=scenarios[0];env=build_srairl_envs(scenarios,c,config)[0]
    device=policies[0].device
    recorder=Recorder(directory);episode_rows=[];steps=0;returns=0.;errors=[];actor_errors={scenario.elcv_id:[],scenario.rfv_id:[]};collisions=0
    video=EpisodeVideo(directory,1)
    min_distance=float('inf');min_ttc=float('inf');metadrive_collision=False;metadrive_out_of_road=False
    try:
        obs,_=env.reset();episode_return=0.;episode_errors=[];collision=False
        video.write(render_srairl_frame(directory,env,env._last_info))
        while True:
            state=torch.as_tensor(obs['elcv'],dtype=torch.float32,device=device).reshape(1,-1)
            with torch.no_grad():
                action_e=policies[0].policy.actor(state).squeeze(0).cpu().numpy()
                action_f=policies[1].policy.actor(state).squeeze(0).cpu().numpy()
            obs,info=env.step({'elcv':action_e*training.action_scale,'rfv':action_f*training.action_scale})
            video.write(render_srairl_frame(directory,env,info))
            steps+=1;episode_return+=info['tracking_reward']
            episode_errors.extend(info['position_errors_m'].values())
            for ident,value in info['position_errors_m'].items():actor_errors.setdefault(ident,[]).append(float(value))
            collision=collision or info['collision_proxy_any']
            metadrive_collision=metadrive_collision or info['metadrive_primary_collision']
            metadrive_out_of_road=metadrive_out_of_road or info['metadrive_out_of_road']
            if info['min_interaction_clearance_m'] is not None:
                min_distance=min(min_distance,float(info['min_interaction_clearance_m']))
            if info['min_interaction_ttc_s'] is not None:min_ttc=min(min_ttc,float(info['min_interaction_ttc_s']))
            recorder.write({'step/tracking_reward':info['tracking_reward'],
                'step/position_rmse_m':float(np.mean(list(info['position_errors_m'].values()))),
                'step/controlled_separation_m':info['controlled_separation_m'],
                **({'step/min_interaction_ttc_s':info['min_interaction_ttc_s']} if info['min_interaction_ttc_s'] is not None else {})},step=steps)
            atomic_json(directory/'status.json',dict(state='running',kind='eval',step=steps,total=max(1,int(round((env.episode_end_time-env.start_time)/env.dt))),
                episodes=0,global_steps=steps,dynamics_mode=c.get('dynamics_mode','legacy_kinematic'),device=str(device)))
            if info['done'] or (directory/'STOP').exists():break
        if not (directory/'STOP').exists():
            rmse=float(np.mean(episode_errors)) if episode_errors else None
            fde=float(np.mean([values[-1] for values in actor_errors.values() if values])) if episode_errors else None
            row=dict(episode=1,scenario_id=scenario.scenario_id,steps=steps,tracking_return=episode_return,
                     mean_position_error_m=rmse,final_position_error_m=fde,
                     ade_m=rmse,fde_m=fde,minimum_distance_m=None if not np.isfinite(min_distance) else min_distance,
                     minimum_ttc_s=None if not np.isfinite(min_ttc) else min_ttc,
                     route_completion=float(info['route_completion']),
                     collision_proxy_any=bool(collision),controlled_pair_collision_proxy=bool(info['controlled_collision_proxy']),
                     success=bool(env.time>=env.episode_end_time-env.dt*.51),
                     metadrive_primary_collision=bool(metadrive_collision),
                     metadrive_out_of_road=bool(metadrive_out_of_road),
                     dynamics_mode=c.get('dynamics_mode','legacy_kinematic'),metadrive_world=bool(c.get('metadrive_world',True)),
                     traffic_reactive=bool(c.get('srairl_reactive_background',True)))
            episode_rows.append(row);errors=episode_errors;returns=episode_return;collisions=int(collision)
            with (directory/'episodes.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            recorder.write({'evaluation/tracking_return':episode_return,'evaluation/mean_position_error_m':rmse,
                            'evaluation/ade_m':rmse,'evaluation/fde_m':fde,
                            'evaluation/minimum_distance_m':row['minimum_distance_m'],
                            'evaluation/collision_proxy_rate':float(collision),
                            'evaluation/controlled_pair_collision_proxy_rate':float(info['controlled_collision_proxy']),
                            'evaluation/metadrive_primary_collision_rate':float(metadrive_collision),
                            'evaluation/metadrive_out_of_road_rate':float(metadrive_out_of_road),
                            'evaluation/success_rate':float(row['success'])},step=steps)
            if row['minimum_ttc_s'] is not None:recorder.write({'evaluation/minimum_ttc_s':row['minimum_ttc_s']},step=steps)
        calibration_path=ROOT/'outputs/calibration/srairl_action_adapter_v2/summary.json'
        calibration=json.loads(calibration_path.read_text()) if calibration_path.is_file() else None
        calibration_passed=bool(calibration and calibration.get('equivalence_status')=='passed')
        physics_mode=c.get('dynamics_mode')=='metadrive_physics'
        summary=dict(algorithm='SR-AIRL',episodes=len(episode_rows),global_steps=steps,
                     tracking_return=returns if episode_rows else None,
                     mean_position_error_m=float(np.mean(errors)) if errors else None,
                     ade_m=float(np.mean(errors)) if errors else None,fde_m=episode_rows[0]['fde_m'] if episode_rows else None,
                     minimum_distance_m=None if not np.isfinite(min_distance) else min_distance,
                     minimum_ttc_s=None if not np.isfinite(min_ttc) else min_ttc,
                     collision_proxy_rate=float(collisions) if episode_rows else None,
                     controlled_pair_collision_proxy_rate=float(episode_rows[0]['controlled_pair_collision_proxy']) if episode_rows else None,
                     metadrive_primary_collision_rate=float(metadrive_collision) if episode_rows else None,
                     metadrive_out_of_road_rate=float(metadrive_out_of_road) if episode_rows else None,
                     success_rate=float(episode_rows[0]['success']) if episode_rows else None,
                     scenario_id=scenario.scenario_id,dynamics_mode=c.get('dynamics_mode','legacy_kinematic'),metadrive_world=bool(c.get('metadrive_world',True)),
                     traffic_reactive=bool(c.get('srairl_reactive_background',True)),collision_metric='axis_aligned_proxy',
                     action_calibration=calibration if physics_mode else None,
                     quantitative_transfer_evaluation_allowed=bool(physics_mode and calibration_passed),
                     experiment_status=('quantitative' if calibration_passed else 'exploratory_only') if physics_mode else 'ngsim_trajectory_reproduction',
                     background_dynamics='bounded IDM-like longitudinal response around recorded NGSIM reference; not calibrated human-driver dynamics',
                     result_interpretation=(('Both SR-AIRL actors use MetaDrive vehicle physics. '+('Full-grid action calibration passed.' if calibration_passed else 'Action calibration failed or is missing; these results are exploratory only. ')+ 'Background response is not fitted to human-driver data.')
                         if physics_mode else
                         'MetaDrive NGSIM reconstruction with policy-controlled upstream kinematic actors and reactive background replay; not vehicle-physics closed loop.'))
        atomic_json(directory/'summary.json',summary)
        return dict(state='stopped' if (directory/'STOP').exists() else 'completed',kind='eval',step=steps,total=steps,
                    episodes=len(episode_rows),global_steps=steps,device=str(device))
    finally:
        video.close();recorder.close()
        env.close()


def train_reward_guide(directory, c):
    stage = ROOT/'rl/srairl/upstream/Step3_Reward_Guide'
    device = c['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    command = [sys.executable, '-u', str(stage/'main.py'),
               '--device', device, '--total_epoch', str(c['epochs']),
               '--batch_size', str(c['batch_size']), '--seed', str(c['seed']),
               '--data_path', str(ROOT/'rl/srairl/upstream/data/sample_data.pkl'),
               '--output_dir', str(directory), '--studio_dir', str(directory),
               '--srairl_model', c['srairl_model_path']]
    completed = subprocess.run(command, cwd=stage, check=False)
    if completed.returncode:
        raise RuntimeError(f"upstream reward-guide process exited with code {completed.returncode}")
    summary_path = directory/'summary.json'
    if not summary_path.is_file():
        raise RuntimeError('upstream reward-guide process exited without a selected checkpoint summary')
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    summary.update(seed=c['seed'],device=device,epochs=c['epochs'],batch_size=c['batch_size'],
                   srairl_model_id=c['srairl_model_id'])
    atomic_json(summary_path,summary)
    write_upstream_runtime_metadata(directory,'reward_guide',c,device)
    return dict(state='stopped' if (directory/'STOP').exists() else 'completed',kind='train',
                step=int(summary.get('best_epoch') or 0)+1,total=c['epochs'],episodes=0,device=device)


def train_srairl_predictor(directory, c):
    stage = ROOT/'rl/srairl/upstream/Step4_Predictor'
    device = c['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    command = [sys.executable, '-u', str(stage/'main.py'),
               '--device', device, '--total_epoch', str(c['epochs']),
               '--batch_size', str(c['batch_size']), '--seed', str(c['seed']),
               '--data_path', str(ROOT/'rl/srairl/upstream/data/sample_data.pkl'),
               '--output_dir', str(directory), '--studio_dir', str(directory),
               '--reward_guide_path', c['reward_guide_path']]
    completed = subprocess.run(command, cwd=stage, check=False)
    if completed.returncode:
        raise RuntimeError(f"upstream predictor process exited with code {completed.returncode}")
    summary_path=directory/'summary.json'
    if not summary_path.is_file():
        raise RuntimeError('upstream predictor exited without a validation-selected checkpoint summary')
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    summary.update(seed=c['seed'],device=device,epochs=c['epochs'],batch_size=c['batch_size'],
                   reward_guide_model_id=c['reward_guide_model_id'])
    atomic_json(summary_path,summary)
    write_upstream_runtime_metadata(directory,'predictor',c,device)
    return dict(state='stopped' if (directory/'STOP').exists() else 'completed',kind='train',
                step=int(summary.get('best_epoch') or 0)+1,total=c['epochs'],episodes=0,device=device)


def train_upstream_planner(directory, c):
    stage=ROOT/'rl/srairl/upstream/Step2_Planner'
    device=c['device']
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    data_path=str(ROOT/'rl/srairl/upstream/data/sample_data.pkl')
    goal_dir=directory/'planner_goal';dppo_dir=directory/'planner_dppo'
    atomic_json(directory/'status.json',dict(state='running',kind='train',step=0,total=c['goal_epochs'],
        episodes=0,device=device,phase='planner_goal'))
    goal_command=[sys.executable,'-u',str(stage/'train_goal.py'),'--device',device,
        '--data_path',data_path,'--output_dir',str(goal_dir),'--studio_dir',str(directory),
        '--epochs',str(c['goal_epochs']),'--seed',str(c['seed'])]
    if subprocess.run(goal_command,cwd=stage,check=False).returncode:
        raise RuntimeError('upstream goal-predictor training failed')
    if (directory/'STOP').exists():
        summary={'task':'planner','algorithm':'SR-AIRL Diffusion Planner',
            'goal_predictor':json.loads((directory/'planner_goal_summary.json').read_text(encoding='utf-8')),
            'diffusion_planner':None,'seed':c['seed'],'device':device,
            'interpretation':'Stopped after goal-predictor stage; no DPPO diffusion checkpoint was produced.'}
        atomic_json(directory/'summary.json',summary)
        write_upstream_runtime_metadata(directory,'planner',c,device)
        return dict(state='stopped',kind='train',step=0,total=c['dppo_steps'],episodes=0,device=device)
    atomic_json(directory/'status.json',dict(state='running',kind='train',step=0,total=c['dppo_steps'],
        episodes=0,device=device,phase='planner_dppo'))
    dppo_command=[sys.executable,'-u',str(stage/'train_dppo.py'),'--device',device,
        '--data_path',data_path,'--output_dir',str(dppo_dir),'--studio_dir',str(directory),
        '--steps',str(c['dppo_steps']),'--batch_size',str(c['batch_size']),'--seed',str(c['seed'])]
    if subprocess.run(dppo_command,cwd=stage,check=False).returncode:
        raise RuntimeError('upstream DPPO planner training failed')
    goal=json.loads((directory/'planner_goal_summary.json').read_text(encoding='utf-8'))
    dppo=json.loads((directory/'planner_dppo_summary.json').read_text(encoding='utf-8'))
    summary={'task':'planner','algorithm':'SR-AIRL Diffusion Planner',
        'goal_predictor':goal,'diffusion_planner':dppo,'dataset':data_path,'seed':c['seed'],
        'device':device,'upstream_revision':'a762fd1c4099ce27eda1fa4f26372c007fab0129',
        'interpretation':'Planner model training on the upstream fixed NGSIM trajectory dataset; closed-loop MetaDrive evaluation is not included in this stage.'}
    atomic_json(directory/'summary.json',summary)
    write_upstream_runtime_metadata(directory,'planner',c,device)
    return dict(state='stopped' if (directory/'STOP').exists() else 'completed',kind='train',
        step=int(dppo.get('steps',0)),total=c['dppo_steps'],episodes=0,device=device)


def write_upstream_runtime_metadata(directory, task, c, device):
    dataset=ROOT/'rl/srairl/upstream/data/sample_data.pkl'
    checkpoint=c.get('srairl_model_path') or c.get('reward_guide_path')
    inputs=[]
    if checkpoint:
        checkpoint_path=Path(checkpoint).resolve()
        checkpoint_files=sorted(checkpoint_path.glob('*.pt')) if checkpoint_path.is_dir() else [checkpoint_path]
        for item in checkpoint_files:
            if item.is_file():
                inputs.append({'path':str(item),'sha256':hashlib.sha256(item.read_bytes()).hexdigest()})
    versions={}
    for package in ('torch','numpy','pandas','einops','tensorboard'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    metadata={'task':task,'upstream_repository':'https://github.com/cixinachu/SR-AIRL',
        'upstream_revision':'a762fd1c4099ce27eda1fa4f26372c007fab0129','device':device,
        'seed':c['seed'],'dataset':{'path':str(dataset.relative_to(ROOT)),
            'sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
            'records':203,'train_records':150,'validation_records':53,
            'validation_note':'The upstream code calls this partition test; it is used for checkpoint selection here.'},
        'input_checkpoints':inputs,
        'versions':versions}
    atomic_json(directory/'runtime_metadata.json',metadata)


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);args=p.parse_args()
    directory=args.job.resolve();job=json.loads((directory/'request.json').read_text());c=job['settings'];kind=job['kind']
    torch.set_num_threads(1)
    atomic_json(directory/'status.json',dict(state='starting',kind=kind,step=0))
    try:
        if job.get('task') in ('planner','reward_guide','predictor'):
            result=(train_upstream_planner(directory,c) if job['task']=='planner' else
                    train_reward_guide(directory,c) if job['task']=='reward_guide' else
                    train_srairl_predictor(directory,c))
            atomic_json(directory/'status.json',result)
            return
        if job.get('task')=='srairl':
            c=job['settings']
            atomic_json(directory/'runtime_metadata.json',dict(task='SR-AIRL',dynamics_mode=c.get('dynamics_mode','legacy_kinematic'),metadrive_world=bool(c.get('metadrive_world',True)),
                source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                    for folder in ('env','rl','datasets','studio') for f in (ROOT/folder).rglob('*.py')}))
            if c.get('operation')=='expert_collect':result=collect_srairl_expert(directory,c)
            else:result=train_srairl(directory,c) if kind=='train' else evaluate_srairl(directory,c)
            atomic_json(directory/'status.json',result)
            return
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
            env=ViewEnv(StudioEnv(research_config=cfg, view_mode=c.get('view_mode','topdown'), preview=True),directory,.1,overview=True)
            try:
                env.reset(seed=c['seed_start'])
                env.frame(force=True)
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
