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
        draw.text((12,10),'MetaDrive · ELCV policy (blue) · RFV policy (orange) · RLV current (green) · OLV target (purple) · other replay (gray)',fill='white')
        target=directory/'frame.tmp.jpg';image.save(target,quality=82);target.replace(directory/'frame.jpg')
        return
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
    draw.rectangle((0,0,width,56),fill=(25,31,39));draw.text((20,16),'SR-AIRL · NGSIM trajectory view · legacy kinematic (not MetaDrive physics)',fill='white')
    draw.text((20,height-30),f"t={t:.1f}s   tracking reward={info['tracking_reward']:.4f}   collision proxy={info['controlled_collision_proxy']}",fill='white')
    target=directory/'frame.tmp.jpg';image.save(target,quality=82);target.replace(directory/'frame.jpg')


def attach_srairl_frame(directory,env,interval=.4):
    last=[0.]
    def update(active,info):
        now=time.monotonic()
        if info.get('done') or now-last[0]>=interval:
            render_srairl_frame(directory,active,info);last[0]=now
    env.frame_callback=update

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


def build_srairl_scenarios(c, required_split):
    from env.core.types import InteractionScenario
    from datasets.importer import load_clip
    scenarios=[]
    for row in c['srairl_scenarios']:
        clip,manifest=load_clip(row['dataset_id'])
        if manifest['split']!=required_split:
            raise ValueError(f"SR-AIRL requires {required_split} clips, got {manifest['split']}")
        scenario=InteractionScenario.from_clip(clip,manifest,**row['roles'])
        scenarios.append(scenario)
    return scenarios


def create_srairl_metadrive_world(scenario,split,view_mode='topdown',reactive_background=True):
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
    return StudioEnv(research_config=cfg,view_mode=view_mode,dataset_clip=clip,controlled_track_ids=role_ids,
                     reactive_background=reactive_background)


def build_srairl_envs(scenarios,c,config):
    from env.tasks.srairl_env import SRAIRLEnv
    envs=[]
    for scenario in scenarios:
        world=create_srairl_metadrive_world(scenario,'train' if scenario.split=='train' else 'eval',c.get('srairl_view','topdown'),
                                            c.get('srairl_reactive_background',True)) if c.get('metadrive_world',True) else None
        envs.append(SRAIRLEnv(scenario,config=config,dynamics_mode='legacy_kinematic',metadrive_world=world))
    return envs


def save_srairl_input_manifest(directory,c):
    from datasets.importer import load_clip
    rows=[]
    for item in c['srairl_scenarios']:
        _,manifest=load_clip(item['dataset_id'])
        rows.append({'dataset_id':item['dataset_id'],'clip_sha256':manifest['clip_sha256'],
                     'source_sha256':manifest.get('source_sha256'), 'source':manifest['source'],
                     'split':manifest['split'],'recording_group':manifest.get('recording_group'),
                     'roles':item['roles']})
    atomic_json(directory/'dataset_manifest.json',{'task':'SR-AIRL','dynamics_mode':'legacy_kinematic','metadrive_world':bool(c.get('metadrive_world',True)),
        'reactive_background':bool(c.get('srairl_reactive_background',True)),'scenarios':rows})


def train_srairl(directory,c):
    from datasets.srairl.expert_builder import ExpertDatasetBuilder
    from env.tasks.srairl_env import SRAIRLEnv
    from rl.srairl.config import SRAIRLConfig
    from rl.srairl.trainer import SRAIRLTrainer,SRAIRLTrainingConfig
    config=SRAIRLConfig.from_mapping()
    save_srairl_input_manifest(directory,c)
    scenarios=build_srairl_scenarios(c,'train')
    envs=build_srairl_envs(scenarios,c,config)
    for env in envs:attach_srairl_frame(directory,env)
    expert=[transition for scene in scenarios for transition in ExpertDatasetBuilder(config=config).build(scene)]
    training=SRAIRLTrainingConfig(episodes=c['episodes'],ppo_epochs=c['ppo_epochs'],seed=c['seed'],device=c['device'])
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
                dynamics_mode='legacy_kinematic'))
            return not (directory/'STOP').exists()
        trainer.train(callback=report)
        trainer.save()
        state='stopped' if (directory/'STOP').exists() else 'completed'
        summary=dict(algorithm='SR-AIRL',episodes=trainer.episode,global_steps=trainer.global_steps,
                     scenarios=[s.scenario_id for s in scenarios],expert_transitions=len(expert),
                     dynamics_mode='legacy_kinematic',metadrive_world=bool(c.get('metadrive_world',True)),checkpoint=str(directory.relative_to(ROOT)))
        atomic_json(directory/'summary.json',summary)
        return dict(state=state,kind='train',step=trainer.episode,total=c['episodes'],episodes=trainer.episode,
                    global_steps=trainer.global_steps,device=str(trainer.device))
    finally:
        recorder.close()
        for env in envs:env.close()


def evaluate_srairl(directory,c):
    from env.tasks.srairl_env import SRAIRLEnv
    from rl.srairl.config import SRAIRLConfig
    from rl.srairl.trainer import SRAIRLTrainingConfig,SRAIRLTrainer
    path=Path(c['model_path']).resolve()
    if not path.is_dir() or path.parent!=(ROOT/'models').resolve() and (ROOT/'models'/'srairl').resolve() not in path.parents:
        raise ValueError('SR-AIRL model path is outside models/')
    saved=json.loads((path/'config.json').read_text())
    config=SRAIRLConfig.from_mapping(saved['srairl'])
    train_cfg_values=dict(saved['training']);train_cfg_values['device']=c['device']
    training=SRAIRLTrainingConfig(**train_cfg_values)
    policies=SRAIRLTrainer.load_policies(path,config,training,device=c['device'] if c['device']!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    scenarios=build_srairl_scenarios(c,'eval')
    save_srairl_input_manifest(directory,c)
    scenario=scenarios[0];env=build_srairl_envs(scenarios,c,config)[0]
    attach_srairl_frame(directory,env)
    device=policies[0].device
    recorder=Recorder(directory);episode_rows=[];steps=0;returns=0.;errors=[];actor_errors={scenario.elcv_id:[],scenario.rfv_id:[]};collisions=0
    min_distance=float('inf');min_ttc=float('inf');metadrive_collision=False
    try:
        obs,_=env.reset();episode_return=0.;episode_errors=[];collision=False
        while True:
            state=torch.as_tensor(obs['elcv'],dtype=torch.float32,device=device).reshape(1,-1)
            with torch.no_grad():
                action_e=policies[0].policy.actor(state).squeeze(0).cpu().numpy()
                action_f=policies[1].policy.actor(state).squeeze(0).cpu().numpy()
            obs,info=env.step({'elcv':action_e*training.action_scale,'rfv':action_f*training.action_scale})
            steps+=1;episode_return+=info['tracking_reward']
            episode_errors.extend(info['position_errors_m'].values())
            for ident,value in info['position_errors_m'].items():actor_errors.setdefault(ident,[]).append(float(value))
            collision=collision or info['collision_proxy_any']
            metadrive_collision=metadrive_collision or info['metadrive_primary_collision']
            if info['min_interaction_clearance_m'] is not None:
                min_distance=min(min_distance,float(info['min_interaction_clearance_m']))
            if info['min_interaction_ttc_s'] is not None:min_ttc=min(min_ttc,float(info['min_interaction_ttc_s']))
            recorder.write({'step/tracking_reward':info['tracking_reward'],
                'step/position_rmse_m':float(np.mean(list(info['position_errors_m'].values()))),
                'step/controlled_separation_m':info['controlled_separation_m'],
                **({'step/min_interaction_ttc_s':info['min_interaction_ttc_s']} if info['min_interaction_ttc_s'] is not None else {})},step=steps)
            atomic_json(directory/'status.json',dict(state='running',kind='eval',step=steps,total=max(1,int(round((env.episode_end_time-env.start_time)/env.dt))),
                episodes=0,global_steps=steps,dynamics_mode='legacy_kinematic',device=str(device)))
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
                     dynamics_mode='legacy_kinematic',metadrive_world=bool(c.get('metadrive_world',True)),
                     traffic_reactive=bool(c.get('srairl_reactive_background',True)))
            episode_rows.append(row);errors=episode_errors;returns=episode_return;collisions=int(collision)
            with (directory/'episodes.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            recorder.write({'evaluation/tracking_return':episode_return,'evaluation/mean_position_error_m':rmse,
                            'evaluation/ade_m':rmse,'evaluation/fde_m':fde,
                            'evaluation/minimum_distance_m':row['minimum_distance_m'],
                            'evaluation/collision_proxy_rate':float(collision),
                            'evaluation/controlled_pair_collision_proxy_rate':float(info['controlled_collision_proxy']),
                            'evaluation/metadrive_primary_collision_rate':float(metadrive_collision),
                            'evaluation/success_rate':float(row['success'])},step=steps)
            if row['minimum_ttc_s'] is not None:recorder.write({'evaluation/minimum_ttc_s':row['minimum_ttc_s']},step=steps)
        summary=dict(algorithm='SR-AIRL',episodes=len(episode_rows),global_steps=steps,
                     tracking_return=returns if episode_rows else None,
                     mean_position_error_m=float(np.mean(errors)) if errors else None,
                     ade_m=float(np.mean(errors)) if errors else None,fde_m=episode_rows[0]['fde_m'] if episode_rows else None,
                     minimum_distance_m=None if not np.isfinite(min_distance) else min_distance,
                     minimum_ttc_s=None if not np.isfinite(min_ttc) else min_ttc,
                     collision_proxy_rate=float(collisions) if episode_rows else None,
                     controlled_pair_collision_proxy_rate=float(episode_rows[0]['controlled_pair_collision_proxy']) if episode_rows else None,
                     metadrive_primary_collision_rate=float(metadrive_collision) if episode_rows else None,
                     success_rate=float(episode_rows[0]['success']) if episode_rows else None,
                     scenario_id=scenario.scenario_id,dynamics_mode='legacy_kinematic',metadrive_world=bool(c.get('metadrive_world',True)),
                     traffic_reactive=bool(c.get('srairl_reactive_background',True)),collision_metric='axis_aligned_proxy',
                     background_dynamics='bounded IDM-like longitudinal response around recorded NGSIM reference; not calibrated human-driver dynamics',
                     result_interpretation='MetaDrive NGSIM reconstruction with two policy-controlled legacy-kinematic actors and reactive background replay; not calibrated MetaDrive vehicle physics or validated interactive traffic.')
        atomic_json(directory/'summary.json',summary)
        return dict(state='stopped' if (directory/'STOP').exists() else 'completed',kind='eval',step=steps,total=steps,
                    episodes=len(episode_rows),global_steps=steps,device=str(device))
    finally:
        recorder.close()
        env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);args=p.parse_args()
    directory=args.job.resolve();job=json.loads((directory/'request.json').read_text());c=job['settings'];kind=job['kind']
    torch.set_num_threads(1)
    atomic_json(directory/'status.json',dict(state='starting',kind=kind,step=0))
    try:
        if job.get('task')=='srairl':
            c=job['settings']
            atomic_json(directory/'runtime_metadata.json',dict(task='SR-AIRL',dynamics_mode='legacy_kinematic',metadrive_world=bool(c.get('metadrive_world',True)),
                source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                    for folder in ('env','rl','datasets','studio') for f in (ROOT/folder).rglob('*.py')}))
            result=train_srairl(directory,c) if kind=='train' else evaluate_srairl(directory,c)
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
