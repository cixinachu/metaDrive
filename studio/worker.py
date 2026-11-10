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


def _upstream_subprocess_env():
    """Keep the repository's native MetaDrive adapters importable from stage cwd."""
    environment=os.environ.copy()
    existing=environment.get('PYTHONPATH','')
    environment['PYTHONPATH']=str(ROOT)+(os.pathsep+existing if existing else '')
    return environment


from env.adapters.srairl_calibration import action_calibration_passed, calibration_summary_path


def srairl_experiment_protocol(settings,calibration_passed=False):
    source=settings.get('scene_source','ngsim')
    dynamics=settings.get('dynamics_mode','legacy_kinematic')
    reactive=bool(settings.get('srairl_reactive_background',False))
    scenario=settings.get('scenario','straight')
    if source=='ngsim' and dynamics=='legacy_kinematic' and not reactive:
        return {'id':'A','name':'NGSIM legacy fixed-replay reproduction','qualification':'reproduction'}
    if source=='ngsim' and dynamics=='legacy_kinematic' and reactive:
        return {'id':'B','name':'NGSIM reactive-background sensitivity','qualification':'sensitivity_only'}
    if source=='procedural' and dynamics=='metadrive_physics' and scenario in ('straight','curve','merge'):
        protocol='D' if scenario=='merge' else 'C'
        return {'id':protocol,'name':'MetaDrive ramp-merge transfer' if protocol=='D' else 'MetaDrive lane-change transfer',
                'qualification':'quantitative_transfer' if calibration_passed else 'exploratory_only'}
    return {'id':'custom','name':'Custom SR-AIRL experiment','qualification':'exploratory_only'}


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
    source='MetaDrive' if getattr(env,'scene_source','procedural')=='procedural' else str(env.scene_source)
    draw.rectangle((0,0,width,56),fill=(25,31,39));draw.text((20,16),f'SR-AIRL · {source} · {env.dynamics_mode}',fill='white')
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


def materialize_metadrive_dataset(directory, dataset_ids):
    """Resolve one or more UI-selected MetaDrive expert archives to one source."""
    ids=[dataset_ids] if isinstance(dataset_ids,str) else list(dataset_ids or [])
    if not ids:
        raise ValueError('select at least one MetaDrive expert dataset')
    paths=[(ROOT/item).resolve() for item in ids]
    from datasets.srairl.native_windows import native_dataset_info
    native_dataset_info(paths,require_current=True)
    paths=sorted(paths,key=lambda path:hashlib.sha256(path.read_bytes()).hexdigest())
    if len(paths)==1:
        return paths[0]
    from datasets.srairl.expert_builder import merge_expert_archives
    output=Path(directory)/'native_metadrive_experts.npz'
    merge_expert_archives(paths,output)
    return output


def metadrive_source_revision(dataset_ids):
    ids=[dataset_ids] if isinstance(dataset_ids,str) else list(dataset_ids or [])
    paths=sorted(((ROOT/item).resolve() for item in ids),key=lambda path:path.name)
    if not paths:raise ValueError('select at least one MetaDrive expert dataset')
    digest=hashlib.sha256()
    for path in paths:
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode('ascii'))
    return digest.hexdigest()


from rl.srairl.pipeline.provenance import checkpoint_set_revision


def srairl_reward_revision(directory):
    directory=Path(directory)
    return checkpoint_set_revision((directory/'discriminator_elcv.pt',directory/'discriminator_rfv.pt'),
                                   'temporal-dynamic-v2')


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
    if c.get('scene_source','procedural')=='procedural':
        from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
        from env.tasks.srairl_interaction import SRAIRLInteractionScenario
        initial_state_samples=load_srairl_initial_state_samples(c)
        scenario_types=c.get('training_scenarios') or [c['scenario']]
        envs=[]
        for index,scenario_type in enumerate(scenario_types):
            interaction=dict(c.get('interaction_scenarios',{}).get(
                scenario_type,c.get('interaction_scenario',{})))
            traffic_density=c.get('traffic_density_by_scenario',{}).get(scenario_type,c['traffic_density'])
            envs.append(SRAIRLProceduralEnv(scenario_type,c['lane_num'],traffic_density,
                (c.get('scene_seed',c['seed'])+index)%(2**32),c['horizon'],c.get('srairl_view','topdown'),
                config=config,traffic_seed=(c.get('traffic_seed')+index)%(2**32) if c.get('traffic_seed') is not None else None,
                interaction_scenario=SRAIRLInteractionScenario(**interaction),
                rfv_response=c.get('rfv_response','policy'),initial_state_samples=initial_state_samples))
        return envs
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
        'scene_source':c.get('scene_source','procedural'),'scenario':c.get('scenario','straight'),
        'lane_num':c.get('lane_num'),'traffic_density':c.get('traffic_density'),
        'initialization_protocol':c.get('initialization_protocol','canonical'),
        'interaction_scenario':c.get('interaction_scenario'),'rfv_response':c.get('rfv_response','policy'),
        'expert_dataset_id':c.get('expert_dataset_id'),
        'source_scenarios':rows})


def train_srairl(directory,c):
    from rl.srairl.config import SRAIRLConfig
    from rl.srairl.trainer import SRAIRLTrainer,SRAIRLTrainingConfig
    config=SRAIRLConfig.from_mapping()
    save_srairl_input_manifest(directory,c)
    if not c.get('expert_dataset_id'):
        raise ValueError('SR-AIRL training requires a validated MetaDrive expert dataset')
    scenarios=build_srairl_scenarios(c,'train')
    envs=[];env_factory=None;environment_count=1
    if c.get('scene_source','procedural')=='procedural':
        from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
        from env.tasks.srairl_interaction import SRAIRLInteractionScenario
        scenario_types=c.get('training_scenarios') or [c.get('scenario','straight')]
        initial_state_samples=load_srairl_initial_state_samples(c)
        environment_count=len(scenario_types)
        def env_factory(episode):
            index=int(episode)%len(scenario_types);scenario_type=scenario_types[index]
            interaction=dict(c.get('interaction_scenarios',{}).get(
                scenario_type,c.get('interaction_scenario',{})))
            traffic_density=c.get('traffic_density_by_scenario',{}).get(scenario_type,c['traffic_density'])
            return SRAIRLProceduralEnv(scenario_type,c['lane_num'],traffic_density,
                (c.get('scene_seed',c['seed'])+int(episode))%(2**32),c['horizon'],c.get('srairl_view','topdown'),
                config=config,traffic_seed=(c.get('traffic_seed',c['seed'])+int(episode))%(2**32),
                interaction_scenario=SRAIRLInteractionScenario(**interaction),
                rfv_response=c.get('rfv_response','policy'),initial_state_samples=initial_state_samples)
    else:
        envs=build_srairl_envs(scenarios,c,config)
    expert_path=None
    from datasets.srairl.expert_builder import load_transitions
    expert_path=materialize_metadrive_dataset(directory,c['expert_dataset_id'])
    expert,_=load_transitions(expert_path,split='train')
    training=SRAIRLTrainingConfig(episodes=c['episodes'],ppo_epochs=c['ppo_epochs'],
        lambda_rank=c.get('rank_lambda',.1),seed=c.get('rl_seed',c['seed']),device=c['device'])
    trainer=SRAIRLTrainer(envs,expert,config=config,training_config=training,output_dir=directory,
        env_factory=env_factory,environment_count=environment_count,dynamics_mode=c.get('dynamics_mode'))
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
        saved['reward_version']='temporal-dynamic-v2'
        saved['reward_model_revision']=srairl_reward_revision(directory)
        saved['training_scenarios']=c.get('training_scenarios') or [c.get('scenario','straight')]
        saved['rank_lambda']=c.get('rank_lambda',.1)
        if expert_path is not None:
            saved['source_dataset_revision']=metadrive_source_revision(c['expert_dataset_id'])
            saved['split_version']='scenario_grouped_seed_70_15_15_v1'
            saved['component_dataset_revision']=hashlib.sha256((saved['source_dataset_revision']+':srairl:11-frame-temporal-dynamic-v2').encode()).hexdigest()
            saved['expert_dataset_ids']=c.get('expert_dataset_id')
        saved['scene_seed']=c.get('scene_seed',c.get('seed'));saved['traffic_seed']=c.get('traffic_seed')
        saved['rl_seed']=c.get('rl_seed',c.get('seed'))
        saved['reactive_background']=bool(c.get('srairl_reactive_background',False))
        saved['scene_source']='procedural';saved['scenario']=c.get('scenario','straight')
        saved['lane_num']=c.get('lane_num');saved['traffic_density']=c.get('traffic_density');saved['horizon']=c.get('horizon')
        saved['traffic_density_by_scenario']=c.get('traffic_density_by_scenario',{})
        saved['interaction_scenario']=c.get('interaction_scenario');saved['rfv_response']=c.get('rfv_response','policy')
        saved['initialization_protocol']=c.get('initialization_protocol','canonical')
        saved['initial_state_samples']=[]
        atomic_json(directory/'config.json',saved)
        state='stopped' if (directory/'STOP').exists() else 'completed'
        summary=dict(algorithm='SR-AIRL',episodes=trainer.episode,global_steps=trainer.global_steps,
                     scenarios=c.get('training_scenarios') or [c.get('scenario','straight')],expert_transitions=len(expert),
                     expert_source='MetaDrive saved demonstration',
                     expert_dataset_id=c.get('expert_dataset_id'),
                     scene_source='procedural',scenario=c.get('scenario','straight'),
                     lane_num=c.get('lane_num'),traffic_density=c.get('traffic_density'),
                     traffic_density_by_scenario=c.get('traffic_density_by_scenario',{}),
                     initialization_protocol=c.get('initialization_protocol','canonical'),
                     initial_state_sample_count=len(getattr(trainer.last_env,'initial_state_samples',[])),
                     background_traffic_count=(trainer.last_env._last_info.get('traffic_vehicle_count') if hasattr(trainer.last_env,'_last_info') else None),
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
    from datasets.srairl.metadrive_expert import collect_metadrive_expert,save_raw_episode
    from datasets.srairl.teacher_controller import TEACHER_VERSION
    c={**c,'expert_controller':TEACHER_VERSION}
    from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
    from env.tasks.srairl_interaction import SRAIRLInteractionScenario
    recorder=Recorder(directory)
    try:
        transitions=[];episodes=[];raw_episodes=[];accepted_raw=[]
        simulated_steps=0
        scenario_types=c.get('scenarios') or [c['scenario']]
        total_episodes=c['demo_episodes']*len(scenario_types)
        for scenario_index,scenario_type in enumerate(scenario_types):
            if (directory/'STOP').exists():break
            scene_seed=c['seed']+scenario_index*100_000
            atomic_json(directory/'status.json',dict(state='running',kind='train',phase='initializing',
                step=len(episodes),episodes=len(episodes),total=total_episodes,device='cpu',
                scenario=scenario_type,scene_seed=scene_seed))
            print(f'专家采集：{scenario_type}，密度 {c["traffic_density"]}，种子 {scene_seed}，计划 {c["demo_episodes"]} 回合',flush=True)
            interaction_values=c.get('interaction_scenarios',{}).get(
                scenario_type,c.get('interaction_scenario',{}))
            env=SRAIRLProceduralEnv(scenario_type,c['lane_num'],c['traffic_density'],
                (c.get('scene_seed',c['seed'])+scenario_index*100_000)%(2**32),
                c['horizon'],c['srairl_view'],
                traffic_seed=(c.get('traffic_seed',c['seed'])+scenario_index*100_000)%(2**32),
                interaction_scenario=(SRAIRLInteractionScenario(**interaction_values)
                    if interaction_values else None))
            try:
                last_episode_step=[None,0]
                def report(state):
                    nonlocal simulated_steps
                    state=dict(state)
                    if 'episode_step' in state:
                        local_episode=int(state.get('episodes',0))
                        if last_episode_step[0]!=local_episode:last_episode_step[:]=[local_episode,0]
                        current_step=int(state['episode_step'])
                        simulated_steps+=max(0,current_step-last_episode_step[1])
                        last_episode_step[1]=current_step
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
                            step=simulated_steps)
                    state.update(kind='train',device='cpu',phase='collecting',
                        step=len(episodes)+int(state.get('episodes',0)),
                        episodes=len(episodes)+int(state.get('episodes',0)),
                        total=total_episodes,global_steps=simulated_steps,scene_seed=scene_seed)
                    state['scenario']=scenario_type
                    state['scenario_index']=scenario_index
                    atomic_json(directory/'status.json',state)
                    return not (directory/'STOP').exists()
                current_transitions,current_episodes,current_raw=collect_metadrive_expert(
                    env,c['demo_episodes'],(c['seed']+scenario_index*100_000)%(2**32),report,vary_initial_state=True)
                # Persist each finished scene before starting the next one.
                for row,raw_episode in zip(current_episodes,current_raw):
                    index=len(raw_episodes)+1
                    accepted=bool(raw_episode.get('validation',{}).get('valid'))
                    relative=Path('expert_episodes' if accepted else 'failed_episodes')/f'episode_{index:05d}.npz'
                    save_raw_episode(directory/relative,raw_episode)
                    row.update(scenario=scenario_type,episode=index,episode_file=str(relative))
                    with (directory/'episodes.jsonl').open('a') as stream:
                        stream.write(json.dumps(row,allow_nan=False)+'\n')
                    raw_episodes.append(raw_episode)
                    if accepted:accepted_raw.append(raw_episode)
                transitions.extend(current_transitions);episodes.extend(current_episodes)
                print(f'{scenario_type} 已结束：尝试 {len(current_episodes)}，合格 {sum(bool(row["accepted"]) for row in current_episodes)}',flush=True)
            finally:env.close()
        if not transitions:
            atomic_json(directory/'summary.json',dict(task='srairl_expert_collection',source='MetaDrive',
                scenario=c['scenario'],episodes=len(episodes),accepted_episodes=0,
                rejected_episodes=len(episodes),episode_records='failed_episodes/'))
            if (directory/'STOP').exists():
                return dict(state='stopped',kind='train',step=len(episodes),total=total_episodes,
                    episodes=len(episodes),global_steps=simulated_steps,device='cpu')
            raise RuntimeError('没有示范回合通过验收；不合格原始回合已保存在 failed_episodes/')
        dataset_path=directory/'expert_transitions.npz'
        saved_dataset=ROOT/'datasets'/'srairl'/'metadrive'/f"{directory.name}.npz"
        pair_count=save_transitions(saved_dataset,transitions,dict(source='MetaDrive',scene_source='procedural',
            scenario=c['scenario'],scenarios=scenario_types,lane_num=c['lane_num'],traffic_density=c['traffic_density'],
            horizon=c['horizon'],seed=c['seed'],episodes=len(episodes),controller=c['expert_controller'],
            interaction_scenario=c.get('interaction_scenario'),
            state_definition='SRAIRLObservationAdapter 16D frame stacked HISTORY_LEN times; action=[longitudinal_acceleration,lateral_acceleration] in m/s^2',
            dynamics_mode='metadrive_physics',background_traffic='MetaDrive IDM traffic manager',
            trajectory_definition='Global world coordinates [x,y,vx,vy] for ELCV, RFV, RLV_current, RLV_target; 20 real lane-tracking history frames and at least 100 real future frames after completed lane change',
            action_channels='desired physical acceleration [a_long,a_lat], normalized MetaDrive [steering,throttle_brake], measured physical [a_long,a_lat]',
            calibrated_desired_acceleration_domain_mps2={'longitudinal':[-1.5,1.5],'lateral':[-1.5,1.5]},
            interaction_template='ELCV moves from source lane into the gap between RFV and RLV_target; RFV follows its leader and may respond'),
            raw_episodes=accepted_raw)
        import shutil
        shutil.copy2(saved_dataset,dataset_path)
        summary=dict(task='srairl_expert_collection',source='MetaDrive',scene_source='procedural',
            scenario=c['scenario'],scenarios=scenario_types,lane_num=c['lane_num'],traffic_density=c['traffic_density'],
            horizon=c['horizon'],seed=c['seed'],episodes=len(episodes),accepted_episodes=len(accepted_raw),
            rejected_episodes=len(episodes)-len(accepted_raw),transition_pairs=pair_count,
            interaction_scenario=c.get('interaction_scenario'),
            transitions=len(transitions),expert_controller=c['expert_controller'],
            simulated_interaction_steps=simulated_steps,
            teacher_outcome_counts={name:sum(row.get('teacher_outcome','unknown')==name for row in episodes)
                for name in sorted({row.get('teacher_outcome','unknown') for row in episodes})},
            episodes_by_scenario={name:dict(attempted=sum(row['scenario']==name for row in episodes),
                accepted=sum(row['scenario']==name and row['accepted'] for row in episodes),
                rejected=sum(row['scenario']==name and not row['accepted'] for row in episodes)) for name in scenario_types},
            controller_interpretation='Rule-based ELCV gap acceptance and RFV headway-response teacher; a scripted demonstration source, not an optimal or human expert.',
            background_traffic='MetaDrive IDM traffic manager',dataset_file='expert_transitions.npz',
            reusable_dataset=str(saved_dataset.relative_to(ROOT)),
            dynamics_mode='metadrive_physics')
        atomic_json(directory/'summary.json',summary)
        state='stopped' if (directory/'STOP').exists() else 'completed'
        return dict(state=state,kind='train',step=len(episodes),total=total_episodes,episodes=len(episodes),
            global_steps=simulated_steps,device='cpu')
    finally:
        recorder.close()


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
        calibration_path=calibration_summary_path()
        calibration=json.loads(calibration_path.read_text()) if calibration_path.is_file() else None
        calibration_passed=action_calibration_passed(calibration,require_closed_loop=True)
        protocol=srairl_experiment_protocol(c,calibration_passed)
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
            experiment_protocol=protocol,
            experiment_status=protocol['qualification'],
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
        calibration_path=calibration_summary_path()
        calibration=json.loads(calibration_path.read_text()) if calibration_path.is_file() else None
        calibration_passed=action_calibration_passed(calibration,require_closed_loop=True)
        physics_mode=c.get('dynamics_mode')=='metadrive_physics'
        protocol=srairl_experiment_protocol(c,calibration_passed)
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
                     quantitative_transfer_evaluation_allowed=bool(protocol['qualification']=='quantitative_transfer'),
                     experiment_protocol=protocol,
                     experiment_status=protocol['qualification'],
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
    from env.tasks.srairl_interaction import REFERENCE_FRAME_VERSION
    reward_config=json.loads((Path(c['srairl_model_path'])/'config.json').read_text())
    if reward_config.get('reference_frame_version')!=REFERENCE_FRAME_VERSION or reward_config.get('source_dataset_revision')!=metadrive_source_revision(c['dataset_id']):
        raise ValueError('Reward Guide requires a reward model trained on the same current MetaDrive dataset/frame')
    stage = ROOT/'rl/srairl/upstream/Step3_Reward_Guide'
    device = c['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_path=materialize_metadrive_dataset(directory,c['dataset_id']);c['_dataset_path']=str(dataset_path)
    command = [sys.executable, '-u', str(stage/'main.py'),
               '--device', device, '--total_epoch', str(c['epochs']),
               '--batch_size', str(c['batch_size']), '--seed', str(c['seed']),
               '--data_path', str(dataset_path),
               '--output_dir', str(directory), '--studio_dir', str(directory),
               '--srairl_model', c['srairl_model_path']]
    completed = subprocess.run(command, cwd=stage, check=False,env=_upstream_subprocess_env())
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
    from rl.srairl.pipeline.model_bundle import torch_load_metadata
    from env.tasks.srairl_interaction import REFERENCE_FRAME_VERSION
    guide_meta=torch_load_metadata(c['reward_guide_path']).get('metadata',{})
    if guide_meta.get('reference_frame_version')!=REFERENCE_FRAME_VERSION or guide_meta.get('source_dataset_revision')!=metadrive_source_revision(c['dataset_id']):
        raise ValueError('Predictor requires a guide trained on the same current MetaDrive dataset/frame')
    stage = ROOT/'rl/srairl/upstream/Step4_Predictor'
    device = c['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_path=materialize_metadrive_dataset(directory,c['dataset_id']);c['_dataset_path']=str(dataset_path)
    command = [sys.executable, '-u', str(stage/'main.py'),
               '--device', device, '--total_epoch', str(c['epochs']),
               '--batch_size', str(c['batch_size']), '--seed', str(c['seed']),
               '--data_path', str(dataset_path),
               '--output_dir', str(directory), '--studio_dir', str(directory),
               '--reward_guide_path', c['reward_guide_path']]
    completed = subprocess.run(command, cwd=stage, check=False,env=_upstream_subprocess_env())
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
    data_path=str(materialize_metadrive_dataset(directory,c['dataset_id']));c['_dataset_path']=data_path
    goal_dir=directory/'planner_goal';dppo_dir=directory/'planner_dppo'
    atomic_json(directory/'status.json',dict(state='running',kind='train',step=0,total=c['goal_epochs'],
        episodes=0,device=device,phase='planner_goal'))
    goal_command=[sys.executable,'-u',str(stage/'train_goal.py'),'--device',device,
        '--data_path',data_path,'--output_dir',str(goal_dir),'--studio_dir',str(directory),
        '--epochs',str(c['goal_epochs']),'--seed',str(c['seed'])]
    if subprocess.run(goal_command,cwd=stage,check=False,env=_upstream_subprocess_env()).returncode:
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
    if subprocess.run(dppo_command,cwd=stage,check=False,env=_upstream_subprocess_env()).returncode:
        raise RuntimeError('upstream DPPO planner training failed')
    goal=json.loads((directory/'planner_goal_summary.json').read_text(encoding='utf-8'))
    dppo=json.loads((directory/'planner_dppo_summary.json').read_text(encoding='utf-8'))
    summary={'task':'planner','algorithm':'SR-AIRL Diffusion Planner',
        'goal_predictor':goal,'diffusion_planner':dppo,'dataset':data_path,'seed':c['seed'],
        'device':device,'upstream_revision':'a762fd1c4099ce27eda1fa4f26372c007fab0129',
        'interpretation':'Planner model training uses the selected native MetaDrive expert dataset; this stage alone does not constitute closed-loop MetaDrive evaluation.'}
    atomic_json(directory/'summary.json',summary)
    write_upstream_runtime_metadata(directory,'planner',c,device)
    return dict(state='stopped' if (directory/'STOP').exists() else 'completed',kind='train',
        step=int(dppo.get('steps',0)),total=c['dppo_steps'],episodes=0,device=device)


def write_upstream_runtime_metadata(directory, task, c, device):
    dataset=Path(c['_dataset_path'] if c.get('_dataset_path') else
                 materialize_metadrive_dataset(directory,c['dataset_id'])).resolve()
    checkpoint=c.get('srairl_model_path') or c.get('reward_guide_path')
    inputs=[]
    if checkpoint:
        checkpoint_path=Path(checkpoint).resolve()
        checkpoint_files=sorted([*checkpoint_path.glob('*.pt'),*checkpoint_path.glob('*.pth')]) if checkpoint_path.is_dir() else [checkpoint_path]
        for item in checkpoint_files:
            if item.is_file():
                inputs.append({'path':str(item),'sha256':hashlib.sha256(item.read_bytes()).hexdigest()})
    versions={}
    for package in ('torch','numpy','pandas','einops','tensorboard'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    from datasets.srairl.native_windows import native_dataset_info
    dataset_info=native_dataset_info(dataset,require_current=True)
    source_revision=metadrive_source_revision(c['dataset_id'])
    split_version=dataset_info['split_protocol']
    component_revision=hashlib.sha256((source_revision+':'+task+':'+split_version+':native-window-v1').encode()).hexdigest()
    dependency_revisions={}
    if task=='reward_guide':
        dependency_revisions['reward_model_revision']=srairl_reward_revision(c['srairl_model_path'])
    elif task=='predictor':
        guide_path=Path(c['reward_guide_path']).resolve()
        dependency_revisions['reward_guide_revision']=checkpoint_set_revision([guide_path],'reward-guide-v1')
    metadata={'task':task,'upstream_repository':'https://github.com/cixinachu/SR-AIRL',
        'upstream_revision':'a762fd1c4099ce27eda1fa4f26372c007fab0129','device':device,
        'seed':c['seed'],'dataset':{'path':str(dataset.relative_to(ROOT)),
            'sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
            'episodes':dataset_info['episodes'],'windows':dataset_info['windows'],
            'windows_by_split':dataset_info['windows_by_split'],'split_protocol':dataset_info['split_protocol'],
            'validation_note':'Whole MetaDrive episodes are assigned to train/validation/test before windows are constructed.'},
        'reference_frame_version':__import__('env.tasks.srairl_interaction',fromlist=['REFERENCE_FRAME_VERSION']).REFERENCE_FRAME_VERSION,
        'road_geometry_version':__import__('env.tasks.srairl_interaction',fromlist=['ROAD_GEOMETRY_CONFIG']).ROAD_GEOMETRY_CONFIG['version'],
        'source_dataset_revision':source_revision,'component_dataset_revision':component_revision,
        'dataset_revision':source_revision,'split_version':split_version,
        **dependency_revisions,
        'input_checkpoints':inputs,
        'versions':versions}
    atomic_json(directory/'runtime_metadata.json',metadata)
    # Keep compatibility metadata inside the actual checkpoint payloads used by ModelBundle.
    for checkpoint_file in [*directory.rglob('*.pt'),*directory.rglob('*.pth')]:
        try:payload=torch.load(checkpoint_file,map_location='cpu',weights_only=True)
        except Exception:continue
        if not isinstance(payload,dict) or not isinstance(payload.get('metadata'),dict):continue
        relative_parent=checkpoint_file.parent.relative_to(directory)
        checkpoint_component=relative_parent.as_posix() if str(relative_parent)!='.' else task
        checkpoint_revision=hashlib.sha256((source_revision+':'+task+':'+checkpoint_component+':'+split_version+':native-window-v1').encode()).hexdigest()
        payload['metadata'].update(reference_frame_version=metadata['reference_frame_version'],road_geometry_version=metadata['road_geometry_version'],source_dataset_revision=source_revision,
            component_dataset_revision=checkpoint_revision,dataset_revision=source_revision,
            split_version=split_version,component_name=checkpoint_component,**dependency_revisions)
        torch.save(payload,checkpoint_file)


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);args=p.parse_args()
    directory=args.job.resolve();job=json.loads((directory/'request.json').read_text());c=job['settings'];kind=job['kind']
    torch.set_num_threads(1)
    atomic_json(directory/'status.json',dict(state='starting',kind=kind,step=0))
    try:
        if job.get('task')=='srairl' and kind=='preview':
            from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
            from env.tasks.srairl_interaction import SRAIRLInteractionScenario
            env=SRAIRLProceduralEnv(c['scenario'],c['lane_num'],c['traffic_density'],c['seed'],
                c['horizon'],c['srairl_view'],traffic_seed=c['traffic_seed'],
                interaction_scenario=SRAIRLInteractionScenario(**c['interaction_scenario']))
            try:
                env.reset(seed=c['seed'])
                pixels=render_srairl_frame(directory,env,env._last_info)
                Image.fromarray(pixels).save(directory/'frame.tmp.jpg',quality=85)
                (directory/'frame.tmp.jpg').replace(directory/'frame.jpg')
                atomic_json(directory/'status.json',dict(state='completed',kind='preview',step=0))
            finally:env.close()
            return
        if job.get('task')=='evaluation':
            from env.tasks.srairl_interaction import SRAIRLInteractionScenario
            from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
            from rl.srairl.config import SRAIRLConfig
            from rl.srairl.pipeline.model_bundle import SRAIRLModelBundle
            from tasks.interaction_evaluation_task import InteractionEvaluationTask
            c=job['settings'];paths=c['model_bundle_paths']
            bundle=SRAIRLModelBundle.from_checkpoints(
                planner_goal_path=paths['planner_goal_path'],
                planner_diffusion_path=paths['planner_diffusion_path'],
                reward_guide_path=paths['reward_guide_path'],
                predictor_path=paths['predictor_path'],
                reward_model_path=paths['reward_model_path'])
            atomic_json(directory/'model_bundle.json',bundle.to_manifest())
            task=InteractionEvaluationTask(
                model_bundle=bundle,
                calibration_summary=calibration_summary_path(),
                device=('cuda' if c.get('srairl_device')=='auto' and torch.cuda.is_available() else
                        'cpu' if c.get('srairl_device')=='auto' else c.get('srairl_device')),
                candidate_count=c['candidate_count'],
                replan_steps=c['replan_steps'])
            config=SRAIRLConfig.from_mapping()
            all_metrics=[]
            evaluation_started=time.monotonic()
            for episode in range(c['episodes']):
                env=SRAIRLProceduralEnv(
                    scenario_type=c['scenario'],lane_num=c['lane_num'],traffic_density=c['traffic_density'],
                    seed=c['scene_seed']+episode,horizon=c['horizon'],view_mode=c['srairl_view'],
                    config=config,traffic_seed=c['traffic_seed']+episode,
                    interaction_scenario=SRAIRLInteractionScenario(**c['interaction_scenario']),
                    rfv_response='policy',initial_state_samples=load_srairl_initial_state_samples(c))
                try:
                    output=directory/f'episode_{episode+1:04d}.json'
                    result=task.evaluate(env,seed=c['scene_seed']+episode,output_path=output,
                                         scenario_id=f"{c['scenario']}-{episode:04d}",episode_index=episode+1)
                    row=result['metrics'].to_dict();all_metrics.append(row)
                    with (directory/'episodes.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(row,allow_nan=False)+'\n')
                    values={
                        'evaluation/lane_change_success_rate':float(row['lane_change_success']),
                        'evaluation/collision_rate':float(row['collision']),
                        'evaluation/out_of_road_rate':float(row['out_of_road']),
                        'evaluation/route_completion':float(row['route_completion']),
                        'evaluation/candidate_count':float(row['candidate_count']),
                        'evaluation/selected_srairl_return':float(np.mean(row['selected_srairl_returns'])) if row['selected_srairl_returns'] else 0.,
                        'evaluation/planning_latency_s':float(np.mean(row['planning_latency_s'])) if row['planning_latency_s'] else 0.,
                        'evaluation/predictor_latency_s':float(np.mean(row['predictor_latency_s'])) if row['predictor_latency_s'] else 0.,
                    }
                    for key in ('minimum_ttc_s','minimum_distance_m','time_to_lane_change_s'):
                        if row.get(key) is not None:values['evaluation/'+key]=float(row[key])
                    for key,source in (('evaluation/delta_v_rfv_mps','rfv_delta_v_mps'),
                                       ('evaluation/target_gap_change_m','target_gap_change_m')):
                        values[key]=float(row[source])
                    with (directory/'metrics.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps({'step':episode+1,'elapsed':time.monotonic()-evaluation_started,
                                                 'values':values},allow_nan=False)+'\n')
                    atomic_json(directory/'status.json',dict(state='running',kind='eval',step=episode+1,
                        total=c['episodes'],episodes=episode+1,device=str(task.device),phase='closed_loop'))
                finally:
                    env.close()
            summary={'task':'evaluation','protocol':'closed_loop_metadrive_receding_horizon',
                'scenario':c['scenario'],'initialization_protocol':c['initialization_protocol'],
                'episodes':len(all_metrics),'candidate_count':c['candidate_count'],
                'replan_steps':c['replan_steps'],'seed':c['seed'],'model_bundle_id':c['model_bundle_id'],
                'metrics':all_metrics,'calibration_hash':task.calibration_hash,
                'success_rate':float(np.mean([r['lane_change_success'] for r in all_metrics])),
                'collision_rate':float(np.mean([r['collision'] for r in all_metrics])),
                'out_of_road_rate':float(np.mean([r['out_of_road'] for r in all_metrics])),
                'failure_counts':{name:sum(r.get('failure_type')==name for r in all_metrics)
                    for name in ('model_candidates_infeasible','tracker_failure','collision','out_of_road')},
                'test_note':'Each generated seed is evaluated once; scenes are not independent if they share a source recording.'}
            atomic_json(directory/'summary.json',summary)
            atomic_json(directory/'status.json',dict(state='completed',kind='eval',step=c['episodes'],
                total=c['episodes'],episodes=c['episodes'],device=str(task.device),protocol=summary['protocol']))
            return
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
