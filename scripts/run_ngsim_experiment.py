"""Train/evaluate the frozen multi-clip NGSIM experiment with any local algorithm."""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,csv,json,time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from datasets.experiment import NGSIMCollectionEnv,DEFAULT_EXPERIMENT
from datasets.importer import ROOT,digest
from rl.algorithms import REGISTRY,create_model,load_model
from rl.vec_env import ResearchVecEnv
from studio.config import DEFAULTS,atomic_json
from studio.worker import Recorder,empirical_cvar


def append_episode(folder,row):
    path=folder/'episodes.csv';exists=path.exists()
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(row));
        if not exists:w.writeheader()
        w.writerow(row)
    with (folder/'episodes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')


class RecordEpisodes(BaseCallback):
    def __init__(self,folder):super().__init__();self.folder=folder;self.n=0
    def _on_step(self):
        info=self.locals['infos'][0]
        if 'episode_safety' in info:
            self.n+=1
            row=dict(episode=self.n,step=self.num_timesteps,clip_id=info['experiment_clip'],event_class=info['event_class'],recording=info['recording'],**info['episode_safety'])
            append_episode(self.folder,row)
            for k,v in row.items():
                if isinstance(v,(int,float,bool)):self.logger.record('episode/'+k,v)
            self.logger.dump(self.num_timesteps)
        return not (self.folder/'STOP').exists()


class SaveCheckpoints(BaseCallback):
    def __init__(self,folder,algorithm,settings,experiment_sha,every):
        super().__init__();self.folder=folder;self.algorithm=algorithm;self.settings=settings;self.experiment_sha=experiment_sha;self.every=every;self.next_step=every
    def _on_step(self):
        while self.num_timesteps>=self.next_step:
            path=self.folder/f'checkpoint_{self.next_step:09d}.zip'
            self.model.save(path)
            atomic_json(path.with_suffix('.json'),dict(algorithm=self.algorithm,settings=self.settings,experiment=self.experiment_sha,sha256=digest(path),step=self.num_timesteps))
            self.next_step+=self.every
        return True


def clustered_intervals(rows,tail_fraction,replicates=2000,seed=8675309):
    """Recording-cluster bootstrap; repeated clips from a sampled cluster stay together."""
    groups={}
    for row in rows:groups.setdefault(row.get('recording_cluster',row['recording']),[]).append(row)
    names=sorted(groups)
    if len(names)<2:return dict(cluster_count=len(names),replicates=0,ci95={},note='Fewer than two recording clusters; interval not estimable.')
    rng=np.random.default_rng(seed); draws={k:[] for k in ('success_rate','collision_rate','out_of_road_rate','mean_cost','mean_discounted_cost','mean_route_completion','discounted_cost_cvar')}
    for _ in range(replicates):
        selected=rng.choice(names,size=len(names),replace=True)
        sample=[row for name in selected for row in groups[name]]
        values={
            'success_rate':float(np.mean([r['success'] for r in sample])),
            'collision_rate':float(np.mean([r['collision'] for r in sample])),
            'out_of_road_rate':float(np.mean([r['out_of_road'] for r in sample])),
            'mean_cost':float(np.mean([r['episode_risk_sum'] for r in sample])),
            'mean_discounted_cost':float(np.mean([r['discounted_cost'] for r in sample])),
            'mean_route_completion':float(np.mean([r['route_completion'] for r in sample])),
            'discounted_cost_cvar':empirical_cvar([r['discounted_cost'] for r in sample],tail_fraction),
        }
        for key,value in values.items():draws[key].append(value)
    return dict(cluster_count=len(names),cluster_ids=names,replicates=replicates,ci95={k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in draws.items()},note='Cluster bootstrap over recording blocks. With very few recording blocks these intervals are coarse and should be treated as descriptive.')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','eval']);p.add_argument('--algorithm',choices=REGISTRY,default='sac');p.add_argument('--model',type=Path)
    p.add_argument('--split',choices=['validation','test_ood'],default='validation');p.add_argument('--seed',type=int,default=0);p.add_argument('--timesteps',type=int,default=100000)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--settings',type=Path,help='JSON settings override; same names as Studio')
    p.add_argument('--experiment',type=Path,default=DEFAULT_EXPERIMENT);p.add_argument('--output',type=Path)
    p.add_argument('--clip-ids-file',type=Path,help='Optional JSON list of fixed clip IDs from the selected split (validation checkpoint panel)')
    p.add_argument('--checkpoint-every',type=int,default=25000,help='Training checkpoint interval in environment steps')
    args=p.parse_args()
    if args.mode=='eval' and (args.model is None or not args.model.is_file()):p.error('eval requires an existing --model')
    trained={}
    if args.mode=='eval' and args.model.with_suffix('.json').exists():
        meta=json.loads(args.model.with_suffix('.json').read_text(encoding='utf-8'))
        if meta.get('algorithm',args.algorithm)!=args.algorithm:p.error('--algorithm does not match checkpoint metadata')
        if meta.get('sha256') and digest(args.model)!=meta['sha256']:p.error('Checkpoint fingerprint mismatch')
        trained=meta.get('settings',{})
    c={**DEFAULTS,**trained,**(json.loads(args.settings.read_text()) if args.settings else {}), 'algorithm':args.algorithm,'seed':args.seed,'timesteps':args.timesteps,'device':args.device,'horizon':200}
    for key in ('reward_mode','cost_mode','gamma_cost','tail_fraction'):
        if key in trained and c[key]!=trained[key]:p.error(f'Evaluation {key} must match training')
    folder=args.output or ROOT/('models' if args.mode=='train' else 'results')/f'ngsim_{time.strftime("%Y%m%d_%H%M%S")}_{args.algorithm}_{args.mode}_s{args.seed}'
    folder.mkdir(parents=True,exist_ok=False)
    if args.mode=='train' and args.checkpoint_every<=0:p.error('--checkpoint-every must be positive')
    clip_ids=None
    if args.clip_ids_file:
        if args.mode!='eval':p.error('--clip-ids-file is only valid for eval')
        clip_ids=json.loads(args.clip_ids_file.read_text(encoding='utf-8'))
        if not isinstance(clip_ids,list):p.error('--clip-ids-file must contain a JSON list')
    env=NGSIMCollectionEnv(c,'train' if args.mode=='train' else args.split,args.experiment,clip_ids=clip_ids)
    c.update(data_source='ngsim',dataset_id=env.entries[0]['clip_id'],seed_start=1000 if args.mode=='train' else 0,num_scenarios=1,episodes=1)
    environment_config=dict(env.env.research_config)
    environment_config['dataset_collection']=dict(version=env.experiment['version'],split=env.split,clip_ids=[e['clip_id'] for e in env.entries],note='dataset is a representative clip; reset uses the collection')
    atomic_json(folder/'environment_config.json',environment_config)
    snapshot=dict(**env.experiment,clip_sha256={x['clip_id']:json.loads((ROOT/'datasets/processed'/x['clip_id']/'manifest.json').read_text())['clip_sha256'] for x in env.entries})
    atomic_json(folder/'dataset_collection.json',snapshot);atomic_json(folder/'settings.json',c)
    logger=configure(str(folder/'tensorboard'),['csv','tensorboard']);logger.output_formats.append(Recorder(folder))
    atomic_json(folder/'status.json',dict(state='running',kind=args.mode))
    try:
        if args.mode=='train':
            vec=ResearchVecEnv([lambda:Monitor(env)])
            model=create_model(c,vec);model.set_logger(logger);cb=RecordEpisodes(folder)
            ckpt=SaveCheckpoints(folder,args.algorithm,c,env.experiment['membership_manifest_sha256'],args.checkpoint_every)
            from stable_baselines3.common.callbacks import CallbackList
            model.learn(total_timesteps=c['timesteps'],callback=CallbackList([cb,ckpt]),log_interval=1);logger.dump(model.num_timesteps);model.save(folder/'model.zip')
            atomic_json(folder/'model.json',dict(algorithm=args.algorithm,settings=c,environment_config=environment_config,experiment=env.experiment['version'],membership_manifest_sha256=env.experiment['membership_manifest_sha256'],sha256=digest(folder/'model.zip'),step=model.num_timesteps))
            summary=dict(steps=model.num_timesteps,episodes=cb.n,split='train',algorithm=args.algorithm)
        else:
            model=load_model(args.algorithm,args.model,device=args.device);rows=[]
            for i in range(len(env.entries)):
                obs,_=env.reset();cost=0.;power=1.
                while True:
                    action,_=model.predict(obs,deterministic=True);obs,r,term,trunc,info=env.step(action);cost+=power*info['cost'];power*=c['gamma_cost']
                    if term or trunc:break
                row=dict(episode=i+1,clip_id=info['experiment_clip'],event_class=info['event_class'],recording=info['recording'],recording_cluster=env.current.get('recording_cluster',info['recording']),discounted_cost=cost,**info['episode_safety']);append_episode(folder,row);rows.append(row)
                for k,v in row.items():
                    if isinstance(v,(int,float,bool)):logger.record('episode/'+k,v)
                logger.dump(i+1);print(i+1,len(env.entries),row['clip_id'],flush=True)
            metrics=['success','collision','out_of_road','episode_reward','episode_risk_sum','route_completion','mean_speed','discounted_cost']
            strata={cls:{k:float(np.mean([r[k] for r in rows if r['event_class']==cls])) for k in metrics} for cls in sorted({r['event_class'] for r in rows})}
            cluster=clustered_intervals(rows,c['tail_fraction'])
            summary=dict(episodes=len(rows),split=args.split,algorithm=args.algorithm,model_sha256=digest(args.model),per_class=strata,
                         sample_mean={k:float(np.mean([r[k] for r in rows])) for k in metrics},
                         success_rate=float(np.mean([r['success'] for r in rows])),collision_rate=float(np.mean([r['collision'] for r in rows])),
                         out_of_road_rate=float(np.mean([r['out_of_road'] for r in rows])),mean_cost=float(np.mean([r['episode_risk_sum'] for r in rows])),
                         mean_discounted_cost=float(np.mean([r['discounted_cost'] for r in rows])),
                         mean_route_completion=float(np.mean([r['route_completion'] for r in rows])),
                         discounted_cost_cvar=empirical_cvar([r['discounted_cost'] for r in rows],c['tail_fraction']),
                         sample_discounted_cost_cvar=empirical_cvar([r['discounted_cost'] for r in rows],c['tail_fraction']),tail_fraction=c['tail_fraction'],
                         effective_tail_samples=len(rows)*c['tail_fraction'],cluster_bootstrap_95ci=cluster,
                         evaluation_clip_ids=[r['clip_id'] for r in rows],evaluation_recording_clusters=sorted({r['recording_cluster'] for r in rows}),
                         note='Fixed NGSIM trajectory-replay policy comparison. Stratified clips are not the natural NGSIM traffic distribution. Recorded background vehicles do not react to ego actions. Cluster-bootstrap intervals are descriptive; the number of recording blocks is small.')
        summary['membership_manifest_sha256']=env.experiment['membership_manifest_sha256']
        atomic_json(folder/'summary.json',summary);atomic_json(folder/'status.json',dict(state='completed',kind=args.mode));print(folder,flush=True)
    except Exception:
        atomic_json(folder/'status.json',dict(state='failed',kind=args.mode));raise
    finally:env.close();logger.close()
if __name__=='__main__':main()
