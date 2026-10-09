"""Short multi-clip updates/save/load checks; temporary weights never become formal models."""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
import sys,json,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from stable_baselines3.common.monitor import Monitor
from datasets.experiment import DEFAULT_EXPERIMENT,NGSIMCollectionEnv
from datasets.importer import ROOT
from studio.config import DEFAULTS
from rl.algorithms import REGISTRY,create_model,load_model
from rl.vec_env import ResearchVecEnv


def main():
    original=json.loads(DEFAULT_EXPERIMENT.read_text());results=[]
    with tempfile.TemporaryDirectory(prefix='ngsim-collection-acceptance-') as tmp:
        tmp=Path(tmp);subset=dict(original,clips=sum([[c for c in original['clips'] if c['split']==s][:2] for s in ['train','validation']],[]))
        path=tmp/'subset.json';path.write_text(json.dumps(subset),encoding='utf-8')
        for alg in REGISTRY:
            c=dict(DEFAULTS,algorithm=alg,seed=0,timesteps=12,learning_starts=2,batch_size=2,buffer_size=100,hidden_size=16,n_quantiles=8,horizon=3,device='cpu')
            base=NGSIMCollectionEnv(c,'train',path);vec=ResearchVecEnv([lambda:Monitor(base)])
            try:
                model=create_model(c,vec);model.learn(total_timesteps=12,log_interval=1);assert model._n_updates>0
                model.save(tmp/(alg+'.zip'));updates=model._n_updates
            finally:vec.close()
            model=load_model(alg,tmp/(alg+'.zip'),device='cpu');env=NGSIMCollectionEnv(c,'validation',path);seen=[]
            try:
                for _ in range(2):
                    obs,_=env.reset();seen.append(env.current['clip_id'])
                    for step in range(3):
                        action,_=model.predict(obs,deterministic=True);obs,r,term,trunc,info=env.step(action)
                        assert np.isfinite(obs).all() and 0<=info['cost']<=1
                        if term or trunc:break
                assert len(set(seen))==2
                try:env.reset()
                except RuntimeError:pass
                else:raise AssertionError('Evaluation repeated exhausted collection')
            finally:env.close()
            results.append(dict(algorithm=alg,training_steps=12,gradient_updates=updates,eval_clips=seen,passed=True));print(results[-1],flush=True)
    (ROOT/'outputs/ngsim_algorithm_acceptance.json').write_text(json.dumps(dict(experiment=original['version'],selection_config_sha256=original['config_sha256'],checks=results),indent=2),encoding='utf-8')
if __name__=='__main__':main()
