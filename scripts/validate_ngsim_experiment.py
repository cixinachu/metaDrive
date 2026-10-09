"""Validate frozen membership and exercise representative real MetaDrive episodes."""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
import sys,json,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from datasets.importer import ROOT,load_clip,digest
from datasets.experiment import DEFAULT_EXPERIMENT
from studio.config import clean_request,environment_config
from env.safe_metadrive_env import SafeMetaDriveEnv
from env.recorded_traffic import sample


def main():
    selection=json.loads(DEFAULT_EXPERIMENT.read_text());entries=selection['clips'];groups=collections.defaultdict(list)
    assert len({e['clip_id'] for e in entries})==len(entries)
    result=dict(experiment=selection['version'],source_sha256=selection['source_sha256'],clips=len(entries),counts=dict(collections.Counter(e['split'] for e in entries)),runtime=[])
    identities=collections.defaultdict(set)
    for e in entries:
        clip,m=load_clip(e['clip_id']);groups[(e['split'],e['event_class'])].append(e)
        assert m['source_sha256']['trajectories.csv']==selection['source_sha256']
        assert m['selection_config_sha256']==selection['config_sha256'] and m['selection_code_sha256']==selection['selection_code_sha256']
        assert m['experiment_split']==e['split'] and m['event_class']==e['event_class']
        identities[e['split']].add(m['recording_group'])
        ego=[t for t in clip['tracks'] if t['ego']];assert len(ego)==1 and len(ego[0]['states'])==201
        np.testing.assert_allclose(np.diff(np.array(ego[0]['states'])[:,0]),.1,atol=1e-8)
        assert clip['duration']==20 and np.isfinite(np.asarray(ego[0]['states'])).all()
    splits=list(identities)
    for i,s in enumerate(splits):
        for other in splits[i+1:]:assert not identities[s]&identities[other]
    for (split,cls),items in sorted(groups.items()):
        e=items[0];clip,_=load_clip(e['clip_id']);seed=1000 if split=='train' else 0
        c=clean_request(dict(data_source='ngsim',dataset_id=e['clip_id'],episodes=1,num_scenarios=1,seed_start=seed,horizon=10),'preview')
        env=SafeMetaDriveEnv(research_config=environment_config(c,'preview'))
        try:
            obs,_=env.reset(seed=seed);assert obs.shape==(259,) and np.isfinite(obs).all()
            ego=next(t for t in clip['tracks'] if t['ego']);np.testing.assert_allclose(env.agent.position,sample(ego,0)[:2],atol=1e-4)
            manager=env.engine.traffic_manager;max_error=0
            for step in range(10):
                obs,reward,term,trunc,info=env.step([0,0]);assert np.isfinite(obs).all() and np.isfinite(reward) and 0<=info['cost']<=1
                for tr in manager.tracks:
                    ref=sample(tr,(step+1)*.1)
                    if ref is not None:
                        vehicle=manager.active[tr['id']];max_error=max(max_error,float(np.max(np.abs(np.asarray(vehicle.position)-ref[:2]))))
                if term or trunc:break
            assert max_error<1e-3
            result['runtime'].append(dict(split=split,event_class=cls,clip_id=e['clip_id'],steps=step+1,collision=bool(info['collision']),out_of_road=bool(info['out_of_road']),replay_max_error_m=max_error))
            print(split,cls,step+1,max_error,flush=True)
        finally:env.close()
    result['all_membership_checks_passed']=True
    (ROOT/'outputs/ngsim_full_acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
