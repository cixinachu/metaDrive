"""Verify real downloaded NGSIM replay positions and policy-controlled ego."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from datasets.importer import catalog,load_clip
from env.recorded_traffic import sample
from env.safe_metadrive_env import SafeMetaDriveEnv
from studio.config import clean_request,environment_config

results=[]
for m in catalog():
    if m['source']!='ngsim' or m['synthetic']:continue
    clip,_=load_clip(m['id'])
    c=clean_request(dict(data_source='ngsim',dataset_id=m['id'],lane_num=m['lane_num'],episodes=1,
                        seed_start=1000 if m['split']=='train' else 0,horizon=10),'preview')
    e=SafeMetaDriveEnv(research_config=environment_config(c,'preview'))
    try:
        obs,_=e.reset(seed=c['seed_start']);assert obs.shape==(259,)
        ego=next(t for t in clip['tracks'] if t['ego'])
        np.testing.assert_allclose(e.agent.position,sample(ego,0)[:2],atol=1e-4)
        assert abs(e.agent.LENGTH-ego['length'])<1e-6
        manager=e.engine.traffic_manager
        assert all(not t['ego'] for t in manager.tracks)
        from metadrive.utils.pg.utils import ray_localization
        for x in range(15,int(clip['road_length'])-10,5):
            for lane in range(m['lane_num']):
                _,on=ray_localization((1,0),(x,lane*m['lane_width']+.2),e.engine,return_on_lane=True)
                assert on,(m['id'],x,lane)
        max_error=0
        for i in range(10):
            obs,r,done,truncated,info=e.step([0,0])
            for track in manager.tracks:
                ref=sample(track,(i+1)*.1)
                if ref is None:assert track['id'] not in manager.active;continue
                v=manager.active[track['id']]
                error=np.max(np.abs(np.asarray(v.position)-ref[:2]));max_error=max(error,max_error)
                assert error<1e-3
                np.testing.assert_allclose(v.velocity,ref[2:],atol=1e-5)
            if done or truncated:break
        assert np.linalg.norm(np.asarray(e.agent.position)-sample(ego,(i+1)*.1)[:2])>.01
        assert i==9 and not info['out_of_road'], 'Unexpected early termination / lane localization failure'
        assert np.isfinite(obs).all() and 0<=info['cost']<=1
        results.append(dict(id=m['id'],source=m['recording_group'],steps=i+1,active_traffic=info['active_traffic_count'],max_position_error_m=float(max_error)))
    finally:e.close()
print(json.dumps(results,indent=2))
