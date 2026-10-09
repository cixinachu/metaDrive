"""Synthetic format fixtures; these are not real highD recordings."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from datasets.importer import import_clip, load_clip


def write(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


class ImportTests(unittest.TestCase):
    def test_highd_center_direction_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);store=p/'converted'
            write(p/'01_recordingMeta.csv',[dict(id=1,frameRate=25,upperLaneMarkings='0;3.5;7',lowerLaneMarkings='10;13.5;17')])
            write(p/'01_tracksMeta.csv',[dict(id=1,drivingDirection=1),dict(id=2,drivingDirection=2)])
            rr=[dict(frame=f,id=i,x=100-f*.4 if i==1 else f*.4,y=4.25 if i==1 else 10.75,width=4,height=2) for f in range(1,52) for i in (1,2)]
            write(p/'01_tracks.csv',rr)
            opts=dict(source='highd',path=str(p/'01_tracks.csv'),ego_id=1,split='train',duration=2,synthetic=True)
            m=import_clip(opts,store);clip,_=load_clip(m['id'],store)
            self.assertEqual(len(clip['tracks']),1)
            states=np.asarray(clip['tracks'][0]['states'])
            self.assertAlmostEqual(states[0,2],3.5)
            np.testing.assert_allclose(states[:,3],10,atol=1e-8)
            self.assertEqual(clip['tracks'][0]['length'],4)
            # Synthetic highD fixture also exercises the actual MetaDrive adapter.
            from unittest.mock import patch
            from env.research_config import load_config
            from env.safe_metadrive_env import SafeMetaDriveEnv
            cfg=load_config()
            cfg['environment'].update(map='',lane_num=clip['lane_num'],lane_width=clip['lane_width'],exit_length=clip['road_length'],traffic_density=0,horizon=10)
            cfg['dataset']=m
            with patch('datasets.importer.load_clip',return_value=(clip,m)):
                env=SafeMetaDriveEnv(research_config=cfg)
                try:
                    observation,_=env.reset(seed=1000)
                    self.assertEqual(observation.shape,(259,))
                    self.assertAlmostEqual(env.agent.velocity[0],10,places=4)
                    for _ in range(10):
                        _,_,done,truncated,info=env.step([0,0])
                        if done or truncated: break
                    self.assertTrue(done or truncated)
                finally: env.close()
            with self.assertRaisesRegex(ValueError,'另一数据划分'):import_clip({**opts,'split':'eval'},store)

    def test_ngsim_units_duplicates_gap_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);store=p/'converted'
            rr=[dict(Vehicle_ID=1,Global_Time=100000+i*100,Local_X=6,Local_Y=100+i*3,v_Length=15,v_Width=6,Lane_ID=1) for i in range(21)]
            write(p/'ng.csv',rr+[rr[0]])
            opts=dict(source='ngsim',path=str(p/'ng.csv'),ego_id=1,split='eval',duration=2,recording_group='synthetic',lane_count=2,synthetic=True)
            m=import_clip(opts,store);clip,_=load_clip(m['id'],store)
            t=clip['tracks'][0];self.assertEqual(m['duplicates_removed'],1)
            self.assertAlmostEqual(t['length'],4.572)
            self.assertAlmostEqual(t['states'][0][3],9.144)
            self.assertAlmostEqual(t['states'][0][2],1.5*3.6576-6*.3048)
            file=store/m['id']/'clip.json';file.write_text(file.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'校验'):load_clip(m['id'],store)
            write(p/'ng.csv',rr[:10]+rr[12:])
            with self.assertRaisesRegex(ValueError,'缺帧'):import_clip(opts,store)

    def test_ngsim_conflicting_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            rr=[dict(Vehicle_ID=1,Global_Time=i*100,Local_X=6,Local_Y=100+i*3,v_Length=15,v_Width=6,Lane_ID=1) for i in range(21)]
            write(p/'ng.csv',rr+[{**rr[0],'Local_Y':999}])
            with self.assertRaisesRegex(ValueError,'重复时间'):
                import_clip(dict(source='ngsim',path=str(p/'ng.csv'),ego_id=1,split='eval',recording_group='test'),p/'out')
