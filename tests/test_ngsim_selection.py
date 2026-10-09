import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.select_ngsim import stable_changes,materialize
from datasets.importer import load_clip

class SelectionTests(unittest.TestCase):
    def test_stable_lane_change_rejects_flicker(self):
        self.assertEqual(stable_changes(np.r_[np.ones(15),np.full(15,2)]),[15])
        self.assertEqual(stable_changes(np.r_[np.ones(15),2,np.ones(15)]),[])
        self.assertEqual(stable_changes(np.r_[np.ones(15),np.full(15,3)]),[])

    def fixture(self):
        rows=[]
        for ident,lateral,offset in [(1,6,0),(2,18,30)]:
            for i in range(201):
                rows.append(dict(vehicle_id=ident,global_time=100000+i*100,local_x=lateral,local_y=100+offset+i*.5,v_length=15,v_width=6,lane_id=1 if ident==1 else 2))
        c=dict(candidate_id='a'*24,start_ms=100000,end_ms=120000,ego_id=1,split='train',recording='i-80:123',event_class='car_following',location='i-80')
        return c,pd.DataFrame(rows),dict(lane_width_m=3.6576,version='test'),dict(sha256='test')

    def test_materialization_geometry_and_fingerprint(self):
        c,g,cfg,source=self.fixture()
        with tempfile.TemporaryDirectory() as tmp,patch('scripts.select_ngsim.STORE',Path(tmp)):
            ident=materialize(c,g,6,cfg,source,'config')
            clip,m=load_clip(ident,Path(tmp));self.assertEqual(m['experiment_split'],'train');self.assertEqual(clip['duration'],20)
            ego=next(t for t in clip['tracks'] if t['ego']);self.assertEqual(len(ego['states']),201)
            self.assertAlmostEqual(ego['states'][0][1],30);self.assertAlmostEqual(ego['states'][0][3],1.524)
            self.assertFalse(m['synthetic'])

    def test_invalid_background_and_initial_overlap_are_rejected(self):
        c,g,cfg,source=self.fixture()
        g.loc[(g.vehicle_id==2)&(g.global_time==110000),'local_y']+=1000
        with tempfile.TemporaryDirectory() as tmp,patch('scripts.select_ngsim.STORE',Path(tmp)):
            with self.assertRaisesRegex(ValueError,'background_position_jump'):materialize(c,g,6,cfg,source,'config')
        c,g,cfg,source=self.fixture();g.loc[g.vehicle_id==2,'local_x']=6;g.loc[g.vehicle_id==2,'local_y']-=30
        with tempfile.TemporaryDirectory() as tmp,patch('scripts.select_ngsim.STORE',Path(tmp)):
            with self.assertRaisesRegex(ValueError,'initial_ego_overlap'):materialize(c,g,6,cfg,source,'config')

if __name__=='__main__':unittest.main()
