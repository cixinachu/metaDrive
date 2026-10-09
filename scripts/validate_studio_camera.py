"""Check that Studio camera modes preserve a fixed-action rollout (requires graphics)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from studio.visualization import StudioEnv
from studio.config import DEFAULTS,environment_config
cfg=environment_config({**DEFAULTS,'scenario':'straight'},'preview')
records=[]
for mode in ['topdown','bird3d','chase3d','first3d']:
 e=StudioEnv(cfg,mode);rows=[]
 try:
  o,_=e.reset(seed=1000);rows.append(np.r_[o,0,0])
  for t in range(30):
   o,r,d,tr,i=e.step([.01,.5]);rows.append(np.r_[o,r,i['cost']])
   if mode!='topdown' and t%10==0:e.camera_frame()
  records.append(np.asarray(rows))
 finally:e.close()
for r in records[1:]:np.testing.assert_allclose(r,records[0],rtol=0,atol=1e-6)
print('PASS: identical 259D observations, rewards and costs over 30 actions in all four modes')
