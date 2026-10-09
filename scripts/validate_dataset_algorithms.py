import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,subprocess,tempfile
from pathlib import Path
from datasets.importer import catalog
from studio.config import clean_request
from rl.algorithms import REGISTRY
records=[]
with tempfile.TemporaryDirectory(prefix='dataset-algorithms-') as d:
 root=Path(d)
 for alg in REGISTRY:
  for kind in ('train','eval'):
   m=next(x for x in catalog() if x['split']==kind and x['source']=='ngsim')
   c=clean_request(dict(algorithm=alg,data_source='ngsim',dataset_id=m['id'],lane_num=m['lane_num'],
       view_mode='topdown',timesteps=12,learning_starts=2,batch_size=2,buffer_size=100,hidden_size=16,
       horizon=10,log_freq=1,episodes=1,seed_start=1000 if kind=='train' else 0),kind)
   p=root/(alg+'_'+kind);p.mkdir()
   (p/'request.json').write_text(json.dumps(dict(kind=kind,settings=c,model_path=str(root/(alg+'_train')/'model.zip') if kind=='eval' else None)))
   r=subprocess.run([sys.executable,'studio/worker.py','--job',str(p)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
   assert r.returncode==0,r.stdout[-4000:]
   status=json.loads((p/'status.json').read_text());assert status['state']=='completed'
   assert (p/'dataset_clip.json').exists() and (p/'frame.jpg').stat().st_size>1000
   rows=[json.loads(line) for line in (p/'metrics.jsonl').read_text().splitlines()]
   if kind=='train':assert any('train/actor_loss' in row['values'] for row in rows)
   print(alg,kind,status,flush=True);records.append(dict(algorithm=alg,kind=kind,status=status))
Path('outputs/dataset_acceptance.json').write_text(json.dumps(records,indent=2))
