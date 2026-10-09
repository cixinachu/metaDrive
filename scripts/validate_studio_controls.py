import urllib.request,urllib.error,json,time,io,zipfile
from pathlib import Path
from PIL import Image
BASE='http://127.0.0.1:8765'
ROOT=Path(__file__).resolve().parents[1]
def api(path,data=None):
 req=urllib.request.Request(BASE+path,data=json.dumps(data).encode() if data is not None else None,headers={'Content-Type':'application/json'})
 return json.load(urllib.request.urlopen(req))
def wait(j):
 for _ in range(240):
  d=api('/api/jobs/'+j)
  if d['status']['state'] not in ('starting','running'):return d
  time.sleep(.25)
 raise TimeoutError(j)
report={}
for scenario in ('procedural','straight','merge','intersection','t_junction','roundabout'):
 j=api('/api/jobs',{'kind':'preview','settings':{'scenario':scenario,'seed_start':1000,'num_scenarios':1}})['id']
 d=wait(j);assert d['status']['state']=='completed',d
 img=Image.open(io.BytesIO(urllib.request.urlopen(BASE+'/api/jobs/'+j+'/frame').read()))
 assert img.size==(800,800)
 report[scenario]=j
 print(scenario,j,flush=True)
j=api('/api/jobs',{'kind':'train','settings':dict(algorithm='wcsac_iqn',scenario='straight',device='cpu',timesteps=100000,learning_starts=8,batch_size=8,hidden_size=32,n_quantiles=8,buffer_size=1000,log_freq=10)})['id']
try:
 api('/api/jobs',{'kind':'preview','settings':{}})
 raise AssertionError('Concurrent task was accepted')
except urllib.error.HTTPError as e:assert e.code==400
for _ in range(100):
 d=api('/api/jobs/'+j)
 if d['status'].get('step',0)>20:break
 time.sleep(.2)
api('/api/jobs/'+j+'/stop',{})
d=wait(j);assert d['status']['state']=='stopped',d
assert ((ROOT/'models')/j/'model.zip').exists()
report['stop_save']=dict(id=j,status=d['status'])
for kind,settings in [('train',dict(seed_start=0)),('eval',dict(seed_start=1000)),('eval',dict(seed_start=0,num_scenarios=1,episodes=2))]:
 try:api('/api/jobs',{'kind':kind,'settings':settings});raise AssertionError('Invalid config accepted')
 except urllib.error.HTTPError as e:assert e.code==400
report['rejected_bad_split_and_duplicate_samples']=True
(ROOT/'outputs/studio_controls_validation.json').write_text(json.dumps(report,indent=2))
print('Scene previews, stop/save, concurrency and validation all passed.',flush=True)
