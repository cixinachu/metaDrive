"""End-to-end tests against a running Studio server (uses tiny training budgets)."""
import json,time,urllib.request,zipfile,io
from pathlib import Path
BASE='http://127.0.0.1:8765'
ROOT=Path(__file__).resolve().parents[1]

def api(path,data=None):
    req=urllib.request.Request(BASE+path,data=json.dumps(data).encode() if data else None,headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(req))

def wait(job):
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        r=api('/api/jobs/'+job)
        if r['status']['state'] not in ('running','starting'):
            if r['status']['state']!='completed':raise RuntimeError(json.dumps(r))
            return r
        time.sleep(.5)
    raise TimeoutError(job)

results={}
for algorithm in ('sac','sac_lagrangian','wcsac','wcsac_iqn'):
    c=dict(algorithm=algorithm,scenario='straight',timesteps=160,learning_starts=32,batch_size=16,buffer_size=500,
           hidden_size=32,n_quantiles=8,device='cpu',horizon=60,seed_start=1000,num_scenarios=2,
           checkpoint_freq=80,log_freq=10,frame_interval=.1)
    job=api('/api/jobs',{'kind':'train','settings':c})['id']
    r=wait(job)
    metrics=api('/api/jobs/'+job+'/metrics')['rows'];tags={k for row in metrics for k in row['values']}
    assert 'train/actor_loss' in tags
    if algorithm!='sac':assert 'train/safety_loss' in tags and 'train/multiplier' in tags
    p=ROOT/'models'/job
    assert (p/'frame.jpg').stat().st_size>1000
    models=api('/api/catalog')['models'];m=next(m for m in models if m['path']==str((p/'model.zip').relative_to(ROOT)))
    evaljob=api('/api/jobs',{'kind':'eval','settings':dict(algorithm=algorithm,model_id=m['id'],scenario='straight',seed_start=0,num_scenarios=2,episodes=2,horizon=40,log_freq=10,device='cpu')})['id']
    ev=wait(evaljob)
    assert ev['summary']['episodes']==2
    assert (ROOT/'results'/evaljob/'summary.json').exists()
    assert len({row['scene_seed'] for row in ev['episodes']})==2
    z=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(BASE+f'/api/jobs/{job}/download?type=tensorboard').read()))
    assert any('events.out.tfevents' in name for name in z.namelist())
    curve=urllib.request.urlopen(BASE+f'/api/jobs/{job}/download?type=curve&tag=train%2Factor_loss').read()
    assert b'train/actor_loss' in curve
    results[algorithm]=dict(train=job,eval=evaljob,train_steps=r['summary']['timesteps'],eval_episodes=2,
                            scalar_tags=len(tags),tensorboard_download=True,curve_download=True,frame=True)
    print(algorithm,results[algorithm],flush=True)
    (ROOT/'outputs/studio_validation.json').write_text(json.dumps(results,indent=2))
print('All four algorithms passed real MetaDrive train/save/reload/eval/download checks.',flush=True)
