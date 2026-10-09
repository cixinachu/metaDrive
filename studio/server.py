"""Local-only dashboard. Simulations run in isolated subprocesses, one at a time."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import csv
import hashlib
import io
import json
import mimetypes
import subprocess
import threading
import time
import zipfile
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from studio.config import ROOT,RUNS,JOB_ROOTS,run_root,DEFAULTS,SCENARIOS,ALGORITHM_NAMES,clean_request,atomic_json

from rl.algorithms import metadata as algorithm_metadata
from datasets.importer import catalog as dataset_catalog, import_clip

from studio.log_cache import JsonlCache, tail_text

LOG_CACHE=JsonlCache()
LOCK=threading.RLock()
PROCESS=None
ACTIVE=None


def read_json(path, fallback=None):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,ValueError):return fallback


def model_catalog():
    result=[]
    paths=list((ROOT/'models').glob('*/*.zip'))+list(RUNS.glob('*/*.zip'))
    for p in sorted(paths,key=lambda p:p.stat().st_mtime,reverse=True):
        if '.tmp.' in p.name:continue
        meta=read_json(p.with_suffix('.json'),{})
        env=meta.get('environment_config') or read_json(p.parent/'environment_config.json',{})
        # Do not guess the architecture for models produced by this dashboard.
        if (p.parent/'request.json').exists() and 'algorithm' not in meta:continue
        result.append(dict(id=hashlib.sha256(str(p.relative_to(ROOT)).encode()).hexdigest()[:20],
             label=f'{meta.get("algorithm","sac")} · {p.parent.name}/{p.name}',algorithm=meta.get('algorithm','sac'),
             path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,settings=meta.get('settings',{}),environment_config=env))
    return result


def find_model(model_id):
    return next((m for m in model_catalog() if m['id']==model_id),None)


def job_dir(job_id):
    if not job_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in job_id):
        raise ValueError('无效任务编号')
    for root in JOB_ROOTS:
        p=root/job_id
        if p.is_dir() and (p/'request.json').exists():return p
    raise ValueError('任务不存在')


def records(path):
    if not path.exists():return []
    rows=[]
    for line in path.read_text().splitlines():
        try:rows.append(json.loads(line))
        except ValueError:pass # Last line may still be being written.
    return rows


def status(directory):
    s=read_json(directory/'status.json',{'state':'starting'})
    with LOCK:
        if ACTIVE==directory.name and PROCESS is not None and PROCESS.poll() is not None and s['state'] in ('running','starting'):
            s.update(state='failed',error='工作进程已退出，请查看运行日志')
        elif ACTIVE!=directory.name and s['state'] in ('running','starting'):
            s.update(state='interrupted',error='服务重启前的任务，未自动恢复')
    return s


def start_job(kind,raw):
    global PROCESS,ACTIVE
    if kind not in ('preview','train','eval'):raise ValueError('未知任务类型')
    c=clean_request(raw,kind)
    model=None
    if kind=='eval':
        model=find_model(c['model_id'])
        if not model:raise ValueError('请选择存在的模型')
        c['algorithm']=model['algorithm']
        trained=model['environment_config']
        if trained:
            for section,key,ui in [('environment','reward_mode','reward_mode'),('safety','cost_mode','cost_mode')]:
                if trained.get(section,{}).get(key,c[ui]) != c[ui]:raise ValueError('评估 reward/cost 模式必须与训练模型一致')
        settings=model['settings']
        for key in ('gamma_cost','tail_fraction'):
            if key in settings:c[key]=settings[key]
    with LOCK:
        if PROCESS is not None and PROCESS.poll() is None:
            previous=read_json(job_dir(ACTIVE)/'status.json',{})
            if previous.get('state') in ('completed','stopped','failed'):
                try:PROCESS.wait(timeout=5)
                except subprocess.TimeoutExpired:raise ValueError('上一个任务正在释放资源，请稍后再试。')
            else:raise ValueError('已有任务运行中。请先停止或等待完成。')
        name=f'{time.strftime("%Y%m%d_%H%M%S")}_{kind}_{time.time_ns()%1000000:06d}'
        directory=run_root(kind)/name;directory.mkdir(parents=True)
        atomic_json(directory/'request.json',dict(kind=kind,settings=c,model_path=str(ROOT/model['path']) if model else None))
        atomic_json(directory/'status.json',dict(state='starting',kind=kind,step=0))
        with (directory/'worker.log').open('w') as log:
            PROCESS=subprocess.Popen([sys.executable,'-u',str(ROOT/'studio/worker.py'),'--job',str(directory)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        ACTIVE=name
        return {'id':name}


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass

    def respond(self,data,status=200,content_type='application/json; charset=utf-8',filename=None):
        if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        if filename:self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
        self.end_headers()
        try:self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):pass

    def do_GET(self):
        try:self.get()
        except (ValueError,FileNotFoundError) as e:self.respond({'error':str(e)},404)
        except Exception as e:self.respond({'error':str(e)},500)

    def get(self):
        parsed=urlparse(self.path);path=parsed.path;q=parse_qs(parsed.query)
        if path=='/api/catalog':
            return self.respond(dict(defaults=DEFAULTS,scenarios=[dict(id=k,name=v[0]) for k,v in SCENARIOS.items()],algorithms=ALGORITHM_NAMES,algorithm_metadata=algorithm_metadata(),models=model_catalog(),datasets=dataset_catalog(),ngsim_scenarios=read_json(ROOT/'configs/datasets/ngsim_scenario_taxonomy.json',{})))
        if path=='/api/jobs':
            jobs=[]
            for p in sorted((p for root in JOB_ROOTS for p in root.glob('*') if p.is_dir() and (p/'request.json').exists()),key=lambda p:p.name,reverse=True):
                if p.is_dir():jobs.append(dict(id=p.name,request=read_json(p/'request.json',{}),status=status(p)))
            return self.respond(dict(active=ACTIVE,jobs=jobs))
        if path.startswith('/api/jobs/'):
            parts=path.split('/');directory=job_dir(parts[3]);action=parts[4] if len(parts)>4 else ''
            if action=='frame':
                p=directory/'frame.jpg'
                return self.respond(p.read_bytes(),content_type='image/jpeg')
            if action=='download':return self.download(directory,q)
            if action=='metrics':
                offset=int(q.get('offset',['0'])[0]);rows=LOG_CACHE.read(directory/'metrics.jsonl')
                return self.respond(dict(rows=rows[max(0,offset):],offset=len(rows)))
            log=directory/'worker.log'
            return self.respond(dict(id=directory.name,status=status(directory),request=read_json(directory/'request.json',{}),
                summary=read_json(directory/'summary.json'),episodes=LOG_CACHE.read(directory/'episodes.jsonl'),log=tail_text(log)))
        files={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
        if path in files:
            p=ROOT/'studio/static'/files[path]
            return self.respond(p.read_bytes(),content_type=mimetypes.guess_type(p.name)[0] or 'text/plain')
        raise ValueError('Not found')

    def download(self,directory,q):
        kind=q.get('type',['all'])[0]
        if kind in ('curve','metrics','episodes'):
            out=io.StringIO();writer=csv.writer(out)
            if kind=='episodes':
                rows=records(directory/'episodes.jsonl');keys=list(dict.fromkeys(k for r in rows for k in r));writer.writerow(keys)
                writer.writerows([[r.get(k) for k in keys] for r in rows])
            else:
                tag=q.get('tag',[None])[0];writer.writerow(['step','elapsed_seconds','tag','value'])
                for row in records(directory/'metrics.jsonl'):
                    for k,v in row['values'].items():
                        if kind!='curve' or k==tag:writer.writerow([row['step'],row['elapsed'],k,v])
            return self.respond(out.getvalue().encode('utf-8-sig'),content_type='text/csv; charset=utf-8',filename=kind+'.csv')
        if kind=='frame':return self.respond((directory/'frame.jpg').read_bytes(),content_type='image/jpeg',filename='frame.jpg')
        if kind=='model':return self.respond((directory/'model.zip').read_bytes(),content_type='application/zip',filename='model.zip')
        if kind not in ('all','tensorboard'):raise ValueError('未知下载类型')
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
            for p in directory.rglob('*'):
                if p.is_file() and '.tmp' not in p.name and p.name!='STOP' and (kind=='all' or 'tensorboard' in p.parts):
                    z.write(p,str(p.relative_to(directory)))
        return self.respond(buffer.getvalue(),content_type='application/zip',filename=directory.name+'_'+kind+'.zip')

    def do_POST(self):
        try:
            # Local API has no CORS; refuse cross-origin browser mutations and DNS rebinding.
            host=self.headers.get('Host','').split(':')[0]
            if host not in ('127.0.0.1','localhost','[::1]'):raise ValueError('仅允许本地访问')
            origin=self.headers.get('Origin')
            if origin and urlparse(origin).netloc != self.headers.get('Host'):raise ValueError('拒绝跨来源请求')
            size=int(self.headers.get('Content-Length','0'))
            if size>65536:raise ValueError('请求过大')
            raw=json.loads(self.rfile.read(size) or '{}')
            if self.path=='/api/datasets/import':
                with LOCK:
                    return self.respond(import_clip(raw))
            if self.path=='/api/jobs':return self.respond(start_job(raw['kind'],raw['settings']))
            if self.path.startswith('/api/jobs/') and self.path.endswith('/stop'):
                directory=job_dir(self.path.split('/')[3]);(directory/'STOP').touch()
                return self.respond({'ok':True,'message':'将在当前步骤结束后保存模型并停止'})
            raise ValueError('未知接口')
        except (ValueError,KeyError) as e:self.respond({'error':str(e)},400)
        except Exception as e:self.respond({'error':str(e)},500)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    RUNS.mkdir(parents=True,exist_ok=True)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'MetaDrive Studio: http://localhost:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        if PROCESS and PROCESS.poll() is None:
            (job_dir(ACTIVE)/'STOP').touch()
            try:PROCESS.wait(timeout=20)
            except subprocess.TimeoutExpired:PROCESS.terminate()
        server.server_close()


if __name__=='__main__':main()
