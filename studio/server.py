"""Local-only dashboard. Simulations run in isolated subprocesses, one at a time."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.adapters.srairl_calibration import action_calibration_passed, calibration_summary_path
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
from studio.config import ROOT,RUNS,JOB_ROOTS,run_root,DEFAULTS,SCENARIOS,ALGORITHM_NAMES,clean_request,clean_srairl_request,clean_reward_guide_request,clean_predictor_request,clean_planner_request,clean_interaction_evaluation_request,atomic_json
from tasks.registry import TASK_REGISTRY

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


def srairl_model_catalog():
    result=[]
    for p in sorted((ROOT/'models').glob('*/config.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        meta=read_json(p,{}) or {}
        if meta.get('algorithm')!='SR-AIRL' or meta.get('reward_version')!='temporal-dynamic-v2' or \
           meta.get('srairl',{}).get('history_len')!=11 or \
           not (p.parent/'policy_elcv.pt').is_file() or not (p.parent/'policy_rfv.pt').is_file():continue
        from env.tasks.srairl_interaction import ROAD_GEOMETRY_CONFIG, REFERENCE_FRAME_VERSION
        if meta.get('road_geometry')!=ROAD_GEOMETRY_CONFIG or meta.get('reference_frame_version')!=REFERENCE_FRAME_VERSION or not meta.get('source_dataset_revision'):continue
        if not all((p.parent/name).is_file() for name in ('discriminator_elcv.pt','discriminator_rfv.pt')):continue
        result.append(dict(id=str(p.parent.relative_to(ROOT)),label=f"SR-AIRL · {p.parent.name}",path=str(p.parent.relative_to(ROOT)),
                           episode=meta.get('episode',0),global_steps=meta.get('global_steps',0),
                           dynamics_mode=meta.get('dynamics_mode','legacy_kinematic'),
                           scene_source=meta.get('scene_source','ngsim'),scenario=meta.get('scenario','straight'),
                           lane_num=meta.get('lane_num'),traffic_density=meta.get('traffic_density'),horizon=meta.get('horizon'),
                           interaction_scenario=meta.get('interaction_scenario'),
                           initialization_protocol=meta.get('initialization_protocol','canonical')))
    return result


def srairl_expert_catalog():
    result=[]
    for path in sorted((ROOT/'datasets'/'srairl'/'metadrive').glob('*.npz'),key=lambda p:p.stat().st_mtime,reverse=True):
        try:
            import numpy as np
            with np.load(path,allow_pickle=False) as data:metadata=json.loads(str(data['metadata'].item()))
            from datasets.srairl.native_windows import native_dataset_info
            info=native_dataset_info(path,require_current=True)
            result.append(dict(id=str(path.relative_to(ROOT)),label=f"MetaDrive 示范 · {metadata.get('scenario','?')} · {info['episodes']} episodes · {info['windows']} windows",
                path=str(path.relative_to(ROOT)),transition_pairs=metadata.get('transition_pairs',0),scenario=metadata.get('scenario'),
                controller=metadata.get('controller'),seed=metadata.get('seed'),episodes=info['episodes'],
                attempted_episodes=metadata.get('episodes',info['episodes']),
                rejected_episodes=max(0,int(metadata.get('episodes',info['episodes']))-info['episodes']),
                acceptance_rate=info['episodes']/max(1,int(metadata.get('episodes',info['episodes']))),
                windows=info['windows'],windows_by_split=info['windows_by_split'],
                windows_by_scenario=info['windows_by_scenario'],split_protocol=info['split_protocol']))
        except Exception:
            continue
    return result


def reward_guide_model_catalog():
    result=[]
    for p in sorted((ROOT/'models').glob('*/results/best_model.pth'),key=lambda p:p.stat().st_mtime,reverse=True):
        meta=read_json(p.parents[1]/'summary.json',{}) or {}
        if meta.get('task')!='reward_guide':continue
        from rl.srairl.pipeline.model_bundle import torch_load_metadata
        from env.tasks.srairl_interaction import REFERENCE_FRAME_VERSION
        try:checkpoint_meta=torch_load_metadata(p).get('metadata',{})
        except Exception:continue
        if checkpoint_meta.get('reference_frame_version')!=REFERENCE_FRAME_VERSION or not checkpoint_meta.get('source_dataset_revision'):continue
        result.append(dict(id=str(p.parents[1].relative_to(ROOT)),label=f"SR-AIRL Reward Guide · {p.parents[1].name}",
                           path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,metrics=meta.get('metrics',{})))
    return result


def srairl_pipeline_bundle_catalog():
    """List only complete, metadata-compatible component bundles."""
    import torch
    def payload(path):
        try:
            value=torch.load(path,map_location='cpu',weights_only=True)
            return value if isinstance(value,dict) else {}
        except Exception:
            return {}
    planners=[];guides=[];predictors=[]
    for summary_path in (ROOT/'models').glob('*/summary.json'):
        summary=read_json(summary_path,{}) or {};base=summary_path.parent
        task=summary.get('task')
        if task=='planner':
            goal=base/'planner_goal/results/goal_model.pth';planner=base/'planner_dppo/state_final.pt'
            if goal.is_file() and planner.is_file():
                goal_data=payload(goal);planner_data=payload(planner)
                planners.append((base,goal,planner,goal_data.get('metadata',{}),planner_data.get('metadata',{})))
        elif task=='reward_guide':
            checkpoint=base/'results/best_model.pth'
            if checkpoint.is_file():guides.append((base,checkpoint,payload(checkpoint).get('metadata',{})))
        elif task=='predictor':
            checkpoint=base/'results/best_model.pth'
            if checkpoint.is_file():predictors.append((base,checkpoint,payload(checkpoint).get('metadata',{})))
    rewards=[]
    for config_path in (ROOT/'models').glob('*/config.json'):
        config=read_json(config_path,{}) or {};base=config_path.parent
        if config.get('algorithm')=='SR-AIRL' and (base/'discriminator_elcv.pt').is_file() and (base/'discriminator_rfv.pt').is_file():
            rewards.append((base,config))
    result=[]
    for _,goal,planner,goal_meta,planner_meta in planners:
        revision=goal_meta.get('dataset_revision')
        if not revision or planner_meta.get('dataset_revision')!=revision:continue
        for _,guide,guide_meta in guides:
            if guide_meta.get('dataset_revision')!=revision:continue
            for _,predictor,predictor_meta in predictors:
                if predictor_meta.get('dataset_revision')!=revision:continue
                for reward_dir,_ in rewards:
                    try:
                        from rl.srairl.pipeline.model_bundle import SRAIRLModelBundle
                        bundle=SRAIRLModelBundle.from_checkpoints(
                            planner_goal_path=goal,planner_diffusion_path=planner,
                            reward_guide_path=guide,predictor_path=predictor,
                            reward_model_path=reward_dir)
                        manifest=bundle.to_manifest()
                    except Exception:
                        continue
                    identifier=hashlib.sha256(json.dumps(manifest['checkpoints'],sort_keys=True).encode()).hexdigest()[:20]
                    result.append({'id':identifier,'label':f'SR-AIRL pipeline · {reward_dir.name} · {goal.parents[2].name}',
                        'reward_model_path':str(reward_dir.relative_to(ROOT)),
                        'planner_goal_path':str(goal.relative_to(ROOT)),
                        'planner_diffusion_path':str(planner.relative_to(ROOT)),
                        'reward_guide_path':str(guide.relative_to(ROOT)),
                        'predictor_path':str(predictor.relative_to(ROOT)),
                        'dataset_revision':revision})
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
    task=raw.pop('task','safe_rl')
    scenario_family=raw.pop('scenario_family',None)
    scenario_type=raw.pop('scenario_type',None)
    if scenario_family is not None:
        if scenario_family not in ('highway','ngsim','highd'):
            raise ValueError('未知场景来源；请选择 highway、ngsim 或 highd')
        if scenario_family in ('ngsim','highd'):
            raw.setdefault('data_source',scenario_family)
        if scenario_family=='highway':
            raw.setdefault('data_source','procedural')
            if scenario_type is not None:
                raw['scenario']=scenario_type
    if task not in TASK_REGISTRY:raise ValueError('未知实验任务')
    descriptor=TASK_REGISTRY[task]
    if descriptor.status != 'available' and task!='evaluation':
        raise ValueError(f'{descriptor.label} 尚未具备可运行的完整训练链：{descriptor.description}')
    if task=='evaluation':
        if kind!='eval':raise ValueError('综合生成式 SR-AIRL 任务只允许评估')
        calibration=read_json(calibration_summary_path(),{}) or {}
        if not action_calibration_passed(calibration,require_closed_loop=True):
            raise ValueError('Action Adapter 标定尚未通过网格和留出集，不能运行综合闭环评估')
        selected=next((m for m in srairl_pipeline_bundle_catalog() if m['id']==raw.get('pipeline_bundle_id')),None)
        if selected is None:raise ValueError('请选择具有完整兼容检查点的 SR-AIRL ModelBundle')
        c=clean_interaction_evaluation_request(raw)
        c['model_bundle_id']=selected['id']
        c['model_bundle_paths']={key:str((ROOT/selected[key]).resolve()) for key in (
            'reward_model_path','planner_goal_path','planner_diffusion_path','reward_guide_path','predictor_path')}
        with LOCK:
            if PROCESS is not None and PROCESS.poll() is None:
                previous=read_json(job_dir(ACTIVE)/'status.json',{})
                if previous.get('state') not in ('completed','stopped','failed'):
                    raise ValueError('已有任务运行中。请先停止或等待完成。')
                try:PROCESS.wait(timeout=5)
                except subprocess.TimeoutExpired:raise ValueError('上一个任务正在释放资源，请稍后再试。')
            name=f'{time.strftime("%Y%m%d_%H%M%S")}_eval_{time.time_ns()%1000000:06d}'
            directory=run_root('eval')/name;directory.mkdir(parents=True)
            atomic_json(directory/'request.json',dict(task=task,kind=kind,settings=c))
            atomic_json(directory/'status.json',dict(state='starting',kind=kind,step=0))
            with (directory/'worker.log').open('w') as log:
                PROCESS=subprocess.Popen([sys.executable,'-u',str(ROOT/'studio/worker.py'),'--job',str(directory)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            ACTIVE=name
            return {'id':name}
    if task == 'srairl':
        c=clean_srairl_request(raw,kind)
        if kind=='eval':
            selected=next((m for m in srairl_model_catalog() if m['id']==c['model_id']),None)
            if selected is None:raise ValueError('请选择存在的 SR-AIRL 模型')
            c['model_path']=str((ROOT/selected['path']).resolve())
        with LOCK:
            if PROCESS is not None and PROCESS.poll() is None:
                previous=read_json(job_dir(ACTIVE)/'status.json',{})
                if previous.get('state') in ('completed','stopped','failed'):
                    try:PROCESS.wait(timeout=5)
                    except subprocess.TimeoutExpired:raise ValueError('上一个任务正在释放资源，请稍后再试。')
                else:raise ValueError('已有任务运行中。请先停止或等待完成。')
            name=f'{time.strftime("%Y%m%d_%H%M%S")}_{kind}_{time.time_ns()%1000000:06d}'
            directory=run_root(kind)/name;directory.mkdir(parents=True)
            atomic_json(directory/'request.json',dict(task=task,kind=kind,settings=c))
            atomic_json(directory/'status.json',dict(state='starting',kind=kind,step=0))
            with (directory/'worker.log').open('w') as log:
                PROCESS=subprocess.Popen([sys.executable,'-u',str(ROOT/'studio/worker.py'),'--job',str(directory)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            ACTIVE=name
            return {'id':name}
    if task in ('planner','reward_guide','predictor'):
        if task=='planner':
            c=clean_planner_request(raw,kind)
        elif task=='reward_guide':
            c=clean_reward_guide_request(raw,kind)
            selected=next((m for m in srairl_model_catalog() if m['id']==c['srairl_model_id']),None)
            if selected is None:raise ValueError('请选择包含 SR-AIRL 策略与判别器检查点的已完成模型')
            c['srairl_model_path']=str((ROOT/selected['path']).resolve())
        else:
            c=clean_predictor_request(raw,kind)
            selected=next((m for m in reward_guide_model_catalog() if m['id']==c['reward_guide_model_id']),None)
            if selected is None:raise ValueError('请选择包含验证集最优检查点的已完成奖励引导器模型')
            c['reward_guide_path']=str((ROOT/selected['path']).resolve())
        with LOCK:
            if PROCESS is not None and PROCESS.poll() is None:
                previous=read_json(job_dir(ACTIVE)/'status.json',{})
                if previous.get('state') in ('completed','stopped','failed'):
                    try:PROCESS.wait(timeout=5)
                    except subprocess.TimeoutExpired:raise ValueError('上一个任务正在释放资源，请稍后再试。')
                else:raise ValueError('已有任务运行中。请先停止或等待完成。')
            name=f'{time.strftime("%Y%m%d_%H%M%S")}_{kind}_{time.time_ns()%1000000:06d}'
            directory=run_root(kind)/name;directory.mkdir(parents=True)
            atomic_json(directory/'request.json',dict(task=task,kind=kind,settings=c))
            atomic_json(directory/'status.json',dict(state='starting',kind=kind,step=0))
            with (directory/'worker.log').open('w') as log:
                PROCESS=subprocess.Popen([sys.executable,'-u',str(ROOT/'studio/worker.py'),'--job',str(directory)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            ACTIVE=name
            return {'id':name}
    if task != 'safe_rl':raise ValueError('当前任务暂不可由 Studio 启动')
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
        atomic_json(directory/'request.json',dict(task=task,kind=kind,settings=c,model_path=str(ROOT/model['path']) if model else None))
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
            from env.tasks.srairl_interaction import SCENARIO_SPEED_CONFIG, ROAD_GEOMETRY_CONFIG
            calibration=read_json(calibration_summary_path(),{})
            return self.respond(dict(pipeline_calibration_passed=action_calibration_passed(calibration,require_closed_loop=True),srairl_road_geometry=ROAD_GEOMETRY_CONFIG,srairl_scenario_speeds=SCENARIO_SPEED_CONFIG,defaults=DEFAULTS,scenarios=[dict(id=k,name=v[0]) for k,v in SCENARIOS.items()],tasks={k:vars(v) for k,v in TASK_REGISTRY.items()},algorithms=ALGORITHM_NAMES,algorithm_metadata=algorithm_metadata(),models=model_catalog(),srairl_models=srairl_model_catalog(),reward_guide_models=reward_guide_model_catalog(),srairl_pipeline_bundles=srairl_pipeline_bundle_catalog(),srairl_expert_datasets=srairl_expert_catalog(),srairl_action_calibration=calibration))
        if path=='/api/datasets/catalog':
            return self.respond(dataset_catalog())
        if path=='/api/jobs':
            jobs=[]
            for p in sorted((p for root in JOB_ROOTS for p in root.glob('*') if p.is_dir() and (p/'request.json').exists()),key=lambda p:p.name,reverse=True):
                if p.is_dir():jobs.append(dict(id=p.name,request=read_json(p/'request.json',{}),status=status(p)))
            return self.respond(dict(active=ACTIVE,jobs=jobs))
        if path.startswith('/api/datasets/') and path.endswith('/tracks'):
            identifier=path.split('/')[3]
            from datasets.importer import load_clip
            clip,manifest=load_clip(identifier)
            return self.respond(dict(id=identifier,source=manifest['source'],split=manifest['split'],tracks=[dict(id=str(t['id']),ego=bool(t.get('ego')),states=len(t.get('states',[]))) for t in clip.get('tracks',[])]))
        if path.startswith('/api/jobs/'):
            parts=path.split('/');directory=job_dir(parts[3]);action=parts[4] if len(parts)>4 else ''
            if action=='videos':
                return self.respond({'videos':[p.name for p in sorted((directory/'videos').glob('*.mp4'))]})
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
        if path=='/run':
            p=ROOT/'studio/static'/'run.html'
            return self.respond(p.read_bytes(),content_type='text/html; charset=utf-8')
        if path=='/run.js':
            p=ROOT/'studio/static'/'run.js'
            return self.respond(p.read_bytes(),content_type='text/javascript; charset=utf-8')
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
        if kind=='video':
            name=q.get('name',[''])[0]
            if not name or Path(name).name!=name or not name.endswith('.mp4'):raise ValueError('无效视频文件名')
            return self.respond((directory/'videos'/name).read_bytes(),content_type='video/mp4',filename=name)
        if kind=='model':
            request=read_json(directory/'request.json',{})
            if request.get('task') in ('reward_guide','predictor'):
                checkpoint=directory/'results'/'best_model.pth'
                return self.respond(checkpoint.read_bytes(),content_type='application/octet-stream',filename='best_model.pth')
            if request.get('task')=='planner':
                buffer=io.BytesIO()
                with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
                    for rel in ('planner_goal/results/goal_model.pth','planner_dppo/state_final.pt','summary.json'):
                        checkpoint=directory/rel
                        if checkpoint.is_file():z.write(checkpoint,rel)
                return self.respond(buffer.getvalue(),content_type='application/zip',filename='planner_models.zip')
            return self.respond((directory/'model.zip').read_bytes(),content_type='application/zip',filename='model.zip')
        if kind=='expert':
            request=read_json(directory/'request.json',{})
            if request.get('settings',{}).get('operation')!='expert_collect':raise ValueError('此任务没有 SR-AIRL 专家数据集')
            return self.respond((directory/'expert_transitions.npz').read_bytes(),content_type='application/octet-stream',filename='expert_transitions.npz')
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
            if self.path=='/api/jobs':
                settings=dict(raw['settings'])
                for key in ('task','scenario_family','scenario_type'):
                    if key in raw:settings[key]=raw[key]
                return self.respond(start_job(raw['kind'],settings))
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
