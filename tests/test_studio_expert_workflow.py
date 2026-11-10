"""Regression checks for collection scope, lifecycle, progress, and persistence."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from studio.config import clean_srairl_request


def test_collection_scope_and_seed_bounds():
    common=dict(srairl_demo_episodes=3,srairl_seed=10000,srairl_traffic_density=.05)
    one=clean_srairl_request(dict(common,srairl_operation='expert_collect',srairl_scenario='curve'),'train')
    assert one['scenarios']==['curve'] and one['traffic_density']==.05
    mixed=clean_srairl_request(dict(common,srairl_operation='expert_collect',
        srairl_collect_scenarios=['straight','curve','merge']),'train')
    assert mixed['scenarios']==['straight','curve','merge']
    for changes in ({'srairl_demo_episodes':2},{'srairl_seed':2**32-2},
                    {'srairl_scenario':'invalid'}, {'srairl_traffic_density':float('nan')}):
        with pytest.raises(ValueError):
            clean_srairl_request(dict(common,srairl_operation='expert_collect',**changes),'train')


@pytest.mark.parametrize('stop_after_first',[False,True])
def test_collection_progress_persistence_and_stop(tmp_path,monkeypatch,stop_after_first):
    import studio.worker as worker
    import datasets.srairl.metadrive_expert as expert
    import datasets.srairl.expert_builder as builder
    import env.tasks.srairl_procedural_env as procedural
    job=tmp_path/'job';job.mkdir()
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    created=[];statuses=[]
    atomic=worker.atomic_json
    def record_status(path,value):
        if path.name=='status.json':statuses.append(dict(value))
        atomic(path,value)
    monkeypatch.setattr(worker,'atomic_json',record_status)
    class FakeEnv:
        def __init__(self,scene,*args,**kwargs):
            self.scene=scene;self._last_info={};self.closed=False;created.append(self)
            self.world=SimpleNamespace(role_vehicle=lambda role:SimpleNamespace(speed=15.))
        def close(self):self.closed=True
    def collect(env,n,seed,callback,**kwargs):
        rows=[];raw=[]
        for i in range(n):
            assert callback(dict(state='running',episodes=i,step=i,total=n))
            for step in (1,2):
                assert callback(dict(state='running',episodes=i,step=i,total=n,episode_step=step))
            rows.append(dict(seed=seed+i,accepted=i!=1,steps=2))
            raw.append(dict(validation={'valid':i!=1}))
            assert callback(dict(state='running',episodes=i+1,step=i+1,total=n))
        if stop_after_first:(job/'STOP').touch()
        return [object(),object()],rows,raw
    def save_raw(path,raw):
        path.parent.mkdir(exist_ok=True);path.write_bytes(b'raw')
    def save_dataset(path,transitions,metadata,**kwargs):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'dataset');return len(transitions)//2
    monkeypatch.setattr(procedural,'SRAIRLProceduralEnv',FakeEnv)
    monkeypatch.setattr(expert,'collect_metadrive_expert',collect)
    monkeypatch.setattr(expert,'save_raw_episode',save_raw)
    monkeypatch.setattr(builder,'save_transitions',save_dataset)
    config=clean_srairl_request(dict(srairl_operation='expert_collect',srairl_demo_episodes=3,
        srairl_collect_scenarios=['straight','curve','merge'],srairl_seed=10000),'train')
    result=worker.collect_srairl_expert(job,config)
    scenes=1 if stop_after_first else 3
    assert len(created)==scenes and all(env.closed for env in created)
    assert result['total']==9 and result['episodes']==3*scenes
    assert result['state']==('stopped' if stop_after_first else 'completed')
    assert all(s['total']==9 and 0<=s['step']<=9 for s in statuses)
    rows=[json.loads(line) for line in (job/'episodes.jsonl').read_text().splitlines()]
    assert len(rows)==3*scenes
    assert [r['episode'] for r in rows]==list(range(1,len(rows)+1))
    assert all((job/r['episode_file']).exists() and r['scenario'] in config['scenarios'] for r in rows)
    metrics=[json.loads(line)['step'] for line in (job/'metrics.jsonl').read_text().splitlines()]
    assert metrics==list(range(1,6*scenes+1))
    summary=json.loads((job/'summary.json').read_text())
    assert summary['accepted_episodes']==2*scenes
    assert summary['rejected_episodes']==scenes
    assert len(list((job/'failed_episodes').glob('*.npz')))==scenes


def test_stop_before_initialization_does_not_start_scene(tmp_path):
    import studio.worker as worker
    (tmp_path/'STOP').touch()
    result=worker.collect_srairl_expert(tmp_path,dict(scenarios=['straight','curve','merge'],
        scenario='straight',demo_episodes=3))
    assert result['state']=='stopped' and result['total']==9 and result['episodes']==0


def test_repeated_runs_use_distinct_descriptive_directories(tmp_path,monkeypatch):
    import studio.config as config
    monkeypatch.setattr(config,'MODELS',tmp_path/'models')
    monkeypatch.setattr(config,'RESULTS',tmp_path/'results')
    monkeypatch.setattr(config,'RUNS',tmp_path/'preview')
    settings=dict(operation='expert_collect',scenarios=['curve'],traffic_density=.05,seed=10000)
    first=config.create_run_directory('train','srairl',settings)
    (first/'checkpoint.pt').write_bytes(b'original')
    second=config.create_run_directory('train','srairl',settings)
    assert first!=second and first.parent==second.parent==tmp_path/'models'
    assert 'train_expert_curve_td0p05_seed10000' in first.name
    assert (first/'checkpoint.pt').read_bytes()==b'original'
    batch=config.create_run_directory('train','srairl',dict(settings,scenarios=['straight','curve','merge']))
    assert 'straight-curve-merge' in batch.name
    component=config.create_run_directory('train','planner',dict(seed=7,dataset_id=['a.npz']))
    assert 'planner_data' in component.name and 'seed7' in component.name
    evaluation=config.create_run_directory('eval','safe_rl',dict(algorithm='wcsac',scenario='merge',traffic_density=.15,seed=9))
    assert evaluation.parent==tmp_path/'results' and 'safe_rl_wcsac_merge_td0p15_seed9' in evaluation.name


def test_server_allocates_a_new_job_for_repeated_collection(tmp_path,monkeypatch):
    import studio.config as config
    import studio.server as server
    monkeypatch.setattr(config,'MODELS',tmp_path/'models')
    monkeypatch.setattr(server,'PROCESS',None)
    monkeypatch.setattr(server,'ACTIVE',None)
    monkeypatch.setattr(server.subprocess,'Popen',lambda *a,**kw:SimpleNamespace(poll=lambda:0))
    request=dict(task='srairl',srairl_operation='expert_collect',srairl_scenario='merge',
        srairl_demo_episodes=3,srairl_traffic_density=.15,srairl_seed=30000)
    first=server.start_job('train',dict(request))['id']
    second=server.start_job('train',dict(request))['id']
    assert first!=second
    for name in (first,second):
        path=tmp_path/'models'/name
        assert 'expert_merge_td0p15_seed30000' in name
        assert json.loads((path/'request.json').read_text())['settings']['scenario']=='merge'
        assert (path/'status.json').exists() and (path/'worker.log').exists()


def test_srairl_preview_uses_collection_geometry_without_collecting():
    config=clean_srairl_request(dict(srairl_scenario='merge',srairl_traffic_density=.05,
        srairl_seed=30000,srairl_lane_num=3),'preview')
    assert config['operation']=='scene_preview'
    assert config['scenario']=='merge' and config['traffic_density']==.05
    assert config['interaction_scenario']['elcv_speed']==20.
