"""Validated UI inputs. Frozen v1 configuration files are never mutated."""
from pathlib import Path
import json
import math
from env.research_config import load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'outputs/studio'  # Preview cache and legacy runs
MODELS = ROOT/'models'
RESULTS = ROOT/'results'
JOB_ROOTS = (MODELS, RESULTS, RUNS)


def run_root(kind):
    return {'train': MODELS, 'eval': RESULTS, 'preview': RUNS}[kind]

SCENARIOS = {'straight': ('Highway 直道（训练）', 'S'), 'curve': ('Highway 弯道（迁移）', 'C'), 'merge': ('Highway 汇入（迁移）', 'r')}
from rl.algorithms import REGISTRY
ALGORITHM_NAMES = {key: entry.name for key, entry in REGISTRY.items()}
DEFAULTS = dict(data_source='procedural', dataset_id='', algorithm='sac', scenario='straight', lane_num=3, traffic_density=.1, horizon=1000,
                seed_start=1000, num_scenarios=100, seed=0, traffic_seed=43, environment_seed=42,
                reward_mode='cmdp', cost_mode='continuous', timesteps=100000, learning_rate=.0003,
                batch_size=256, buffer_size=50000, learning_starts=1000, gamma=.99, tau=.005,
                gamma_cost=.99, cost_limit=5., tail_fraction=.1, safety_lr=.0003, dual_lr=.0001,
                initial_multiplier=.1, n_quantiles=32, hidden_size=256, device='auto',
                checkpoint_freq=10000, log_freq=100, frame_interval=.3, view_mode='bird3d', episodes=20, model_id='', deterministic=True)


def clean_request(raw, kind):
    if not isinstance(raw, dict):
        raise ValueError('参数必须为对象')
    unknown = set(raw)-set(DEFAULTS)
    if unknown:
        raise ValueError(f'未知参数: {sorted(unknown)}')
    c = {**DEFAULTS, **raw}
    for key, choices in [('data_source', ['procedural','highd','ngsim']), ('algorithm', ALGORITHM_NAMES), ('scenario', SCENARIOS), ('device', ['auto','cpu','cuda']),
                         ('view_mode', ['topdown','bird3d','chase3d','first3d']), ('reward_mode', ['cmdp','metadrive']), ('cost_mode', ['binary','continuous'])]:
        if c[key] not in choices:
            raise ValueError(f'无效 {key}')
    ints = dict(lane_num=(1,8), horizon=(1,100000), seed_start=(0,2**31-1), num_scenarios=(1,100000), seed=(0,2**32-1),
                traffic_seed=(0,2**32-1), environment_seed=(0,2**32-1), timesteps=(1,100000000),
                batch_size=(2,4096), buffer_size=(100,2000000), learning_starts=(0,10000000),
                n_quantiles=(2,256), hidden_size=(16,1024), checkpoint_freq=(1,10000000), log_freq=(1,10000), episodes=(1,100000))
    floats = dict(traffic_density=(0,1), learning_rate=(1e-8,.1), gamma=(0,.999999), tau=(1e-6,1),
                  gamma_cost=(0,.999999), cost_limit=(0,1000000), tail_fraction=(.001,1), safety_lr=(1e-8,.1),
                  dual_lr=(1e-8,.1), initial_multiplier=(1e-6,50), frame_interval=(.05,10))
    for key,(lo,hi) in {**ints, **floats}.items():
        value = c[key]
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError(f'{key} 必须在 [{lo}, {hi}]')
        if key in ints:
            if int(value) != value:
                raise ValueError(f'{key} 必须为整数')
            c[key] = int(value)
    if c['data_source']=='procedural' and c['lane_num']>4:
        raise ValueError('程序化场景支持 1–4 车道')
    if c['data_source']=='procedural' and kind=='train' and c['scenario']!='straight':
        raise ValueError('正式 Highway 训练只使用直道；弯道和汇入用于迁移评估')
    if type(c['deterministic']) is not bool:
        raise ValueError('deterministic 必须为布尔值')
    if c['batch_size'] > c['buffer_size']:
        raise ValueError('batch_size 不能大于 buffer_size')
    if kind == 'train' and c['learning_starts'] >= c['timesteps']:
        raise ValueError('总训练步数必须大于预热步数，否则不会发生学习')
    if kind == 'eval' and c['episodes'] > c['num_scenarios']:
        raise ValueError('评估回合数不能超过独立场景数；避免把重复场景计为独立样本')
    if c['data_source'] != 'procedural':
        from datasets.importer import load_clip
        clip, manifest = load_clip(c['dataset_id'])
        if manifest['source'] != c['data_source']: raise ValueError('数据源与片段不匹配')
        scenario_type = manifest.get('scenario_type', 'mainline' if manifest.get('source') == 'ngsim' and manifest.get('road_type') in (None, 'freeway', 'highway') else None)
        if manifest.get('road_type') not in (None, 'freeway', 'highway') or (scenario_type and scenario_type != 'mainline'):
            raise ValueError('该 NGSIM 类别尚无匹配的已验收道路网络，不能用于当前训练/评估')
        if kind in ('train','eval') and manifest['split'] != kind:
            raise ValueError('训练只能使用 train 片段，评估只能使用 eval 片段')
        if kind=='eval' and c['episodes']!=1: raise ValueError('单个轨迹片段评估仅支持 1 回合；重复 seed 不构成独立样本')
        c.update(scenario='straight',lane_num=clip['lane_num'],traffic_density=0.,num_scenarios=1,
                 horizon=min(c['horizon'],int(math.floor(clip['duration']/.1+1e-6))))
    environment_config(c, kind)
    return c


def clean_srairl_request(raw, kind):
    """Validate the isolated SR-AIRL legacy-kinematic Studio interface."""
    from datasets.importer import load_clip
    if kind not in ('train','eval'):
        raise ValueError('SR-AIRL Studio currently supports training and evaluation, not MetaDrive preview')
    scenarios=raw.get('srairl_scenarios')
    if not isinstance(scenarios,list) or not scenarios:
        raise ValueError('先选择一个已导入的 NGSIM 片段并指定四个车辆角色')
    if kind=='eval' and len(scenarios)!=1:
        raise ValueError('一次 SR-AIRL 评估对应一个独立轨迹片段')
    checked=[]
    role_keys=('elcv_id','rfv_id','rlv_current_id','rlv_target_id')
    for item in scenarios:
        if not isinstance(item,dict) or not isinstance(item.get('roles'),dict):
            raise ValueError('SR-AIRL 场景必须包含 dataset_id 和 roles')
        clip,manifest=load_clip(item.get('dataset_id'))
        required_split='train' if kind=='train' else 'eval'
        if manifest.get('source')!='ngsim' or manifest.get('split')!=required_split:
            raise ValueError(f'SR-AIRL {kind} 只能使用 NGSIM {required_split} 数据片段')
        roles={key:str(item['roles'].get(key,'')).strip() for key in role_keys}
        if any(not value for value in roles.values()):raise ValueError('ELCV、RFV、当前车道前车和目标车道前车 ID 均须明确填写')
        available={str(track['id']) for track in clip.get('tracks',[])}
        missing=sorted(set(roles.values())-available)
        if missing:raise ValueError(f'片段中找不到车辆轨迹 ID: {missing}')
        if len(set(roles.values()))!=4:raise ValueError('四个 SR-AIRL 车辆角色必须对应不同轨迹')
        from env.core.types import InteractionScenario
        from env.tasks.srairl_env import SRAIRLEnv
        interaction=InteractionScenario.from_clip(clip,manifest,**roles)
        SRAIRLEnv(interaction,dynamics_mode='legacy_kinematic')
        if kind=='train':
            from datasets.srairl.expert_builder import ExpertDatasetBuilder
            transitions=ExpertDatasetBuilder().build(interaction)
            if len(transitions)<4:raise ValueError('SR-AIRL 训练片段至少要含两个时间步的双车专家转移')
        checked.append({'dataset_id':item['dataset_id'],'roles':roles,'manifest':manifest})
    episodes=int(raw.get('srairl_episodes',1000))
    ppo_epochs=int(raw.get('srairl_ppo_epochs',80))
    seed=int(raw.get('srairl_seed',0))
    device=raw.get('srairl_device','auto')
    if not 1<=episodes<=10_000_000 or not 1<=ppo_epochs<=1000 or not 0<=seed<2**32:
        raise ValueError('SR-AIRL 回合数、PPO epochs 或随机种子超出范围')
    if device not in ('auto','cpu','cuda'):raise ValueError('SR-AIRL 设备只能是 auto、cpu 或 cuda')
    if kind=='eval' and not raw.get('srairl_model_id'):raise ValueError('评估时请选择已训练的 SR-AIRL 模型')
    view_mode=raw.get('srairl_view','topdown')
    if view_mode not in ('topdown','chase3d','first3d','bird3d'):raise ValueError('SR-AIRL MetaDrive view is invalid')
    meta_drive_world=raw.get('srairl_metadrive_world',True)
    if type(meta_drive_world) is not bool:raise ValueError('MetaDrive world selection must be boolean')
    reactive_background=raw.get('srairl_reactive_background',True)
    if type(reactive_background) is not bool:raise ValueError('SR-AIRL reactive background selection must be boolean')
    return {'srairl_scenarios':checked,'episodes':episodes,'ppo_epochs':ppo_epochs,'seed':seed,'device':device,
            'model_id':str(raw.get('srairl_model_id','')),'task':'srairl','dynamics_mode':'legacy_kinematic',
            'metadrive_world':meta_drive_world,'srairl_view':view_mode,'srairl_reactive_background':reactive_background}


def environment_config(c, kind):
    split = 'train' if kind == 'train' or (kind == 'preview' and 1000 <= c['seed_start'] < 2000) else 'eval'
    cfg = load_config(ROOT/f'configs/env/{"train" if split=="train" else "eval_id"}.yaml')
    cfg['environment'].update(map=SCENARIOS[c['scenario']][1], horizon=c['horizon'], lane_num=c.get('lane_num',3), traffic_density=c['traffic_density'], reward_mode=c['reward_mode'])
    cfg['scenario'].update(type=c['scenario'], split=split, start_seed=c['seed_start'], num_scenarios=c['num_scenarios'])
    cfg['seeds'].update(rl_seed=c['seed'], environment_seed=c['environment_seed'], traffic_seed=c['traffic_seed'])
    cfg['safety']['cost_mode'] = c['cost_mode']
    if c.get('data_source','procedural') != 'procedural':
        from datasets.importer import load_clip
        clip, manifest = load_clip(c['dataset_id'])
        cfg['environment'].update(map='',lane_num=clip['lane_num'],lane_width=clip['lane_width'],
            exit_length=max(clip['road_length'],50),traffic_density=0.,
            horizon=min(c['horizon'],int(math.floor(clip['duration']/.1+1e-6))))
        cfg['scenario'].update(type='straight',num_scenarios=1)
        cfg['dataset']=manifest
    validate_config(cfg)
    return cfg


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)
