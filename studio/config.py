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
    """Validate the isolated SR-AIRL Studio interface."""
    from datasets.importer import load_clip
    if kind not in ('train','eval'):
        raise ValueError('SR-AIRL Studio currently supports training and evaluation, not MetaDrive preview')
    operation=raw.get('srairl_operation','interaction')
    if operation not in ('interaction','expert_collect'):
        raise ValueError('未知 SR-AIRL 操作类型')
    if operation=='expert_collect':
        if kind!='train':raise ValueError('专家示范采集属于数据生成操作，请选择训练模式')
        scenario_type=raw.get('srairl_scenario','straight');lane_num=raw.get('srairl_lane_num',3)
        density=raw.get('srairl_traffic_density',.15);horizon=raw.get('srairl_horizon',1000)
        episodes=raw.get('srairl_demo_episodes',10);seed=raw.get('srairl_seed',0)
        if scenario_type not in ('straight','curve','merge'):raise ValueError('MetaDrive 示范场景只能为直道、弯道或汇入')
        if isinstance(lane_num,bool) or not isinstance(lane_num,int) or not 2<=lane_num<=4:raise ValueError('车道数须为 2–4')
        if isinstance(density,bool) or not isinstance(density,(int,float)) or not math.isfinite(density) or not 0<=density<=1:raise ValueError('交通密度须在 [0, 1]')
        if isinstance(horizon,bool) or not isinstance(horizon,int) or not 50<=horizon<=100000:raise ValueError('每回合步数须在 [50, 100000]')
        if isinstance(episodes,bool) or not isinstance(episodes,int) or not 1<=episodes<=10000:raise ValueError('示范回合数须在 [1, 10000]')
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**32:raise ValueError('随机种子超出范围')
        view_mode=raw.get('srairl_view','topdown')
        if view_mode not in ('topdown','chase3d','first3d','bird3d'):raise ValueError('MetaDrive 画面视角无效')
        return {'task':'srairl','operation':operation,'scene_source':'procedural','scenario':scenario_type,
            'lane_num':lane_num,'traffic_density':float(density),'horizon':horizon,'demo_episodes':episodes,
            'seed':seed,'scene_seed':seed,'traffic_seed':(seed+1_000_003)%(2**32),'rl_seed':seed,
            'device':'cpu','srairl_view':view_mode,'dynamics_mode':'metadrive_physics',
            'expert_controller':'headway_lane_keeper_v1'}
    scene_source=raw.get('srairl_scene_source','ngsim')
    if scene_source not in ('ngsim','procedural'):
        raise ValueError('SR-AIRL 场景来源只能为 NGSIM 或 MetaDrive 程序化道路')
    scenarios=raw.get('srairl_scenarios',[])
    expert_dataset_id=str(raw.get('srairl_expert_dataset_id','')).strip()
    if expert_dataset_id:
        expert_root=(ROOT/'datasets'/'srairl'/'metadrive').resolve()
        expert_path=(ROOT/expert_dataset_id).resolve()
        if kind!='train' or expert_path.parent!=expert_root or not expert_path.is_file() or expert_path.suffix!='.npz':
            raise ValueError('所选 MetaDrive 专家示范数据集无效')
        from datasets.srairl.expert_builder import load_transitions
        from rl.srairl.config import SRAIRLConfig
        expert_transitions,expert_metadata=load_transitions(expert_path)
        config_dim=16*SRAIRLConfig.from_mapping().history_len
        if any(item.state.shape!=(config_dim,) or item.action.shape!=(2,) for item in expert_transitions):
            raise ValueError('MetaDrive 专家示范的状态/动作维度与当前 SR-AIRL 模型不匹配')
    else:
        expert_metadata=None
    needs_experts=kind=='train' and not expert_dataset_id
    needs_ngsim_scene=scene_source=='ngsim'
    if needs_experts or (needs_ngsim_scene and not expert_dataset_id):
        if not isinstance(scenarios,list) or not scenarios:
            raise ValueError('请选择 NGSIM 片段并明确指定四个车辆角色；训练时该片段提供专家示范')
    else:
        scenarios=[]
    if kind=='eval' and scene_source=='ngsim' and len(scenarios)!=1:
        raise ValueError('一次 SR-AIRL NGSIM 评估对应一个独立轨迹片段')
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
        tracks_by_id={str(track['id']):track for track in clip.get('tracks',[])}
        missing=sorted(set(roles.values())-available)
        if missing:raise ValueError(f'片段中找不到车辆轨迹 ID: {missing}')
        if len(set(roles.values()))!=4:raise ValueError('四个 SR-AIRL 车辆角色必须对应不同轨迹')
        from env.core.types import InteractionScenario
        from env.tasks.srairl_env import SRAIRLEnv
        interaction=InteractionScenario.from_clip(clip,manifest,**roles)
        SRAIRLEnv(interaction,dynamics_mode='legacy_kinematic')
        for track_id in roles.values():
            if float(tracks_by_id[track_id]['states'][0][0])>-.2+1e-6:
                raise ValueError(f'SR-AIRL 轨迹 {track_id} 缺少 0.2 秒预滚动，无法复现上游差分速度/加速度')
        if kind=='train':
            from datasets.srairl.expert_builder import ExpertDatasetBuilder
            transitions=ExpertDatasetBuilder().build(interaction)
            if len(transitions)<4:raise ValueError('SR-AIRL 训练片段至少要含两个时间步的双车专家转移')
        checked.append({'dataset_id':item['dataset_id'],'roles':roles,'manifest':manifest})
    episodes=int(raw.get('srairl_episodes',1000))
    ppo_epochs=int(raw.get('srairl_ppo_epochs',80))
    seed=int(raw.get('srairl_seed',0))
    scene_seed=int(raw.get('srairl_scene_seed',seed))
    traffic_seed=int(raw.get('srairl_traffic_seed',(seed+1_000_003)%(2**32)))
    rl_seed=int(raw.get('srairl_rl_seed',seed))
    device=raw.get('srairl_device','auto')
    if not 1<=episodes<=10_000_000 or not 1<=ppo_epochs<=1000 or not 0<=seed<2**32:
        raise ValueError('SR-AIRL 回合数、PPO epochs 或随机种子超出范围')
    if any(not 0<=value<2**32 for value in (scene_seed,traffic_seed,rl_seed)):
        raise ValueError('SR-AIRL scene/traffic/RL seeds must be in [0, 2^32)')
    if device not in ('auto','cpu','cuda'):raise ValueError('SR-AIRL 设备只能是 auto、cpu 或 cuda')
    if kind=='eval' and not raw.get('srairl_model_id'):raise ValueError('评估时请选择已训练的 SR-AIRL 模型')
    view_mode=raw.get('srairl_view','topdown')
    if view_mode not in ('topdown','chase3d','first3d','bird3d'):raise ValueError('SR-AIRL MetaDrive view is invalid')
    meta_drive_world=raw.get('srairl_metadrive_world',True)
    if type(meta_drive_world) is not bool:raise ValueError('MetaDrive world selection must be boolean')
    reactive_background=raw.get('srairl_reactive_background',False)
    if type(reactive_background) is not bool:raise ValueError('SR-AIRL reactive background selection must be boolean')
    dynamics_mode=raw.get('srairl_dynamics_mode','legacy_kinematic')
    if dynamics_mode not in ('metadrive_physics','legacy_kinematic'):
        raise ValueError('SR-AIRL dynamics must be MetaDrive physics or upstream kinematics')
    if dynamics_mode=='metadrive_physics' and not meta_drive_world:
        raise ValueError('MetaDrive physics mode requires the MetaDrive world')
    scenario_type=raw.get('srairl_scenario','straight')
    if scenario_type not in ('straight','curve','merge'):
        raise ValueError('SR-AIRL MetaDrive map must be straight, curve, or merge')
    lane_num=raw.get('srairl_lane_num',3)
    density=raw.get('srairl_traffic_density',.15)
    horizon=raw.get('srairl_horizon',1000)
    if isinstance(lane_num,bool) or not isinstance(lane_num,int) or not 2<=lane_num<=4:
        raise ValueError('SR-AIRL generated interaction maps require 2–4 lanes')
    if isinstance(density,bool) or not isinstance(density,(int,float)) or not math.isfinite(density) or not 0<=density<=1:
        raise ValueError('SR-AIRL traffic density must be in [0, 1]')
    if isinstance(horizon,bool) or not isinstance(horizon,int) or not 50<=horizon<=100000:
        raise ValueError('SR-AIRL procedural horizon must be in [50, 100000]')
    if scene_source=='procedural' and dynamics_mode!='metadrive_physics':
        raise ValueError('MetaDrive-generated SR-AIRL scenes require MetaDrive vehicle physics')
    if scene_source=='procedural':
        # Procedural traffic is MetaDrive's IDM traffic manager, not NGSIM replay.
        reactive_background=False
    return {'srairl_scenarios':checked,'expert_dataset_id':expert_dataset_id,
            'expert_dataset_metadata':expert_metadata,'episodes':episodes,'ppo_epochs':ppo_epochs,'seed':seed,'device':device,
            'model_id':str(raw.get('srairl_model_id','')),'task':'srairl','dynamics_mode':dynamics_mode,
            'operation':operation,
            'metadrive_world':meta_drive_world,'srairl_view':view_mode,'srairl_reactive_background':reactive_background,
            'scene_source':scene_source,'scenario':scenario_type,'lane_num':lane_num,
            'traffic_density':float(density),'horizon':horizon,'scene_seed':scene_seed,
            'traffic_seed':traffic_seed,'rl_seed':rl_seed}


def clean_reward_guide_request(raw, kind):
    if kind != 'train':
        raise ValueError('奖励引导器当前只提供训练；预测器接入后再开放组合评估')
    model_id = str(raw.get('srairl_model_id', '')).strip()
    if not model_id:
        raise ValueError('训练奖励引导器前，先选择已完成的 SR-AIRL 判别器模型')
    epochs = raw.get('upstream_epochs', 1000)
    batch_size = raw.get('upstream_batch_size', 32)
    seed = raw.get('upstream_seed', 42)
    device = raw.get('upstream_device', 'auto')
    for name, value, low, high in (
        ('upstream_epochs', epochs, 1, 1_000_000),
        ('upstream_batch_size', batch_size, 1, 4096),
        ('upstream_seed', seed, 0, 2**32 - 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f'{name} 必须是 [{low}, {high}] 内的整数')
    if device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('设备只能是 auto、cpu 或 cuda')
    if device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise ValueError('当前环境没有可用 CUDA 设备')
    return {'task': 'reward_guide', 'srairl_model_id': model_id, 'epochs': epochs,
            'batch_size': batch_size, 'seed': seed, 'device': device}


def clean_predictor_request(raw, kind):
    if kind != 'train':
        raise ValueError('响应预测器当前只提供训练；候选评估流程尚未接入')
    model_id = str(raw.get('reward_guide_model_id', '')).strip()
    if not model_id:
        raise ValueError('训练预测器前，先选择已完成的奖励引导器模型')
    epochs = raw.get('upstream_epochs', 1000)
    batch_size = raw.get('upstream_batch_size', 32)
    seed = raw.get('upstream_seed', 42)
    device = raw.get('upstream_device', 'auto')
    for name, value, low, high in (
        ('upstream_epochs', epochs, 1, 1_000_000),
        ('upstream_batch_size', batch_size, 1, 4096),
        ('upstream_seed', seed, 0, 2**32 - 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f'{name} 必须是 [{low}, {high}] 内的整数')
    if device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('设备只能是 auto、cpu 或 cuda')
    if device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise ValueError('当前环境没有可用 CUDA 设备')
    return {'task': 'predictor', 'reward_guide_model_id': model_id, 'epochs': epochs,
            'batch_size': batch_size, 'seed': seed, 'device': device}


def clean_planner_request(raw, kind):
    if kind != 'train':
        raise ValueError('规划器候选生成和闭环评估接口仍在接入；当前提供完整训练链')
    goal_epochs=raw.get('planner_goal_epochs',1000)
    dppo_steps=raw.get('planner_dppo_steps',1000)
    batch_size=raw.get('upstream_batch_size',32)
    seed=raw.get('upstream_seed',42)
    device=raw.get('upstream_device','auto')
    for name,value,low,high in (
        ('planner_goal_epochs',goal_epochs,1,1_000_000),
        ('planner_dppo_steps',dppo_steps,1,1_000_000),
        ('upstream_batch_size',batch_size,1,4096),
        ('upstream_seed',seed,0,2**32-1),
    ):
        if isinstance(value,bool) or not isinstance(value,int) or not low<=value<=high:
            raise ValueError(f'{name} 必须是 [{low}, {high}] 内的整数')
    if device not in ('auto','cpu','cuda'):
        raise ValueError('设备只能是 auto、cpu 或 cuda')
    if device=='cuda':
        import torch
        if not torch.cuda.is_available():raise ValueError('当前环境没有可用 CUDA 设备')
    return {'task':'planner','goal_epochs':goal_epochs,'dppo_steps':dppo_steps,
            'batch_size':batch_size,'seed':seed,'device':device}


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
