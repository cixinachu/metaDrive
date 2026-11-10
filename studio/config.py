"""Validated UI inputs. Frozen v1 configuration files are never mutated."""
from pathlib import Path
import json
import math
from env.research_config import load_config, validate_config
from rl.srairl import reward_config as srairl_defaults
from datasets.srairl.teacher_controller import TEACHER_VERSION

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'outputs/studio'  # Preview cache and legacy runs
MODELS = ROOT/'models'
RESULTS = ROOT/'results'
JOB_ROOTS = (MODELS, RESULTS, RUNS)


def run_root(kind):
    return {'train': MODELS, 'eval': RESULTS, 'preview': RUNS}[kind]


def create_run_directory(kind, task, settings):
    """Allocate a new, descriptive directory even for identical repeated runs."""
    from datetime import datetime
    import hashlib
    import re
    import uuid

    def token(value):
        return re.sub(r'[^a-zA-Z0-9_-]+','-',str(value)).strip('-')[:60]

    operation='expert' if settings.get('operation')=='expert_collect' else task
    parts=[datetime.now().strftime('%Y%m%d_%H%M%S_%f'),kind,token(operation)]
    if task=='safe_rl':parts.append(token(settings.get('algorithm','sac')))
    scenes=settings.get('scenarios') or [settings.get('scenario')]
    scenes=[scene for scene in scenes if scene]
    if scenes:parts.append(token('-'.join(scenes)))
    if settings.get('traffic_density') is not None:
        parts.append('td'+format(float(settings['traffic_density']),'.6g').replace('.','p'))
    datasets=settings.get('dataset_id') or settings.get('expert_dataset_id')
    if datasets:
        fingerprint=hashlib.sha256(json.dumps(datasets,sort_keys=True).encode()).hexdigest()[:8]
        parts.append('data'+fingerprint)
    parts.append('seed'+str(settings.get('seed',0)))
    root=run_root(kind)
    root.mkdir(parents=True,exist_ok=True)
    while True:
        directory=root/('_'.join(parts)+'_'+uuid.uuid4().hex[:8])
        try:
            directory.mkdir()  # Never reuse or overwrite a previous run.
            return directory
        except FileExistsError:
            continue

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
    for key, choices in [('data_source', ['procedural']), ('algorithm', ALGORITHM_NAMES), ('scenario', SCENARIOS), ('device', ['auto','cpu','cuda']),
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
    environment_config(c, kind)
    return c


def _srairl_interaction_template(raw,lane_num):
    from dataclasses import asdict
    from env.tasks.srairl_interaction import SRAIRLInteractionScenario
    defaults=asdict(SRAIRLInteractionScenario());values={}
    try:
        for key,default in defaults.items():
            value=default if key.endswith('_speed') else raw.get('srairl_interaction_'+key,default)
            values[key]=int(value) if isinstance(default,int) and not isinstance(default,bool) else float(value) if isinstance(default,float) else value
        return SRAIRLInteractionScenario(**values).validate(lane_num)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError(f'Invalid SR-AIRL interaction template: {exc}') from exc


def _srairl_interaction_for_scenario(raw,lane_num,scenario_type):
    from dataclasses import asdict
    from env.tasks.srairl_interaction import with_scenario_speeds, SCENARIO_SPEED_CONFIG
    scenario=with_scenario_speeds(_srairl_interaction_template(raw,lane_num),scenario_type)
    values=asdict(scenario)
    # Overrides are opt-in and keyed by scenario; a selected scene must never
    # leak its speed into the other members of a collection batch.
    overrides = raw.get('srairl_speed_overrides', {})
    if not isinstance(overrides, dict):
        raise ValueError('srairl_speed_overrides must be a scenario mapping')
    if set(overrides) - {'straight', 'curve', 'merge'}:
        raise ValueError('unsupported speed override scenario')
    if not isinstance(raw.get('srairl_custom_speeds',False),bool):
        raise ValueError('srairl_custom_speeds must be boolean')
    if raw.get('srairl_custom_speeds', False):
        selected = overrides.get(scenario_type, {})
        if not isinstance(selected, dict) or set(selected) - set(SCENARIO_SPEED_CONFIG[scenario_type]):
            raise ValueError('invalid scenario speed overrides')
        for key, value in selected.items():
            values[key] = float(value)
    from env.tasks.srairl_interaction import SRAIRLInteractionScenario
    return asdict(SRAIRLInteractionScenario(**values).validate(lane_num))


def clean_srairl_request(raw, kind):
    """Validate the isolated SR-AIRL Studio interface."""
    if kind=='preview':
        preview=clean_srairl_request({**raw,'srairl_operation':'expert_collect',
            'srairl_demo_episodes':3,'srairl_collect_scenarios':[raw.get('srairl_scenario','straight')]},'train')
        preview['operation']='scene_preview'
        return preview
    if kind not in ('train','eval'):
        raise ValueError('SR-AIRL Studio supports preview, training and evaluation')
    operation=raw.get('srairl_operation','interaction')
    if operation not in ('interaction','expert_collect'):
        raise ValueError('未知 SR-AIRL 操作类型')
    if operation=='expert_collect':
        if kind!='train':raise ValueError('专家示范采集属于数据生成操作，请选择训练模式')
        scenario_type=raw.get('srairl_scenario','straight');lane_num=raw.get('srairl_lane_num',3)
        if scenario_type not in ('straight','curve','merge'):raise ValueError('专家采集场景无效')
        density=raw.get('srairl_traffic_density',.15);horizon=raw.get('srairl_horizon',1000)
        episodes=raw.get('srairl_demo_episodes',10);seed=raw.get('srairl_seed',srairl_defaults.SEED)
        scenario_types=raw.get('srairl_collect_scenarios') or [scenario_type]
        if not isinstance(scenario_types,(list,tuple)) or not scenario_types or any(name not in ('straight','curve','merge') for name in scenario_types):raise ValueError('MetaDrive 专家示范支持直道、弯道和匝道汇入，可单独或批量选择')
        scenario_types=list(dict.fromkeys(scenario_types))
        scenario_type=scenario_types[0]
        if isinstance(lane_num,bool) or not isinstance(lane_num,int) or not 2<=lane_num<=4:raise ValueError('车道数须为 2–4')
        if 'merge' in scenario_types and lane_num<3:raise ValueError('匝道汇入示范至少需要三条主路车道')
        if isinstance(density,bool) or not isinstance(density,(int,float)) or not math.isfinite(density) or not 0<=density<=1:raise ValueError('交通密度须在 [0, 1]')
        if isinstance(horizon,bool) or not isinstance(horizon,int) or not 200<=horizon<=100000:raise ValueError('示范回合步数须在 [200, 100000]，以容纳换道及后续轨迹')
        if isinstance(episodes,bool) or not isinstance(episodes,int) or not 3<=episodes<=10000:raise ValueError('每个场景至少采集 3 个 episode，才能按 episode 划分训练、验证和测试集')
        if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**32:raise ValueError('随机种子超出范围')
        if seed+(len(scenario_types)-1)*100_000+episodes-1>=2**32:
            raise ValueError('采集种子范围超出上限，请减小起始种子')
        view_mode=raw.get('srairl_view','topdown')
        if view_mode not in ('topdown','chase3d','first3d','bird3d'):raise ValueError('MetaDrive 画面视角无效')
        interaction_scenarios={current_scenario:_srairl_interaction_for_scenario(
            raw,lane_num,current_scenario) for current_scenario in scenario_types}
        base_interaction=interaction_scenarios[scenario_types[0]]
        return {'task':'srairl','operation':operation,'scene_source':'procedural','scenario':scenario_type,
            'scenarios':scenario_types,'interaction_scenarios':interaction_scenarios,
            'lane_num':lane_num,'traffic_density':float(density),'horizon':horizon,'demo_episodes':episodes,
            'seed':seed,'scene_seed':seed,'traffic_seed':(seed+1_000_003)%(2**32),'rl_seed':seed,
            'device':'cpu','srairl_view':view_mode,'dynamics_mode':'metadrive_physics',
            'interaction_scenario':base_interaction,'expert_controller':TEACHER_VERSION}
    scene_source=raw.get('srairl_scene_source','procedural')
    if scene_source!='procedural':
        raise ValueError('当前 SR-AIRL 实验只使用 MetaDrive 生成的道路和车辆')
    scenarios=raw.get('srairl_scenarios',[])
    expert_dataset_id=raw.get('srairl_expert_dataset_id',[])
    expert_dataset_id=[str(item).strip() for item in expert_dataset_id] if isinstance(expert_dataset_id,(list,tuple)) else [str(expert_dataset_id).strip()] if str(expert_dataset_id).strip() else []
    if expert_dataset_id:
        expert_root=(ROOT/'datasets'/'srairl'/'metadrive').resolve()
        expert_paths=[(ROOT/item).resolve() for item in expert_dataset_id]
        if kind!='train' or any(path.parent!=expert_root or not path.is_file() or path.suffix!='.npz' for path in expert_paths):
            raise ValueError('所选 MetaDrive 专家示范数据集无效')
        from datasets.srairl.expert_builder import load_transitions
        from rl.srairl.config import SRAIRLConfig
        expert_transitions=[];expert_metadata=[]
        for path in expert_paths:
            transitions,metadata=load_transitions(path)
            expert_transitions.extend(transitions);expert_metadata.append(metadata)
        from datasets.srairl.native_windows import native_dataset_info
        native_dataset_info(expert_paths,require_current=True)
        training_scenarios=sorted({str(name) for row in expert_metadata
            for name in (row.get('scenarios') or [row.get('scenario')]) if name in ('straight','curve','merge')})
        if not training_scenarios:
            raise ValueError('MetaDrive expert dataset metadata does not identify Straight, Curve, or Merge')
        config_dim=16*SRAIRLConfig.from_mapping().history_len
        if any(item.state.shape!=(config_dim,) or item.action.shape!=(2,) for item in expert_transitions):
            raise ValueError('MetaDrive 专家示范的状态/动作维度与当前 SR-AIRL 模型不匹配')
        scenario_traffic_densities={}
        for metadata in expert_metadata:
            by_scenario=metadata.get('traffic_density_by_scenario',{})
            for name in training_scenarios:
                value=by_scenario.get(name,metadata.get('traffic_density'))
                if value is None:continue
                value=float(value)
                if name in scenario_traffic_densities and not math.isclose(scenario_traffic_densities[name],value,abs_tol=1e-9):
                    raise ValueError(f'专家数据集对 {name} 场景记录了不一致的背景交通密度')
                scenario_traffic_densities[name]=value
        scenario_traffic_densities={name:scenario_traffic_densities.get(name,float(raw.get('srairl_traffic_density',.15)))
                                    for name in training_scenarios}
    else:
        expert_metadata=None
        training_scenarios=[]
        scenario_traffic_densities={}
    if kind=='train' and operation=='interaction' and not expert_dataset_id:
        raise ValueError('请先选择通过质量验收的 MetaDrive 专家示范数据集')
    initialization_protocol=raw.get('srairl_initialization','canonical')
    if initialization_protocol!='canonical':
        raise ValueError('SR-AIRL 初始状态只使用 MetaDrive 场景内的标准化初始布局')
    needs_experts=kind=='train' and not expert_dataset_id
    if scenarios:
        raise ValueError('当前 SR-AIRL 场景由 MetaDrive 程序化生成，不接受轨迹回放片段')
    checked=[]
    episodes=int(raw.get('srairl_episodes',1000))
    ppo_epochs=int(raw.get('srairl_ppo_epochs',80))
    rank_lambda=raw.get('srairl_rank_lambda',srairl_defaults.RANK_LAMBDA)
    if isinstance(rank_lambda,bool) or not isinstance(rank_lambda,(int,float)) or not math.isfinite(rank_lambda) or not 0<=rank_lambda<=10:
        raise ValueError('SR-AIRL 偏好排序权重须在 [0, 10]')
    seed=int(raw.get('srairl_seed',srairl_defaults.SEED))
    scene_seed=int(raw.get('srairl_scene_seed',seed))
    traffic_seed=int(raw.get('srairl_traffic_seed',(seed+1_000_003)%(2**32)))
    rl_seed=int(raw.get('srairl_rl_seed',seed))
    device=raw.get('srairl_device','auto')
    if not 1<=episodes<=srairl_defaults.MAX_EPISODES or not 1<=ppo_epochs<=1000 or not 0<=seed<2**32:
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
    dynamics_mode=raw.get('srairl_dynamics_mode','metadrive_physics')
    if dynamics_mode not in ('metadrive_physics','legacy_kinematic'):
        raise ValueError('SR-AIRL dynamics must be MetaDrive physics or upstream kinematics')
    if dynamics_mode=='metadrive_physics' and not meta_drive_world:
        raise ValueError('MetaDrive physics mode requires the MetaDrive world')
    scenario_type=raw.get('srairl_scenario','straight')
    if scenario_type not in ('straight','curve','merge'):
        raise ValueError('SR-AIRL MetaDrive 道路类型无效')
    lane_num=raw.get('srairl_lane_num',3)
    density=raw.get('srairl_traffic_density',.15)
    horizon=raw.get('srairl_horizon',1000)
    if isinstance(lane_num,bool) or not isinstance(lane_num,int) or not 2<=lane_num<=4:
        raise ValueError('SR-AIRL generated interaction maps require 2–4 lanes')
    if scenario_type=='merge' and lane_num<3:
        raise ValueError('MetaDrive ramp merge requires three mainline lanes')
    if isinstance(density,bool) or not isinstance(density,(int,float)) or not math.isfinite(density) or not 0<=density<=1:
        raise ValueError('SR-AIRL traffic density must be in [0, 1]')
    if isinstance(horizon,bool) or not isinstance(horizon,int) or not 50<=horizon<=100000:
        raise ValueError('SR-AIRL procedural horizon must be in [50, 100000]')
    if scene_source=='procedural' and dynamics_mode!='metadrive_physics':
        raise ValueError('MetaDrive-generated SR-AIRL scenes require MetaDrive vehicle physics')
    if scene_source=='procedural':
        # Procedural traffic is MetaDrive's IDM traffic manager, not NGSIM replay.
        reactive_background=False
    interaction=_srairl_interaction_for_scenario(raw,lane_num,scenario_type)
    scenario_interactions={name:_srairl_interaction_for_scenario(raw,lane_num,name)
                           for name in (training_scenarios or [scenario_type])}
    rfv_response=raw.get('srairl_rfv_response','policy')
    if rfv_response not in ('policy','fixed_speed','scripted_yield','scripted_gap_closing'):
        raise ValueError('RFV response must be policy, fixed_speed, scripted_yield or scripted_gap_closing')
    return {'srairl_scenarios':checked,'expert_dataset_id':expert_dataset_id,
            'expert_dataset_metadata':expert_metadata,'episodes':episodes,'ppo_epochs':ppo_epochs,'seed':seed,'device':device,
            'training_scenarios':training_scenarios,
            'traffic_density_by_scenario':scenario_traffic_densities,
            'rank_lambda':float(rank_lambda),
            'model_id':str(raw.get('srairl_model_id','')),'task':'srairl','dynamics_mode':dynamics_mode,
            'operation':operation,
            'metadrive_world':meta_drive_world,'srairl_view':view_mode,'srairl_reactive_background':reactive_background,
            'scene_source':scene_source,'scenario':scenario_type,'lane_num':lane_num,
            'traffic_density':float(density),'horizon':horizon,'scene_seed':scene_seed,
            'interaction_scenario':interaction,'rfv_response':rfv_response,
            'interaction_scenarios':scenario_interactions,
            'initialization_protocol':initialization_protocol,
            'traffic_seed':traffic_seed,'rl_seed':rl_seed}


def clean_reward_guide_request(raw, kind):
    if kind != 'train':
        raise ValueError('奖励引导器当前只提供训练；预测器接入后再开放组合评估')
    model_id = str(raw.get('srairl_model_id', '')).strip()
    if not model_id:
        raise ValueError('训练奖励引导器前，先选择已完成的 SR-AIRL 判别器模型')
    dataset_id=_native_component_dataset(raw)
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
    return {'task': 'reward_guide', 'srairl_model_id': model_id, 'dataset_id':dataset_id, 'epochs': epochs,
            'batch_size': batch_size, 'seed': seed, 'device': device}


def clean_predictor_request(raw, kind):
    if kind != 'train':
        raise ValueError('响应预测器当前只提供训练；候选评估流程尚未接入')
    model_id = str(raw.get('reward_guide_model_id', '')).strip()
    if not model_id:
        raise ValueError('训练预测器前，先选择已完成的奖励引导器模型')
    dataset_id=_native_component_dataset(raw)
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
    return {'task': 'predictor', 'reward_guide_model_id': model_id, 'dataset_id':dataset_id, 'epochs': epochs,
            'batch_size': batch_size, 'seed': seed, 'device': device}


def clean_planner_request(raw, kind):
    if kind != 'train':
        raise ValueError('规划器候选生成和闭环评估接口仍在接入；当前提供完整训练链')
    dataset_id=_native_component_dataset(raw)
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
    return {'task':'planner','dataset_id':dataset_id,'goal_epochs':goal_epochs,'dppo_steps':dppo_steps,
            'batch_size':batch_size,'seed':seed,'device':device}


def _native_component_dataset(raw):
    value=raw.get('srairl_expert_dataset_id',[])
    dataset_ids=[str(item).strip() for item in value] if isinstance(value,(list,tuple)) else [str(value).strip()] if str(value).strip() else []
    root=(ROOT/'datasets'/'srairl'/'metadrive').resolve()
    paths=[(ROOT/item).resolve() for item in dataset_ids]
    if not paths or any(path.parent!=root or path.suffix!='.npz' or not path.is_file() for path in paths):
        raise ValueError('请选择已采集且包含 train/validation/test 的 MetaDrive 专家示范数据集')
    from datasets.srairl.native_windows import native_dataset_info
    info=native_dataset_info(paths,require_current=True)
    if any(info['windows_by_split'].get(name,0)<1 for name in ('train','validation','test')):
        raise ValueError('MetaDrive 示范集的 train/validation/test 至少各需要一个完整 20+100 窗口')
    return [str(path.relative_to(ROOT)) for path in paths]


def clean_interaction_evaluation_request(raw):
    """Validate the integrated Planner → Predictor → SR-AIRL evaluation task."""
    scenario = raw.get('pipeline_scenario', 'straight')
    initialization = raw.get('pipeline_initialization', 'canonical')
    lane_num = raw.get('srairl_lane_num', 3)
    density = raw.get('srairl_traffic_density', .15)
    horizon = raw.get('srairl_horizon', 500)
    seed = raw.get('pipeline_seed', 0)
    episodes = raw.get('pipeline_episodes', 1)
    candidates = raw.get('pipeline_candidates', 16)
    replan_steps = raw.get('pipeline_replan_steps', 5)
    if scenario not in ('straight', 'curve', 'merge'):
        raise ValueError('综合评估场景只能是直道、弯道或匝道汇入')
    if initialization != 'canonical':
        raise ValueError('MetaDrive 综合评估只使用场景内的标准化初始布局')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('pipeline_seed must be an integer in [0, 2^32)')
    if isinstance(episodes, bool) or not isinstance(episodes, int) or not 1 <= episodes <= 1000:
        raise ValueError('综合评估回合数必须在 [1,1000]')
    if isinstance(candidates, bool) or not isinstance(candidates, int) or not 1 <= candidates <= 64:
        raise ValueError('Planner 候选数必须在 [1,64]')
    if isinstance(replan_steps, bool) or not isinstance(replan_steps, int) or not 1 <= replan_steps <= 20:
        raise ValueError('重规划步数必须在 [1,20]')
    raw_srairl = dict(raw)
    raw_srairl.update(srairl_operation='interaction', srairl_scene_source='procedural',
        srairl_dynamics_mode='metadrive_physics', srairl_initialization=initialization,
        srairl_scenario=scenario, srairl_model_id='pipeline-bundle',
        srairl_seed=seed, srairl_episodes=episodes, srairl_lane_num=lane_num,
        srairl_traffic_density=density, srairl_horizon=horizon,
        srairl_device=raw.get('srairl_device', 'auto'))
    base = clean_srairl_request(raw_srairl, 'eval')
    base.update(task='evaluation', operation='closed_loop', pipeline_scenario=scenario,
        initialization_protocol=initialization, seed=seed, episodes=episodes,
        candidate_count=candidates, replan_steps=replan_steps,
        view_mode=raw.get('srairl_view', 'topdown'))
    return base


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
