"""Build explicit dual-actor transitions from MetaDrive expert episodes."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from env.adapters.srairl_observation import SRAIRLObservationAdapter
from rl.srairl.config import SRAIRLConfig


@dataclass(frozen=True)
class ExpertTransition:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    done: bool
    agent: str
    scenario_id: str
    step_index: int = 0


class ExpertDatasetBuilder:
    """Separate expert extraction from environment reset/step side effects."""
    def __init__(self, history_len=None, dt=0.1, config=None):
        self.config = config or SRAIRLConfig.from_mapping()
        history_len = self.config.history_len if history_len is None else history_len
        if history_len < 1 or dt <= 0:
            raise ValueError("history_len and dt must be positive")
        self.history_len, self.dt = int(history_len), float(dt)
        self.observer = SRAIRLObservationAdapter(self.history_len,
            pos_lat_scale=self.config.pos_lat_scale, pos_long_scale=self.config.pos_long_scale,
            vel_scale=self.config.vel_scale, vel_offset=self.config.vel_offset)

    @staticmethod
    def _sample(track, t):
        from datasets.srairl.kinematics import state
        return state(track,t,dt=.1,require_preroll=True)

    def build(self, scenario):
        roles = {"elcv": scenario.elcv_id, "rfv": scenario.rfv_id,
                 "rlv_current": scenario.rlv_current_id, "rlv_target": scenario.rlv_target_id}
        if any(v is None for v in roles.values()):
            raise ValueError("expert transitions require ELCV, RFV, current-lane leader and target-lane leader IDs")
        start = max(float(scenario.start_time), max(float(scenario.tracks[k]["states"][0][0]) for k in roles.values()))
        end = min(float(scenario.tracks[k]["states"][-1][0]) for k in roles.values())
        if scenario.end_time is not None:end=min(end,float(scenario.end_time))
        if end - start < self.dt:
            raise ValueError("role tracks have no overlapping interval")
        times = np.arange(start, end + 1e-8, self.dt)
        frames = []
        physical = {name: [] for name in roles}
        for t in times:
            frame_roles = {}
            for role, ident in roles.items():
                raw = self._sample(scenario.tracks[ident], float(t))
                if raw is None:
                    raise ValueError(f"role {role} is missing at t={t:.3f}")
                x, y, vx, vy = raw
                frame_roles[role] = np.asarray([y, x, vx, vy])
                physical[role].append([x, y, vx, vy])
            frames.append(self.observer.frame(frame_roles))
        frames = np.asarray(frames, dtype=np.float32)
        physical = {k: np.asarray(v, dtype=np.float64) for k, v in physical.items()}
        from datasets.srairl.kinematics import acceleration
        accelerations = {}
        for role in ("elcv", "rfv"):
            ident = roles[role]
            accelerations[role] = np.asarray([acceleration(scenario.tracks[ident],float(t),self.dt) for t in times],dtype=np.float64)
        stacked = []
        for i in range(len(frames)):
            seq = frames[max(0, i - self.history_len + 1):i + 1]
            if len(seq) < self.history_len:
                seq = np.concatenate([np.repeat(frames[:1], self.history_len-len(seq), axis=0), seq], axis=0)
            stacked.append(seq.reshape(-1))
        result = []
        for i in range(len(frames)-1):
            next_seq = frames[max(0, i + 1 - self.history_len + 1):i + 2]
            if len(next_seq) < self.history_len:
                next_seq = np.concatenate([np.repeat(frames[:1], self.history_len-len(next_seq), axis=0), next_seq], axis=0)
            next_state = next_seq.reshape(-1).astype(np.float32)
            for role in ("elcv", "rfv"):
                result.append(ExpertTransition(stacked[i].astype(np.float32),
                    accelerations[role][i].astype(np.float32), next_state.copy(), i == len(frames)-2,
                    role, scenario.scenario_id, i))
        return result


def save_transitions(path, transitions, metadata, raw_episodes=None):
    """Persist portable, validated paired SR-AIRL transitions as a compressed archive."""
    from pathlib import Path
    import json
    transitions=list(transitions)
    paired={}
    for item in transitions:
        key=(item.scenario_id,int(item.step_index))
        if item.agent not in ('elcv','rfv') or item.agent in paired.setdefault(key,{}):
            raise ValueError('expert data must contain one ELCV and one RFV transition per step')
        paired[key][item.agent]=item
    if not paired or any(set(pair)!={'elcv','rfv'} for pair in paired.values()):
        raise ValueError('expert data must contain complete ELCV/RFV transition pairs')
    ordered=[paired[key] for key in sorted(paired)]
    for pair in ordered:
        a,b=pair['elcv'],pair['rfv']
        if not (np.array_equal(a.state,b.state) and np.array_equal(a.next_state,b.next_state) and a.done==b.done):
            raise ValueError('paired expert transitions must share state, next_state and done')
    states=np.stack([pair['elcv'].state for pair in ordered]).astype(np.float32)
    next_states=np.stack([pair['elcv'].next_state for pair in ordered]).astype(np.float32)
    actions_elcv=np.stack([pair['elcv'].action for pair in ordered]).astype(np.float32)
    actions_rfv=np.stack([pair['rfv'].action for pair in ordered]).astype(np.float32)
    dones=np.asarray([pair['elcv'].done for pair in ordered],dtype=np.bool_)
    scenario_ids=np.asarray([pair['elcv'].scenario_id for pair in ordered],dtype='U128')
    step_indices=np.asarray([pair['elcv'].step_index for pair in ordered],dtype=np.int64)
    if not all(np.isfinite(array).all() for array in (states,next_states,actions_elcv,actions_rfv)):
        raise ValueError('expert transitions contain non-finite values')
    accepted=[episode for episode in (raw_episodes or []) if episode.get('validation',{}).get('valid')]
    episode_splits={}
    if accepted:
        from datasets.srairl.native_windows import assign_episode_splits
        try:labels=assign_episode_splits(accepted)
        except ValueError:labels=['unassigned']*len(accepted)
        for episode,label in zip(accepted,labels):
            episode['split']=label
            episode_splits[(str(episode.get('scenario','')),int(episode['seed']))]=label
    transition_splits=[]
    for pair in ordered:
        ident=pair['elcv'].scenario_id
        match=__import__('re').search(r'^(?:metadrive-)?(.+?)-seed(-?\d+)$',ident)
        transition_splits.append(episode_splits.get((match.group(1),int(match.group(2))), 'train') if match else 'train')
    split_protocol=('scenario_grouped_seed_70_15_15_v1' if accepted and all(x.get('split')!='unassigned' for x in accepted)
                    else 'unassigned_pending_grouped_split_v1' if accepted else None)
    metadata_json=json.dumps(dict(schema_version=2 if accepted else 1,transition_pairs=len(ordered),
                                  split_protocol=split_protocol,**metadata),ensure_ascii=False)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    arrays=dict(states=states,next_states=next_states,actions_elcv=actions_elcv,
        actions_rfv=actions_rfv,dones=dones,scenario_ids=scenario_ids,step_indices=step_indices,
        metadata=np.asarray(metadata_json))
    if accepted:
        arrays['transition_splits']=np.asarray(transition_splits,dtype='U16')
    if accepted:
        state_offsets=[0];action_offsets=[0]
        for episode in accepted:
            state_offsets.append(state_offsets[-1]+len(episode['role_states']))
            action_offsets.append(action_offsets[-1]+len(episode['desired_acceleration']))
        arrays.update(
            role_order=np.asarray(accepted[0]['role_order'],dtype='U32'),
            role_states=np.concatenate([np.asarray(x['role_states'],dtype=np.float64) for x in accepted]),
            trajectory_times_s=np.concatenate([np.asarray(x['times_s'],dtype=np.float64) for x in accepted]),
            trajectory_state_offsets=np.asarray(state_offsets,dtype=np.int64),
            trajectory_desired_acceleration=np.concatenate([np.asarray(x['desired_acceleration'],dtype=np.float32) for x in accepted]),
            trajectory_metadrive_actions=np.concatenate([np.asarray(x['metadrive_actions'],dtype=np.float32) for x in accepted]),
            trajectory_actual_acceleration=np.concatenate([np.asarray(x['actual_acceleration'],dtype=np.float32) for x in accepted]),
            trajectory_reference_states_frenet=np.concatenate([np.asarray(x['reference_states_frenet'],dtype=np.float64) for x in accepted]),
            trajectory_lane_indices=np.asarray([lane for x in accepted for lane in x['lane_indices']],dtype='U64'),
            trajectory_action_offsets=np.asarray(action_offsets,dtype=np.int64),
            trajectory_metadata=np.asarray(json.dumps([{k:v for k,v in x.items() if k not in {
                'role_order','role_states','reference_states_frenet','lane_indices','times_s',
                'desired_acceleration','metadrive_actions','actual_acceleration','tracking_errors'}}
                for x in accepted],ensure_ascii=False,allow_nan=False)))
        if all('tracking_errors' in x for x in accepted):
            arrays['trajectory_tracking_errors']=np.concatenate([
                np.asarray(x['tracking_errors'],dtype=np.float32) for x in accepted])
    np.savez_compressed(path,**arrays)
    return len(ordered)


def load_transitions(path, split='all'):
    """Load and validate a dataset previously written by save_transitions."""
    import json
    with np.load(path,allow_pickle=False) as data:
        metadata=json.loads(str(data['metadata'].item()))
        states=data['states'].astype(np.float32);next_states=data['next_states'].astype(np.float32)
        actions_e=data['actions_elcv'].astype(np.float32);actions_f=data['actions_rfv'].astype(np.float32)
        dones=data['dones'].astype(bool);scenario_ids=data['scenario_ids'].astype(str);steps=data['step_indices'].astype(np.int64)
        selected=np.ones(len(states),dtype=bool)
        if split!='all':
            if split not in ('train','validation','test'):
                raise ValueError('split must be train, validation, test, or all')
            if 'transition_splits' not in data.files:
                raise ValueError('legacy transition archive has no episode-isolated splits')
            selected=data['transition_splits'].astype(str)==split
    n=len(states)
    if not n or states.ndim!=2 or states.shape!=next_states.shape or actions_e.shape!=(n,2) or actions_f.shape!=(n,2) or len(dones)!=n or len(scenario_ids)!=n or len(steps)!=n:
        raise ValueError('invalid SR-AIRL expert dataset shapes')
    if not all(np.isfinite(x).all() for x in (states,next_states,actions_e,actions_f)):
        raise ValueError('SR-AIRL expert dataset contains non-finite values')
    result=[]
    for i in range(n):
        if not selected[i]:
            continue
        for role,actions in (('elcv',actions_e),('rfv',actions_f)):
            result.append(ExpertTransition(states[i].copy(),actions[i].copy(),next_states[i].copy(),
                bool(dones[i]),role,str(scenario_ids[i]),int(steps[i])))
    return result,metadata


def merge_expert_archives(paths, output_path, metadata=None):
    """Merge scenario archives while retaining episode-wise split assignment."""
    import json
    import hashlib
    paths=[Path(path) for path in paths]
    if not paths:
        raise ValueError('at least one MetaDrive expert archive is required')
    transitions=[];episodes=[];sources=[];seen_episodes=set();geometry_version=None;reference_version=None
    for path in paths:
        current,current_metadata=load_transitions(path,split='all')
        transitions.extend(current);sources.append({'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        with np.load(path,allow_pickle=False) as archive:
            if 'trajectory_metadata' not in archive.files:
                raise ValueError(f'{path} has no full four-role MetaDrive trajectories')
            episode_metadata=json.loads(str(archive['trajectory_metadata'].item()))
            state_offsets=archive['trajectory_state_offsets'].astype(np.int64)
            action_offsets=archive['trajectory_action_offsets'].astype(np.int64)
            for index,row in enumerate(episode_metadata):
                identity=(str(row.get('scenario','')),int(row['seed']))
                if identity in seen_episodes:
                    raise ValueError(f'duplicate expert episode/seed: {identity}')
                seen_episodes.add(identity)
                version=row.get('road_geometry',{}).get('version')
                if not version:
                    raise ValueError('legacy expert archive has no road geometry provenance; recollect for formal training')
                if geometry_version is not None and version != geometry_version:
                    raise ValueError('cannot merge different road geometry versions')
                geometry_version=version
                current_reference=row.get('reference_frame_version')
                if not current_reference or (reference_version is not None and reference_version!=current_reference):
                    raise ValueError('cannot merge missing or different reference coordinate versions')
                reference_version=current_reference
                ss,se=map(int,state_offsets[index:index+2]);aa,ae=map(int,action_offsets[index:index+2])
                episodes.append({**row,'role_order':archive['role_order'].astype(str).tolist(),
                    'role_states':archive['role_states'][ss:se].copy(),
                    'reference_states_frenet':archive['trajectory_reference_states_frenet'][ss:se].copy(),
                    'lane_indices':archive['trajectory_lane_indices'][ss:se].astype(str).tolist(),
                    'times_s':archive['trajectory_times_s'][ss:se].copy(),
                    'desired_acceleration':archive['trajectory_desired_acceleration'][aa:ae].copy(),
                    'metadrive_actions':archive['trajectory_metadrive_actions'][aa:ae].copy(),
                    'actual_acceleration':archive['trajectory_actual_acceleration'][aa:ae].copy(),
                    **({'tracking_errors':archive['trajectory_tracking_errors'][aa:ae].copy()} if 'trajectory_tracking_errors' in archive.files else {})})
    merged_metadata={'source':'MetaDrive','scene_source':'procedural','source_archives':sources,
                     'scenarios':sorted({str(row.get('scenario','unknown')) for row in episodes}),
                     **(metadata or {})}
    save_transitions(output_path,transitions,merged_metadata,episodes)
    return Path(output_path)
