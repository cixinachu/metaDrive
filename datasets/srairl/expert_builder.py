"""Build explicit dual-actor expert transitions from canonical NGSIM roles."""
from dataclasses import dataclass
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


def save_transitions(path, transitions, metadata):
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
    metadata_json=json.dumps(dict(schema_version=1,transition_pairs=len(ordered),**metadata),ensure_ascii=False)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,states=states,next_states=next_states,actions_elcv=actions_elcv,
        actions_rfv=actions_rfv,dones=dones,scenario_ids=scenario_ids,step_indices=step_indices,
        metadata=np.asarray(metadata_json))
    return len(ordered)


def load_transitions(path):
    """Load and validate a dataset previously written by save_transitions."""
    import json
    with np.load(path,allow_pickle=False) as data:
        metadata=json.loads(str(data['metadata'].item()))
        states=data['states'].astype(np.float32);next_states=data['next_states'].astype(np.float32)
        actions_e=data['actions_elcv'].astype(np.float32);actions_f=data['actions_rfv'].astype(np.float32)
        dones=data['dones'].astype(bool);scenario_ids=data['scenario_ids'].astype(str);steps=data['step_indices'].astype(np.int64)
    n=len(states)
    if not n or states.ndim!=2 or states.shape!=next_states.shape or actions_e.shape!=(n,2) or actions_f.shape!=(n,2) or len(dones)!=n or len(scenario_ids)!=n or len(steps)!=n:
        raise ValueError('invalid SR-AIRL expert dataset shapes')
    if not all(np.isfinite(x).all() for x in (states,next_states,actions_e,actions_f)):
        raise ValueError('SR-AIRL expert dataset contains non-finite values')
    result=[]
    for i in range(n):
        for role,actions in (('elcv',actions_e),('rfv',actions_f)):
            result.append(ExpertTransition(states[i].copy(),actions[i].copy(),next_states[i].copy(),
                bool(dones[i]),role,str(scenario_ids[i]),int(steps[i])))
    return result,metadata
