"""Episode-grouped windows from validated native MetaDrive demonstrations."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


ROLE_ORDER = ("elcv", "rfv", "rlv_current", "rlv_target")
HISTORY_STEPS = 20
FUTURE_STEPS = 100


def assign_episode_splits(metadata):
    """Split whole seeds independently within each MetaDrive road scenario."""
    groups={}
    for row in metadata:
        groups.setdefault(str(row.get('scenario','unknown')),set()).add(int(row['seed']))
    split_by_episode={}
    for scenario,seeds in groups.items():
        ordered=sorted(seeds);n=len(ordered)
        if n<3:
            raise ValueError(f'{scenario} needs at least three accepted episode seeds for train/validation/test')
        n_train=max(1,int(n*.70));n_validation=max(1,int(n*.15))
        if n_train+n_validation>=n:n_train,n_validation=n-2,1
        for position,seed in enumerate(ordered):
            split='train' if position<n_train else 'validation' if position<n_train+n_validation else 'test'
            split_by_episode[(scenario,seed)]=split
    return [split_by_episode[(str(row.get('scenario','unknown')),int(row['seed']))] for row in metadata]


@dataclass(frozen=True)
class NativeWindow:
    episode_seed: int
    split: str
    history: np.ndarray       # [20, role, y, x, vx, vy, ay, ax]
    future: np.ndarray        # [100, role, y, x, vx, vy, ay, ax]
    state: np.ndarray         # [100, 16], normalized joint interaction state
    history_state: np.ndarray # [20, 16], normalized joint history through decision time
    actions: np.ndarray       # [100, 2, a_long, a_lat]


def _validate_archive(data):
    required = {
        "role_order", "role_states", "trajectory_times_s", "trajectory_state_offsets",
        "trajectory_desired_acceleration", "trajectory_action_offsets",
        "trajectory_metadrive_actions","trajectory_actual_acceleration",
        "trajectory_reference_states_frenet","trajectory_lane_indices","trajectory_metadata",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"MetaDrive expert archive lacks raw trajectories: {missing}")
    roles = tuple(data["role_order"].astype(str).tolist())
    if roles != ROLE_ORDER:
        raise ValueError(f"unsupported MetaDrive role order: {roles}")
    state_offsets = data["trajectory_state_offsets"].astype(np.int64)
    action_offsets = data["trajectory_action_offsets"].astype(np.int64)
    states = data["role_states"]
    times = data["trajectory_times_s"]
    actions = data["trajectory_desired_acceleration"]
    mapped_actions=data["trajectory_metadrive_actions"]
    measured_actions=data["trajectory_actual_acceleration"]
    reference=data["trajectory_reference_states_frenet"]
    lane_indices=data["trajectory_lane_indices"]
    metadata = json.loads(str(data["trajectory_metadata"].item()))
    if len(state_offsets) != len(metadata) + 1 or len(action_offsets) != len(metadata) + 1:
        raise ValueError("MetaDrive episode offsets do not match episode metadata")
    if state_offsets[0] != 0 or state_offsets[-1] != len(states) or len(times) != len(states):
        raise ValueError("invalid MetaDrive trajectory state offsets")
    if reference.shape!=states.shape or lane_indices.shape!=(len(states),4):
        raise ValueError("MetaDrive reference coordinates or lane IDs do not align with four-role states")
    if action_offsets[0] != 0 or action_offsets[-1] != len(actions):
        raise ValueError("invalid MetaDrive trajectory action offsets")
    if mapped_actions.shape!=actions.shape or measured_actions.shape!=actions.shape or actions.ndim!=3 or actions.shape[1:]!=(2,2):
        raise ValueError("MetaDrive desired, mapped and measured actor actions are not aligned")
    if not all(np.isfinite(array).all() for array in (states,times,actions,mapped_actions,measured_actions,reference)):
        raise ValueError("MetaDrive native archive contains non-finite trajectory data")
    return states, times, actions, state_offsets, action_offsets, metadata


def native_dataset_info(path):
    """Validate a reusable dataset and report available episode-level splits."""
    paths=[Path(item) for item in path] if isinstance(path,(list,tuple)) else [Path(path)]
    rows=[];window_counts=[]
    for source in paths:
        with np.load(source,allow_pickle=False) as archive:
            _states,_times,_actions,state_offsets,action_offsets,metadata=_validate_archive(archive)
            for index,row in enumerate(metadata):
                start,end=state_offsets[index:index+2]
                action_start,action_end=action_offsets[index:index+2]
                history_steps=int(row.get('warmup_steps',HISTORY_STEPS))
                n_windows=max(0,min(end-start-FUTURE_STEPS,
                    history_steps+(action_end-action_start)-FUTURE_STEPS+1)-history_steps)
                rows.append(row);window_counts.append(n_windows)
    assigned=assign_episode_splits(rows)
    windows=0;by_split={key:0 for key in ('train','validation','test')}
    by_scenario={}
    for row,split,n_windows in zip(rows,assigned,window_counts):
        scenario=str(row.get('scenario','unknown'))
        by_split[split]+=n_windows;windows+=n_windows
        by_scenario.setdefault(scenario,{key:0 for key in ('train','validation','test')})[split]+=n_windows
    splits=set(assigned)
    if windows < 3 or not {'train','validation','test'}.issubset(splits):
        raise ValueError("MetaDrive dataset needs complete train/validation/test episode splits")
    if any(by_split[key] < 1 for key in ("train", "validation", "test")):
        raise ValueError(f"MetaDrive dataset has an empty split: {by_split}")
    if any(any(counts[key]<1 for key in ('train','validation','test')) for counts in by_scenario.values()):
        raise ValueError(f"Every MetaDrive scenario needs at least one 20+100 window in each split: {by_scenario}")
    display_path=str(paths[0]) if len(paths)==1 else [str(item) for item in paths]
    return dict(path=display_path,episodes=len(rows),windows=windows,windows_by_split=by_split,
                episode_seeds=[int(row['seed']) for row in rows],windows_by_scenario=by_scenario,
                split_protocol='scenario_grouped_seed_70_15_15_v1')


def _coordinates(states, current_index):
    """Express world trajectories in a fixed ELCV frame at the decision time."""
    origin = states[current_index, 0, :2]
    velocity = states[current_index, 0, 2:4]
    speed = float(np.linalg.norm(velocity))
    tangent = velocity / speed if speed > .1 else np.asarray([1., 0.])
    normal = np.asarray([-tangent[1], tangent[0]])
    delta = states[:, :, :2] - origin
    velocity = states[:, :, 2:4]
    longitudinal = np.einsum("trd,d->tr", delta, tangent)
    lateral = np.einsum("trd,d->tr", delta, normal)
    v_longitudinal = np.einsum("trd,d->tr", velocity, tangent)
    v_lateral = np.einsum("trd,d->tr", velocity, normal)
    return np.stack((lateral, longitudinal, v_longitudinal, v_lateral), axis=-1)


def _with_acceleration(coords):
    acceleration = np.zeros_like(coords[..., 2:4])
    acceleration[1:] = np.diff(coords[..., 2:4], axis=0) / .1
    return np.concatenate((coords, acceleration), axis=-1)


def _normalize_joint(coords):
    lows = np.asarray([0., 0., -40., -40.], dtype=np.float64)
    spans = np.asarray([25., 500., 80., 80.], dtype=np.float64)
    normalized = np.clip((coords[..., :4] - lows) / spans, -1., 1.)
    ego = normalized[:, 0, :]
    return np.concatenate((ego, ego - normalized[:, 1, :],
                           normalized[:, 2, :] - ego, normalized[:, 3, :] - ego), axis=-1)


def load_native_windows(path, split="train"):
    """Load full 20-history/100-future windows without pandas or pickle."""
    if split not in ("train", "validation", "test", "all"):
        raise ValueError("split must be train, validation, test, or all")
    windows = []
    with np.load(path, allow_pickle=False) as archive:
        states, _times, actions, state_offsets, action_offsets, metadata = _validate_archive(archive)
        assigned = assign_episode_splits(metadata)
        for index, row in enumerate(metadata):
            ep_split = str(row.get("split", assigned[index]))
            if split != "all" and ep_split != split:
                continue
            ss, se = map(int, state_offsets[index:index + 2])
            aa, ae = map(int, action_offsets[index:index + 2])
            raw = np.asarray(states[ss:se], dtype=np.float64)
            episode_actions = np.asarray(actions[aa:ae], dtype=np.float32)
            history_steps = int(row.get("warmup_steps", HISTORY_STEPS))
            first = max(history_steps, HISTORY_STEPS - 1)
            last_exclusive = min(len(raw) - FUTURE_STEPS, history_steps + len(episode_actions) - FUTURE_STEPS + 1)
            for current in range(first, last_exclusive):
                history_idx = np.arange(current - HISTORY_STEPS + 1, current + 1)
                future_idx = np.arange(current + 1, current + FUTURE_STEPS + 1)
                coords = _with_acceleration(_coordinates(raw, current))
                history = coords[history_idx]
                future = coords[future_idx]
                action_start = current - history_steps
                action_window = episode_actions[action_start:action_start + FUTURE_STEPS]
                if len(action_window) != FUTURE_STEPS:
                    continue
                windows.append(NativeWindow(int(row["seed"]), ep_split, history, future,
                                            _normalize_joint(future),_normalize_joint(history),action_window.copy()))
    return windows
