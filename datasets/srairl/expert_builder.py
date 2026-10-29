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
        values = np.asarray(track["states"], dtype=np.float64)
        if t < values[0, 0] - 1e-6 or t > values[-1, 0] + 1e-6:
            return None
        return np.asarray([np.interp(t, values[:, 0], values[:, i]) for i in range(1, 5)])

    def build(self, scenario):
        roles = {"elcv": scenario.elcv_id, "rfv": scenario.rfv_id,
                 "rlv_current": scenario.rlv_current_id, "rlv_target": scenario.rlv_target_id}
        if any(v is None for v in roles.values()):
            raise ValueError("expert transitions require ELCV, RFV, current-lane leader and target-lane leader IDs")
        start = max(float(scenario.start_time), max(float(scenario.tracks[k]["states"][0][0]) for k in roles.values()))
        end = min(float(scenario.tracks[k]["states"][-1][0]) for k in roles.values())
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
        accelerations = {}
        for role in ("elcv", "rfv"):
            ident = roles[role]
            previous = self._sample(scenario.tracks[ident], float(times[0] - self.dt))
            if previous is None:
                raise ValueError(f"track {ident} needs one frame of pre-roll to calculate expert action")
            previous_velocity = previous[2:4]
            actions = []
            for t, row in zip(times, physical[role]):
                current = self._sample(scenario.tracks[ident], float(t))
                if current is None:
                    raise ValueError(f"track {ident} is missing at t={t:.3f}")
                actions.append((current[2:4] - previous_velocity) / self.dt)
                previous_velocity = current[2:4]
            accelerations[role] = np.asarray(actions, dtype=np.float64)
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
