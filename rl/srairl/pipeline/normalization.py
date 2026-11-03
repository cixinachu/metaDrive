"""Named normalizers matching the local upstream training datasets."""
import numpy as np


class PlannerGoalNormalizer:
    """Step2 goal-history scaling: y/50, x/500, (v+40)/80."""
    version = "planner-goal-v1"

    @staticmethod
    def transform(states):
        x = np.asarray(states, dtype=np.float32).copy()
        if x.shape[-1] != 4:
            raise ValueError("expected [...,4] [y,x,vx,vy]")
        x[..., 0] /= 50.0
        x[..., 1] /= 500.0
        x[..., 2:] = (x[..., 2:] + 40.0) / 80.0
        return x


class RewardPredictorNormalizer:
    """Step3/4 normalize_obs scales [y,x,vx,vy] and clips to [-1,1]."""
    version = "reward-predictor-v1"

    @staticmethod
    def transform(states):
        x = np.asarray(states, dtype=np.float32).copy()
        if x.shape[-1] < 4:
            raise ValueError("expected at least four features [y,x,vx,vy]")
        x[..., 0] = np.clip(x[..., 0] / 25.0, -1.0, 1.0)
        x[..., 1] = np.clip(x[..., 1] / 500.0, -1.0, 1.0)
        x[..., 2:4] = np.clip((x[..., 2:4] + 40.0) / 80.0, -1.0, 1.0)
        return x


def reconstruct_predictor_deltas(history, deltas, dt=0.1):
    """Reconstruct [dy,dx,dvx,dvy] samples to physical [y,x,vx,vy]."""
    hist = np.asarray(history, dtype=np.float64)
    delta = np.asarray(deltas, dtype=np.float64)
    if hist.ndim != 2 or hist.shape[1] != 4 or delta.ndim != 2 or delta.shape[1] != 4:
        raise ValueError("history and deltas must be [N,4] and [T,4]")
    if not len(hist) or not np.isfinite(hist).all() or not np.isfinite(delta).all() or dt <= 0:
        raise ValueError("inputs and dt must be finite and non-empty")
    future = hist[-1][None, :] + np.cumsum(delta, axis=0)
    accel = np.diff(np.vstack([hist[-1], future]), axis=0)[:, 2:4] / dt
    return future, accel
