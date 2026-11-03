"""Build four-role joint candidates with consistent horizons and units."""
import numpy as np

from .schema import JointCandidate, TrajectoryCandidate


def _trajectory(value, dt, horizon):
    if isinstance(value, TrajectoryCandidate):
        result = value
    else:
        result = TrajectoryCandidate(np.asarray(value), dt=dt)
    if len(result.states) != horizon or not np.isclose(result.dt, dt):
        raise ValueError("role trajectory does not match candidate horizon/dt")
    return result


def build_joint_candidates(elcv_candidates, response_predictor, rfv_history,
                           rlv_current_future, rlv_target_future, dt=0.1, seed=0):
    """Pair each ELCV proposal with exactly one guided RFV response."""
    candidates = []
    for index, elcv in enumerate(elcv_candidates):
        elcv_states = elcv.states if isinstance(elcv, TrajectoryCandidate) else np.asarray(elcv)
        if elcv_states.ndim != 2:
            raise ValueError("Planner proposal must be a [T,4] trajectory")
        elcv = _trajectory(elcv, dt, len(elcv_states))
        horizon = len(elcv.states)
        rfv = response_predictor.predict(
            rfv_history=rfv_history,
            elcv_future=elcv,
            rlv_current_future=rlv_current_future,
            rlv_target_future=rlv_target_future,
            seed=int(seed) + index,
        )
        candidates.append(JointCandidate(
            candidate_id=index, elcv=elcv, rfv=_trajectory(rfv, dt, horizon),
            rlv_current=_trajectory(rlv_current_future, dt, horizon),
            rlv_target=_trajectory(rlv_target_future, dt, horizon),
        ))
    return candidates
