"""Forecast the two explicitly scripted/recorded RLV roles only."""
import numpy as np

from .schema import TrajectoryCandidate


def forecast_rlv_roles(env, history, horizon=100, dt=0.1):
    if horizon != 100 or not np.isclose(dt, 0.1):
        raise ValueError("current SR-AIRL runtime is trained for horizon=100, dt=0.1")
    recorded_scene = getattr(env, "scene_source", None) in {"ngsim", "recorded"}
    recorded_scene = recorded_scene or (
        not hasattr(env, "scenario_type") and hasattr(getattr(env, "scenario", None), "tracks")
    )
    if recorded_scene:
        scenario = env.scenario
        futures = []
        for role in ("rlv_current", "rlv_target"):
            ident = getattr(scenario, f"{role}_id")
            track = scenario.tracks[ident]
            rows = []
            for index in range(horizon):
                t = float(env.time) + (index + 1) * dt
                state = env._sample(track, t)
                if state is None:
                    raise ValueError(f"recorded {role} trajectory ends before planner horizon at t={t:.2f}")
                x, y, vx, vy = state
                rows.append([y, x, vx, vy])
            futures.append(TrajectoryCandidate(np.asarray(rows), dt=dt))
        return tuple(futures)

    # Forecast from observed reference-frame motion only. Scripted RLV bodies
    # have no navigation module, and their hidden future simulator path is not
    # an observation available to a deployed planner.
    output=[]
    times=(np.arange(horizon,dtype=float)+1)*dt
    for role in ('rlv_current','rlv_target'):
        observed=np.asarray(history[role],dtype=float)
        if observed.shape!=(20,4) or not np.isfinite(observed).all():
            raise ValueError('RLV forecast requires 20 finite observed reference states')
        rows=np.repeat(observed[-1:,:],horizon,axis=0)
        rows[:,0]+=rows[:,3]*times  # shared lateral position and velocity point left
        rows[:,1]+=rows[:,2]*times
        output.append(TrajectoryCandidate(rows,dt=dt))
    return tuple(output)
