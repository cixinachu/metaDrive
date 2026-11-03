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
                t = float(env.time) + index * dt
                state = env._sample(track, t)
                if state is None:
                    raise ValueError(f"recorded {role} trajectory ends before planner horizon at t={t:.2f}")
                x, y, vx, vy = state
                rows.append([y, x, vx, vy])
            futures.append(TrajectoryCandidate(np.asarray(rows), dt=dt))
        return tuple(futures)

    world = getattr(env, "world", None)
    if world is None:
        raise TypeError("procedural role forecasting requires a MetaDrive world")
    if getattr(env, "scenario_type", None) == "merge":
        paths = getattr(world, "_interaction_paths", None)
        if not paths:
            raise RuntimeError("merge RLV route forecasts are unavailable")
        output = []
        for role in ("rlv_current", "rlv_target"):
            path, s0, speed = paths[role]
            rows = []
            elapsed = float(getattr(world, "_interaction_elapsed", 0.0))
            for index in range(horizon):
                lane, local = world._path_position(path, s0 + speed * (elapsed + index * dt))
                position = np.asarray(lane.position(local, 0.0), dtype=float)
                heading = float(lane.heading_theta_at(local))
                rows.append([position[1], position[0], speed * np.cos(heading), speed * np.sin(heading)])
            output.append(TrajectoryCandidate(np.asarray(rows), dt=dt))
        return tuple(output)

    reference = env._source_lane
    output = []
    for role in ("rlv_current", "rlv_target"):
        vehicle = world.role_vehicle(role)
        lane = vehicle.lane
        s0, lateral = lane.local_coordinates(vehicle.position[:2])
        speed = float(np.dot(np.asarray(vehicle.velocity[:2]),
                             [np.cos(lane.heading_theta_at(s0)), np.sin(lane.heading_theta_at(s0))]))
        rows = []
        for index in range(horizon):
            local = float(s0) + speed * index * dt
            position = np.asarray(lane.position(local, lateral), dtype=float)[:2]
            heading = float(lane.heading_theta_at(local))
            velocity = speed * np.asarray([np.cos(heading), np.sin(heading)])
            source_s, source_d = reference.local_coordinates(position)
            source_heading = float(reference.heading_theta_at(source_s))
            tangent = np.asarray([np.cos(source_heading), np.sin(source_heading)])
            normal = np.asarray([-tangent[1], tangent[0]])
            rows.append([source_d, source_s, velocity @ tangent, velocity @ normal])
        output.append(TrajectoryCandidate(np.asarray(rows), dt=dt))
    return tuple(output)
