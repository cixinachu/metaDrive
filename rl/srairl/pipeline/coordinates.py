"""Single source of truth for upstream, lane-relative, and world coordinates."""
import numpy as np


def _frame(origin, heading):
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (2,) or not np.isfinite(origin).all() or not np.isfinite(heading):
        raise ValueError("origin and heading must be finite [x,y] and radians")
    tangent = np.array([np.cos(heading), np.sin(heading)])
    normal = np.array([-np.sin(heading), np.cos(heading)])
    return origin, tangent, normal


def upstream_to_world(states, origin=(0.0, 0.0), heading=0.0):
    """Map [y,x,vx,vy] in a straight road frame to world [x,y,vx,vy]."""
    data = np.asarray(states, dtype=float)
    if data.shape[-1] != 4 or not np.isfinite(data).all():
        raise ValueError("upstream states must be finite [...,4] [y,x,vx,vy]")
    origin, tangent, normal = _frame(origin, heading)
    xy = origin + data[..., 1, None] * tangent + data[..., 0, None] * normal
    velocity = data[..., 2, None] * tangent + data[..., 3, None] * normal
    return np.concatenate([xy, velocity], axis=-1)


def world_to_upstream(states, origin=(0.0, 0.0), heading=0.0):
    """Map world [x,y,vx,vy] to straight-road [y,x,vx,vy]."""
    data = np.asarray(states, dtype=float)
    if data.shape[-1] != 4 or not np.isfinite(data).all():
        raise ValueError("world states must be finite [...,4] [x,y,vx,vy]")
    origin, tangent, normal = _frame(origin, heading)
    delta = data[..., :2] - origin
    return np.stack([delta @ normal, delta @ tangent,
                     data[..., 2:] @ tangent, data[..., 2:] @ normal], axis=-1)


def upstream_to_lane_state(states, lane, origin=(0.0, 0.0), heading=0.0):
    """Convert [y,x,vx,vy] to tracker [s,d,vs,vd] using lane geometry.

    Tracker d is negative MetaDrive's signed lane lateral coordinate.
    """
    world = upstream_to_world(states, origin, heading)
    flat = world.reshape(-1, 4)
    result = []
    for x, y, vx, vy in flat:
        s, lateral = lane.local_coordinates(np.array([x, y]))
        theta = float(lane.heading_theta_at(s))
        tangent = np.array([np.cos(theta), np.sin(theta)])
        left_normal = np.array([-tangent[1], tangent[0]])
        velocity = np.array([vx, vy])
        result.append([s, -lateral, velocity @ tangent, velocity @ left_normal])
    return np.asarray(result).reshape(world.shape)


def lane_state_to_upstream(states, lane, origin=(0.0, 0.0), heading=0.0):
    """Convert tracker [s,d,vs,vd] to upstream [y,x,vx,vy]."""
    data = np.asarray(states, dtype=float)
    if data.shape[-1] != 4 or not np.isfinite(data).all():
        raise ValueError("lane states must be finite [...,4] [s,d,vs,vd]")
    flat = data.reshape(-1, 4)
    world = []
    for s, d, vs, vd in flat:
        # Tracker d is the negative of MetaDrive's signed lateral offset.
        xy = np.asarray(lane.position(s, -d), dtype=float)[:2]
        theta = float(lane.heading_theta_at(s))
        tangent = np.array([np.cos(theta), np.sin(theta)])
        left_normal = np.array([-tangent[1], tangent[0]])
        velocity = vs * tangent + vd * left_normal
        world.append([xy[0], xy[1], velocity[0], velocity[1]])
    world = np.asarray(world).reshape(data.shape)
    return world_to_upstream(world, origin, heading)
