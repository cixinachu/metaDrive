"""Track lane-relative SR-AIRL candidate trajectories with calibrated controls."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackerGains:
    longitudinal_position: float = 0.35
    longitudinal_speed: float = 1.1
    lateral_position: float = 1.2
    lateral_speed: float = 1.8
    max_longitudinal_accel: float = 3.0
    min_longitudinal_accel: float = -5.0
    max_lateral_accel: float = 3.0


class SRAIRLTrajectoryTracker:
    """Convert [s, d, v_s, v_d] candidate states to desired accelerations.

    ``s`` is MetaDrive lane longitude; ``d`` is lateral displacement along the
    lane's left-hand normal (the negative of MetaDrive's signed lane offset).
    This matches the world-frame lateral acceleration measured by calibration.
    The action adapter alone converts desired accelerations to vehicle input.
    """

    def __init__(self, action_adapter, gains=None):
        self.action_adapter = action_adapter
        self.gains = gains or TrackerGains()

    @staticmethod
    def lane_state(vehicle, lane):
        longitudinal, lateral = lane.local_coordinates(vehicle.position[:2])
        heading = float(lane.heading_theta_at(longitudinal))
        tangent = np.asarray([np.cos(heading), np.sin(heading)])
        normal = np.asarray([-tangent[1], tangent[0]])
        velocity = np.asarray(vehicle.velocity[:2], dtype=float)
        return np.asarray([longitudinal, -lateral, velocity @ tangent, velocity @ normal], dtype=float)

    @staticmethod
    def path_curvature(lane,s):
        lo=max(0.,float(s)-.5);hi=min(float(lane.length),float(s)+.5)
        if hi<=lo:return 0.
        delta=(lane.heading_theta_at(hi)-lane.heading_theta_at(lo)+np.pi)%(2*np.pi)-np.pi
        return float(delta/(hi-lo))

    def desired_acceleration(self, current_state, target_state, *, curvature=0.):
        current = np.asarray(current_state, dtype=float)
        target = np.asarray(target_state, dtype=float)
        if current.shape != (4,) or target.shape != (4,) or not np.isfinite(current).all() or not np.isfinite(target).all():
            raise ValueError("current and target lane states must be finite [s, d, v_s, v_d]")
        g = self.gains
        longitudinal = g.longitudinal_position * (target[0] - current[0]) + g.longitudinal_speed * (target[2] - current[2])
        lateral = float(target[2])**2*float(curvature) + g.lateral_position * (target[1] - current[1]) + g.lateral_speed * (target[3] - current[3])
        return np.asarray([
            np.clip(longitudinal, g.min_longitudinal_accel, g.max_longitudinal_accel),
            np.clip(lateral, -g.max_lateral_accel, g.max_lateral_accel),
        ], dtype=np.float32)

    def action(self, vehicle, lane, target_state, dt):
        desired = self.desired_acceleration(self.lane_state(vehicle, lane), target_state)
        return self.action_adapter.to_metadrive(vehicle, desired, dt)


def trajectory_tracking_metrics(actual_states, reference_states):
    """Return RMSE in lane longitude/lateral position and forward speed."""
    actual=np.asarray(actual_states,dtype=float);reference=np.asarray(reference_states,dtype=float)
    if actual.shape!=reference.shape or actual.ndim!=2 or actual.shape[1]!=4 or not len(actual):
        raise ValueError("tracking states must be matching non-empty (T, 4) arrays")
    errors=actual-reference
    return {
        "longitudinal_rmse_m":float(np.sqrt(np.mean(errors[:,0]**2))),
        "lateral_rmse_m":float(np.sqrt(np.mean(errors[:,1]**2))),
        "speed_rmse_mps":float(np.sqrt(np.mean(errors[:,2]**2))),
    }
