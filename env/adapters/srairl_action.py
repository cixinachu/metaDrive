"""Candidate desired-acceleration to MetaDrive control adapter.

This belongs only to closed-loop transfer. Its calibration must match the
selected vehicle model; no default mapping is supplied or used by SRAIRLEnv.
"""
import numpy as np


class SRAIRLActionAdapter:
    def __init__(self, *, wheelbase_m, max_steering_angle_rad, max_longitudinal_accel):
        self.wheelbase_m = float(wheelbase_m)
        self.max_steering_angle_rad = float(max_steering_angle_rad)
        self.max_longitudinal_accel = float(max_longitudinal_accel)
        if min(self.wheelbase_m, self.max_steering_angle_rad, self.max_longitudinal_accel) <= 0:
            raise ValueError("supply positive, vehicle-calibrated control limits")

    def to_metadrive(self, vehicle, action, dt):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all() or dt <= 0:
            raise ValueError("action must be finite [a_long, a_lat] and dt positive")
        speed = max(float(np.linalg.norm(vehicle.velocity)), 1.0)
        longitudinal_accel = float(np.clip(action[0], -self.max_longitudinal_accel, self.max_longitudinal_accel))
        # Kinematic bicycle curvature relation; calibrate the steering-angle
        # limit against this MetaDrive vehicle before using transfer results.
        steering_angle = np.arctan2(self.wheelbase_m * float(action[1]), speed * speed)
        steering = np.clip(steering_angle / self.max_steering_angle_rad, -1.0, 1.0)
        longitudinal = np.clip(longitudinal_accel / self.max_longitudinal_accel, -1.0, 1.0)
        return np.asarray([steering, longitudinal], dtype=np.float32)
