"""Candidate desired-acceleration to MetaDrive control adapter.

This belongs only to closed-loop transfer. Its calibration must match the
selected vehicle model; no default mapping is supplied or used by SRAIRLEnv.
"""
import numpy as np
from pathlib import Path
import json


class SRAIRLActionAdapter:
    def __init__(self, *, wheelbase_m=2.7, max_steering_angle_rad=0.7,
                 throttle_accel_mps2=2.35, brake_accel_mps2=12.5):
        self.wheelbase_m = float(wheelbase_m)
        self.max_steering_angle_rad = float(max_steering_angle_rad)
        self.throttle_accel_mps2 = float(throttle_accel_mps2)
        self.brake_accel_mps2 = float(brake_accel_mps2)
        if min(self.wheelbase_m, self.max_steering_angle_rad,
               self.throttle_accel_mps2, self.brake_accel_mps2) <= 0:
            raise ValueError("vehicle control scales must be positive")

    @classmethod
    def from_yaml(cls, path=None):
        config_path = Path(path) if path else Path(__file__).resolve().parents[2] / "configs/srairl/metadrive_physics.yaml"
        # The file uses JSON syntax, which is also valid YAML, so loading this
        # small fixed schema does not add a PyYAML runtime dependency.
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(**data["action_adapter"])

    def to_metadrive(self, vehicle, action, dt):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all() or dt <= 0:
            raise ValueError("action must be finite [a_long, a_lat] and dt positive")
        speed = max(float(np.linalg.norm(vehicle.velocity)), 2.0)
        # Kinematic bicycle curvature relation. This transfer mapping is
        # empirical for the configured MetaDrive vehicle and needs validation
        # over the speed range used by an experiment.
        steering_angle = np.arctan2(self.wheelbase_m * float(action[1]), speed * speed)
        max_steering = max(float(getattr(vehicle, "max_steering", self.max_steering_angle_rad)), 0.1)
        steering = np.clip(steering_angle / max_steering, -1.0, 1.0)
        scale = self.throttle_accel_mps2 if action[0] >= 0 else self.brake_accel_mps2
        longitudinal = np.clip(float(action[0]) / scale, -1.0, 1.0)
        return np.asarray([steering, longitudinal], dtype=np.float32)
