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
        self.calibration = None

    def load_calibration(self, path):
        """Load measured v/control/acceleration surfaces for inverse control."""
        path = Path(path)
        if not path.is_file():
            return False
        with np.load(path, allow_pickle=False) as data:
            self.calibration = {key: data[key].astype(np.float64) for key in data.files}
        required = {"speeds", "controls", "longitudinal_accel", "lateral_accel"}
        if not required.issubset(self.calibration):
            missing=required-set(self.calibration)
            self.calibration = None
            raise ValueError(f"calibration file is missing {sorted(missing)}")
        return True

    @classmethod
    def from_yaml(cls, path=None):
        config_path = Path(path) if path else Path(__file__).resolve().parents[2] / "configs/srairl/metadrive_physics.yaml"
        # The file uses JSON syntax, which is also valid YAML, so loading this
        # small fixed schema does not add a PyYAML runtime dependency.
        data = json.loads(config_path.read_text(encoding="utf-8"))
        options=dict(data["action_adapter"])
        calibration=options.pop('calibration_file',None)
        adapter=cls(**options)
        if calibration:
            calibration_path=Path(calibration)
            if not calibration_path.is_absolute():calibration_path=config_path.parents[2]/calibration_path
            adapter.load_calibration(calibration_path)
        return adapter

    def to_metadrive(self, vehicle, action, dt):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all() or dt <= 0:
            raise ValueError("action must be finite [a_long, a_lat] and dt positive")
        speed = max(float(np.linalg.norm(vehicle.velocity)), 2.0)
        if self.calibration is not None:
            speeds=self.calibration['speeds']; controls=self.calibration['controls']
            si=int(np.clip(np.searchsorted(speeds,speed),1,len(speeds)-1))
            lo,hi=si-1,si; w=float(np.clip((speed-speeds[lo])/(speeds[hi]-speeds[lo]),0.,1.))
            table=self.calibration['longitudinal_accel']
            measured=(1-w)*table[lo]+w*table[hi]
            longitudinal=float(_inverse_surface(controls,measured,float(action[0])))
            table=self.calibration['lateral_accel']
            measured=(1-w)*table[lo]+w*table[hi]
            steering=float(_inverse_surface(controls,measured,float(action[1])))
        else:
            # Exploratory fallback retained for the interface baseline only.
            steering_angle = np.arctan2(self.wheelbase_m * float(action[1]), speed * speed)
            max_steering = max(float(getattr(vehicle, "max_steering", self.max_steering_angle_rad)), 0.1)
            steering = np.clip(steering_angle / max_steering, -1.0, 1.0)
            scale = self.throttle_accel_mps2 if action[0] >= 0 else self.brake_accel_mps2
            longitudinal = np.clip(float(action[0]) / scale, -1.0, 1.0)
        return np.asarray([steering, longitudinal], dtype=np.float32)


def _inverse_surface(controls, response, desired):
    """Invert a sampled response, choosing the least aggressive bracketing control."""
    controls=np.asarray(controls,dtype=float);response=np.asarray(response,dtype=float)
    candidates=[]
    for i in range(len(controls)-1):
        y0,y1=response[i],response[i+1]
        if min(y0,y1)-1e-12<=desired<=max(y0,y1)+1e-12 and abs(y1-y0)>1e-12:
            ratio=float(np.clip((desired-y0)/(y1-y0),0.,1.))
            candidates.append(float(controls[i]+ratio*(controls[i+1]-controls[i])))
    if candidates:return float(min(candidates,key=abs))
    return float(controls[int(np.argmin(np.abs(response-desired)))])
