"""Observation adapter matching upstream SR-AIRL's normalized 16D frame."""
from collections import deque
import numpy as np


class SRAIRLObservationAdapter:
    def __init__(self, history_len=4, *, pos_lat_scale=25.0, pos_long_scale=500.0,
                 vel_scale=80.0, vel_offset=-40.0):
        if isinstance(history_len, bool) or int(history_len) < 1:
            raise ValueError("history_len must be a positive integer")
        if min(pos_lat_scale, pos_long_scale, vel_scale) <= 0:
            raise ValueError("normalization scales must be positive")
        self.history_len = int(history_len)
        self.pos_lat_scale = float(pos_lat_scale)
        self.pos_long_scale = float(pos_long_scale)
        self.vel_scale = float(vel_scale)
        self.vel_offset = float(vel_offset)
        self.history = deque(maxlen=self.history_len)

    def normalize_vehicle(self, state):
        # Upstream order after normalize_obs and dropping acceleration:
        # lateral position, longitudinal position, longitudinal speed, lateral speed.
        v = np.asarray(state, dtype=np.float64)
        if v.shape != (4,) or not np.isfinite(v).all():
            raise ValueError("vehicle state must be finite [lat, lon, v_long, v_lat]")
        lows = np.asarray([0., 0., self.vel_offset, self.vel_offset])
        spans = np.asarray([self.pos_lat_scale, self.pos_long_scale, self.vel_scale, self.vel_scale])
        return np.clip((v - lows) / spans, -1., 1.)

    def reset(self, roles):
        frame = self.frame(roles)
        self.history.clear()
        for _ in range(self.history_len):
            self.history.append(frame.copy())
        return self.observe()

    def append(self, roles):
        self.history.append(self.frame(roles))
        return self.observe()

    def observe(self):
        return np.concatenate(tuple(self.history)).astype(np.float32)

    def frame(self, roles):
        required = ("elcv", "rfv", "rlv_current", "rlv_target")
        missing = [k for k in required if k not in roles or roles[k] is None]
        if missing:
            raise ValueError(f"SR-AIRL observation requires explicit roles: {missing}")
        elcv = self.normalize_vehicle(roles["elcv"])
        rfv = self.normalize_vehicle(roles["rfv"])
        rlv_current = self.normalize_vehicle(roles["rlv_current"])
        rlv_target = self.normalize_vehicle(roles["rlv_target"])
        return np.concatenate((elcv, elcv - rfv, rlv_current - elcv, rlv_target - elcv))
