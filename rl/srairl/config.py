"""Typed SR-AIRL configuration backed by the study's reward_config.py."""
from dataclasses import dataclass
from . import reward_config as defaults


@dataclass(frozen=True)
class SRAIRLConfig:
    frame_state_dim: int = defaults.FRAME_STATE_DIM
    history_len: int = defaults.HISTORY_LEN
    gamma: float = defaults.GAMMA
    potential_longitudinal_scale: float = defaults.L_LONG
    potential_lateral_scale: float = defaults.L_LAT
    direction_lambda: float = defaults.DIR_LAMBDA
    alpha_init: float = defaults.ALPHA_INIT
    pos_lat_scale: float = defaults.POS_LAT_SCALE
    pos_long_scale: float = defaults.POS_LONG_SCALE
    vel_scale: float = defaults.VEL_SCALE
    vel_offset: float = defaults.VEL_OFFSET
    dt_s: float = defaults.DT
    acceleration_scale: float = defaults.ACC_SCALE
    jerk_scale: float = defaults.JERK_SCALE
    interaction_prediction_horizon_s: float = defaults.PRED_HORIZON
    interaction_prediction_steps: int = defaults.PRED_STEPS

    @classmethod
    def from_mapping(cls, values=None):
        values = values or {}
        aliases = {
            "FRAME_STATE_DIM": "frame_state_dim", "HISTORY_LEN": "history_len", "GAMMA": "gamma",
            "L_LONG": "potential_longitudinal_scale", "L_LAT": "potential_lateral_scale",
            "DIR_LAMBDA": "direction_lambda", "ALPHA_INIT": "alpha_init",
            "POS_LAT_SCALE": "pos_lat_scale", "POS_LONG_SCALE": "pos_long_scale",
            "VEL_SCALE": "vel_scale", "VEL_OFFSET": "vel_offset",
            "DT": "dt_s", "ACC_SCALE": "acceleration_scale", "JERK_SCALE": "jerk_scale",
            "PRED_HORIZON": "interaction_prediction_horizon_s",
            "PRED_STEPS": "interaction_prediction_steps",
        }
        normalized = {aliases.get(k, k): v for k, v in values.items()}
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(normalized) - allowed)
        if unknown:
            raise ValueError(f"unknown SR-AIRL configuration keys: {unknown}")
        config = cls(**normalized)
        if config.frame_state_dim != 16:
            raise ValueError("the upstream SR-AIRL state frame is 16D")
        if config.history_len < 1 or not 0 < config.gamma <= 1:
            raise ValueError("invalid history_len or gamma")
        if min(config.potential_longitudinal_scale, config.potential_lateral_scale,
               config.alpha_init, config.pos_lat_scale, config.pos_long_scale, config.vel_scale,
               config.dt_s, config.acceleration_scale, config.jerk_scale,
               config.interaction_prediction_horizon_s, config.interaction_prediction_steps) <= 0:
            raise ValueError("scales and alpha_init must be positive")
        if isinstance(config.interaction_prediction_steps, bool) or int(config.interaction_prediction_steps) != config.interaction_prediction_steps:
            raise ValueError("interaction_prediction_steps must be a positive integer")
        if not 0 <= config.direction_lambda <= 1:
            raise ValueError("direction_lambda must be in [0, 1]")
        return config
