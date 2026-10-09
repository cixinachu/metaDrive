"""Research configuration. Files use JSON syntax, a strict YAML subset."""
from copy import deepcopy
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def merge_config(base, overlay):
    result = deepcopy(base)
    for key, value in overlay.items():
        if key not in result:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(value, dict) and key != "vehicle_config":
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path=None, overrides=None):
    config = json.loads((ROOT / "configs/env/base.yaml").read_text())
    if path:
        config = merge_config(config, json.loads(Path(path).read_text()))
    if overrides:
        config = merge_config(config, overrides)
    validate_config(config)
    return config


def validate_config(c):
    e, s, scene = c["environment"], c["safety"], c["scenario"]
    if e["force_destroy"] is not True:
        raise ValueError("Research resets require force_destroy=true; pooled Bullet objects failed full-rollout reproducibility")
    if e["reward_mode"] not in ("cmdp", "metadrive"):
        raise ValueError("Unknown reward mode")
    if s["cost_mode"] not in ("binary", "continuous"):
        raise ValueError("Unknown cost mode")
    if s["vehicle_risk_aggregation"] != "max":
        raise ValueError("softmax is reserved, not implemented; use max")
    if c["traffic"]["randomize_behavior"]:
        raise ValueError("IDM behavior randomization is not implemented; random_traffic is not equivalent")
    if c["traffic"]["mode"] not in ("trigger", "respawn", "hybrid"):
        raise ValueError("Invalid traffic mode")
    for key in ("ttc_threshold", "min_distance", "time_headway", "comfortable_deceleration", "prediction_horizon", "encounter_margin", "road_margin", "road_sensor_range", "epsilon"):
        if not math.isfinite(s[key]) or s[key] <= 0:
            raise ValueError(f"safety.{key} must be positive and finite")
    for key in ("lane_width", "exit_length"):
        if not math.isfinite(e[key]) or e[key] <= 0:
            raise ValueError(f"environment.{key} must be positive and finite")
    for key in ("driving_reward", "speed_reward", "success_reward"):
        if not math.isfinite(c["reward"][key]) or c["reward"][key] < 0:
            raise ValueError(f"reward.{key} must be nonnegative and finite")
    if c["reward"]["use_lateral_reward"]:
        raise ValueError("The research reward excludes lateral shaping; use_lateral_reward must be false")
    for value in (e["horizon"], e["decision_repeat"], e["lane_num"], scene["num_scenarios"], s["road_sensor_rays"]):
        if type(value) is not int or value <= 0:
            raise ValueError("Counts must be positive integers")
    if not 0 <= e["traffic_density"] <= 1 or not 0 < e["physics_world_step_size"] < 1:
        raise ValueError("Invalid density or physics timestep")
    if not 0 < s["critical_threshold"] <= 1 or not -1 <= s["same_direction_cosine"] <= 1:
        raise ValueError("Invalid risk threshold")
    start, end = scene["start_seed"], scene["start_seed"] + scene["num_scenarios"]
    reserved = json.loads((ROOT / "configs/env/train.yaml").read_text())["scenario"]
    train_start = reserved["start_seed"]
    train_end = train_start + reserved["num_scenarios"]
    if type(start) is not int or start < 0:
        raise ValueError("Scene seeds must be nonnegative integers")
    if scene["split"] == "train":
        if start < train_start or end > train_end:
            raise ValueError(f"Training seeds must stay in [{train_start}, {train_end})")
    elif scene["split"] == "eval":
        if start < train_end and end > train_start:
            raise ValueError("Evaluation seeds overlap reserved training range")
    else:
        raise ValueError("split must be train or eval")
    for seed in c["seeds"].values():
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Seeds must be uint32 integers")


class SeedManager:
    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(config["seeds"]["environment_seed"])

    def scene(self, seed=None):
        s = self.config["scenario"]
        start, end = s["start_seed"], s["start_seed"] + s["num_scenarios"]
        chosen = int(self.rng.integers(start, end)) if seed is None else int(seed)
        if not start <= chosen < end:
            raise ValueError(f"Scene seed {chosen} outside configured split [{start}, {end})")
        return chosen

    def traffic(self, scene):
        return int(np.random.SeedSequence([self.config["seeds"]["traffic_seed"], scene]).generate_state(1)[0] % (2**31 - 1))
