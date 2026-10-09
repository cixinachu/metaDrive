"""Single-agent MetaDrive CMDP adapter; no changes to SAC or observations."""
from copy import deepcopy
import math
import numpy as np
from metadrive import MetaDriveEnv
from metadrive.manager.traffic_manager import PGTrafficManager
from .research_config import load_config, validate_config, SeedManager
from .risk_utils import VehicleState
from .safety_cost import compute_safety_cost
from .episode_statistics import EpisodeStatistics


class ResearchTrafficManager(PGTrafficManager):
    def __init__(self, seed_manager):
        self.seed_manager = seed_manager
        super().__init__()

    def seed(self, random_seed):
        super().seed(self.seed_manager.traffic(int(random_seed)))

    def reset(self):
        # Also covers the first engine reset (manager may be created after engine.seed).
        self.seed(self.engine.global_random_seed)
        return super().reset()


def termination_flags(info, termination, at_horizon):
    terminated = bool(info.get("arrive_dest", False)
                      or (info.get("crash", False) and termination["terminate_on_collision"])
                      or (info.get("out_of_road", False) and termination["terminate_on_out_of_road"]))
    return terminated, bool(at_horizon)


class SafeMetaDriveEnv(MetaDriveEnv):
    def __init__(self, config_path=None, research_config=None):
        self.research_config = deepcopy(research_config) if research_config is not None else load_config(config_path)
        validate_config(self.research_config)
        self.dataset_clip = None
        if self.research_config.get('dataset'):
            from datasets.importer import load_clip
            self.dataset_clip, manifest = load_clip(self.research_config['dataset']['id'])
            if manifest['clip_sha256'] != self.research_config['dataset']['clip_sha256']:
                raise ValueError('Dataset fingerprint mismatch')
        self.seed_manager = SeedManager(self.research_config)
        c = self.research_config
        e, scene, traffic = c["environment"], c["scenario"], c["traffic"]
        config = {k: e[k] for k in ("map", "horizon", "traffic_density", "physics_world_step_size", "decision_repeat", "force_destroy")}
        config["map_config"] = {k: e[k] for k in ("lane_width", "lane_num", "exit_length")}
        config.update(c["reward"])
        config.update(start_seed=scene["start_seed"], num_scenarios=scene["num_scenarios"],
                      use_render=False, image_observation=False, is_multi_agent=False,
                      random_traffic=False, traffic_mode=traffic["mode"], traffic_vehicle_config=traffic["vehicle_config"],
                      truncate_as_terminate=False, crash_vehicle_done=False, crash_object_done=False,
                      crash_human_done=False, out_of_road_done=c["termination"]["terminate_on_out_of_road"],
                      log_level=50)
        if e["reward_mode"] == "cmdp":
            config.update(crash_vehicle_penalty=0.0, crash_object_penalty=0.0, crash_sidewalk_penalty=0.0, out_of_road_penalty=0.0)
        if self.dataset_clip:
            config['map_region_size'] = 2048
            config['random_spawn_lane_index'] = False
            from .recorded_traffic import sample
            track = next(t for t in self.dataset_clip['tracks'] if t['ego'])
            x,y,vx,vy = sample(track, 0.)
            config['vehicle_config'] = dict(vehicle_model='varying_dynamics',length=track['length'],width=track['width'],spawn_position_heading=((float(x),float(y)),math.atan2(vy,vx)),
                spawn_velocity=(float(vx),float(vy)), spawn_lane_index=None)
            config['agent_configs'] = {'default_agent': dict(use_special_color=True,spawn_lane_index=None)}
            self.dataset_start_x = float(x)
            self.dataset_goal_x = float(sample(track, self.dataset_clip['duration'])[0])
        self.dt = e["physics_world_step_size"] * e["decision_repeat"]
        self._research_steps = 0
        self._needs_reset = True
        super().__init__(config)

    def _post_process_config(self, config):
        # MetaDrive's easy-map parser skips map generation settings when lane
        # dimensions differ from defaults. Parse topology first, then apply dimensions.
        dimensions = {k: config["map_config"][k] for k in ("lane_num", "lane_width", "exit_length")}
        for k in dimensions:
            config["map_config"][k] = self.default_config_copy["map_config"][k]
        config = super()._post_process_config(config)
        config["map_config"].update(dimensions)
        return config

    def setup_engine(self):
        super().setup_engine()
        if self.dataset_clip:
            from .recorded_traffic import RecordedTrafficManager
            from .recorded_map import RecordedMapManager
            self.engine.update_manager("map_manager", RecordedMapManager())
            self.engine.update_manager("traffic_manager", RecordedTrafficManager(self.dataset_clip))
        else:
            self.engine.update_manager("traffic_manager", ResearchTrafficManager(self.seed_manager))

    def _is_arrive_destination(self, vehicle):
        if self.dataset_clip:
            return bool(self.dataset_goal_x-self.dataset_start_x>5 and vehicle.position[0]>=self.dataset_goal_x and not self._is_out_of_road(vehicle))
        return super()._is_arrive_destination(vehicle)

    def reward_function(self, vehicle_id):
        reward, info = super().reward_function(vehicle_id)
        if self.research_config["environment"]["reward_mode"] == "cmdp":
            reward = self.config["success_reward"] if self._is_arrive_destination(self.agents[vehicle_id]) else info["step_reward"]
        return reward, info

    def done_function(self, vehicle_id):
        _, info = super().done_function(vehicle_id)
        done, _ = termination_flags(info, self.research_config["termination"], False)
        return done, info

    def reset(self, seed=None, options=None):
        scene_seed = self.seed_manager.scene(seed)
        obs, info = super().reset(seed=scene_seed)
        self._research_steps = 0
        self._needs_reset = False
        self.statistics = EpisodeStatistics(self.dt, self.research_config["safety"]["critical_threshold"])
        info.update(self.seed_metadata())
        return obs, info

    def seed_metadata(self):
        return {**self.research_config["seeds"], "scene_seed": int(self.current_seed),
                "traffic_randomness_seed": self.seed_manager.traffic(int(self.current_seed)), "dt": self.dt}

    @staticmethod
    def vehicle_state(vehicle):
        return VehicleState(str(vehicle.id), tuple(vehicle.position), tuple(vehicle.velocity),
                            float(vehicle.heading_theta), float(vehicle.LENGTH), float(vehicle.WIDTH))

    def road_clearance(self):
        """Minimum sampled boundary-point distance to ego rectangle (not route width).

        SideDetector rays hit continuous lane lines and sidewalks in the static world.
        Finite angular resolution can overestimate actual clearance; no hit is None.
        """
        c, v = self.research_config["safety"], self.agent
        sensor = self.engine.get_sensor("side_detector")
        points = np.asarray(sensor.perceive(v, self.engine.physics_world.static_world,
                            c["road_sensor_rays"], c["road_sensor_range"]).cloud_points)
        if not np.isfinite(points).all():
            raise ValueError("Non-finite boundary sensor")
        hit = points < 1.0
        if not hit.any():
            return None
        theta = sensor._get_lidar_range(c["road_sensor_rays"], sensor.start_phase_offset)[hit]
        distance = points[hit] * c["road_sensor_range"]
        x = np.maximum(np.abs(distance * np.cos(theta)) - v.LENGTH / 2, 0)
        y = np.maximum(np.abs(distance * np.sin(theta)) - v.WIDTH / 2, 0)
        return float(np.min(np.hypot(x, y)))

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Call reset before stepping or after an episode ends")
        obs, reward, _, _, info = super().step(action)
        self._research_steps += 1
        ego = self.vehicle_state(self.agent)
        others = [self.vehicle_state(v) for v in self.engine.traffic_manager.traffic_vehicles if v.id != self.agent.id]
        if self.dataset_clip:
            distance = self.dataset_goal_x-self.dataset_start_x
            info['route_completion'] = float(np.clip((self.agent.position[0]-self.dataset_start_x)/max(distance,1e-6),0,1))
            info['dataset_id'] = self.research_config['dataset']['id']
            info['traffic_replay'] = True
        info["metadrive_cost"] = info.get("cost")
        info.update(compute_safety_cost(ego, others, self.road_clearance(), info.get("crash", False),
                                       info.get("out_of_road", False), self.research_config["safety"]))
        info.update(self.seed_metadata())
        self.statistics.update(info, float(reward), float(np.linalg.norm(ego.velocity)))
        terminated, truncated = termination_flags(info, self.research_config["termination"],
                                                  self._research_steps >= self.research_config["environment"]["horizon"])
        if terminated or truncated:
            info["episode_safety"] = {**self.statistics.result(info), **self.seed_metadata(),
                                      "terminated": terminated, "truncated": truncated}
            self._needs_reset = True
        return obs, float(reward), terminated, truncated, info
