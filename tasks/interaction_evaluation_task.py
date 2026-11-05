"""Construct and run the integrated SR-AIRL interaction evaluation pipeline."""
import json
import importlib.metadata
from pathlib import Path

import numpy as np

from env.tasks.srairl_trajectory_tracker import SRAIRLTrajectoryTracker, TrackerGains
from rl.srairl.pipeline.coordinates import upstream_to_lane_state
from rl.srairl.pipeline.evaluator import write_evaluation_record
from rl.srairl.pipeline.executor import SRAIRLClosedLoopPipeline
from rl.srairl.pipeline.model_bundle import SRAIRLModelBundle, sha256
from rl.srairl.pipeline.planner_inference import PlannerInference
from rl.srairl.pipeline.predictor_inference import RewardGuidedResponsePredictor
from rl.srairl.pipeline.role_forecasting import forecast_rlv_roles
from rl.srairl.pipeline.schema import PlanningContext
from rl.srairl.pipeline.selector import CandidateSelector
from rl.srairl.upstream.adapter import load_reward_discriminator


class InteractionEvaluationTask:
    """Run complete receding-horizon evaluation with an externally built env.

    The task refuses to start if checkpoints are missing/incompatible or if
    MetaDrive action calibration is absent. Studio may call this class after
    constructing the selected scenario environment.
    """
    def __init__(self, *, model_bundle, calibration_summary, device="cpu",
                 candidate_count=16, replan_steps=5):
        self.model_bundle = model_bundle
        self.calibration_summary = Path(calibration_summary).resolve()
        if not self.calibration_summary.is_file():
            raise FileNotFoundError(f"action calibration summary missing: {self.calibration_summary}")
        self.calibration_hash = sha256(self.calibration_summary)
        data = json.loads(self.calibration_summary.read_text(encoding="utf-8"))
        if data.get("equivalence_status") != "passed" or not data.get("physics_evaluation_allowed", True):
            raise ValueError("closed-loop evaluation requires passed Action Adapter calibration")
        limits = data.get("validated_acceleration_domain", {})
        ax = limits.get("ax_mps2", [-1.5, 1.5])
        ay = limits.get("ay_mps2", [-1.5, 1.5])
        self.selector_domain = (ax, ay)
        self.device = device
        self.candidate_count = candidate_count
        self.replan_steps = replan_steps

    def evaluate(self, env, *, seed=0, output_path=None, scenario_id=None, episode_index=1):
        planner = PlannerInference(self.model_bundle, device=self.device)
        predictor = RewardGuidedResponsePredictor(self.model_bundle, device=self.device)
        reward_model_dir = self.model_bundle.reward_model_path if self.model_bundle.reward_model_path.is_dir() \
            else self.model_bundle.reward_model_path.parent
        elcv_disc = load_reward_discriminator(reward_model_dir, device=self.device, role="lcv")
        rfv_disc = load_reward_discriminator(reward_model_dir, device=self.device, role="fv")
        cfg = json.loads((reward_model_dir / "config.json").read_text(encoding="utf-8"))
        gamma = float(cfg["srairl"]["gamma"])
        selector = CandidateSelector(elcv_disc, rfv_disc, gamma=gamma,
                                     validated_ax=self.selector_domain[0],
                                     validated_ay=self.selector_domain[1])
        tracker = SRAIRLTrajectoryTracker(env.action_adapter, TrackerGains(
            max_longitudinal_accel=float(self.selector_domain[0][1]),
            min_longitudinal_accel=float(self.selector_domain[0][0]),
            max_lateral_accel=float(max(abs(v) for v in self.selector_domain[1]))))

        def lane_check(role, states):
            source = getattr(env, "_source_lane", None)
            target = getattr(env, "_target_lane", source)
            if source is None:
                return False
            if getattr(env, "scenario_type", None) == "merge":
                paths = getattr(env.world, "_interaction_paths", {})
                allowed_list = []
                for role_path in ("rlv_current", "rlv_target"):
                    for lane in paths.get(role_path, ((), 0., 0.))[0]:
                        if not any(lane is existing for existing in allowed_list):
                            allowed_list.append(lane)
                allowed = tuple(allowed_list) or (source, target)
            else:
                allowed = (source, target)
            for y, x, vx, vy in np.asarray(states):
                if getattr(env, "scenario_type", None) == "merge":
                    point = np.asarray([x, y], dtype=float)
                else:
                    point = np.asarray(source.position(float(x), float(y)), dtype=float)[:2]
                if not any(abs(float(lane.local_coordinates(point)[1])) <= float(lane.width_at(0)) * 0.5
                           for lane in allowed):
                    return False
            if role == "elcv":
                y, x = np.asarray(states[-1])[:2]
                point = np.asarray([x, y], dtype=float) if getattr(env, "scenario_type", None) == "merge" \
                    else np.asarray(source.position(float(x), float(y)), dtype=float)[:2]
                if abs(float(target.local_coordinates(point)[1])) > float(target.width_at(0)) * 0.5:
                    return False
            return True

        def map_to_tracker(role, state, current_env):
            if getattr(current_env, "scenario_type", None) == "merge":
                vehicle = current_env.world.role_vehicle(role)
                return upstream_to_lane_state(state, vehicle.lane)
            # Straight/curve role states share the source lane's local frame.
            y, x, vx, vy = np.asarray(state, dtype=float)
            return np.asarray([x, -y, vx, vy])

        def planning_context(current_env, history):
            ego = current_env.world.role_vehicle("elcv")
            if getattr(current_env, "scenario_type", None) == "merge":
                paths = getattr(current_env.world, "_interaction_paths", {})
                target_path = paths.get("rlv_target", (None, 0., 0.))[0]
                if not target_path:
                    raise RuntimeError("merge target-lane geometry is unavailable")
                speed = max(1.0, float(np.linalg.norm(ego.velocity[:2])))
                desired_x = float(ego.position[0]) + speed * 10.0
                points = []
                for lane in target_path:
                    samples = np.linspace(0.0, float(lane.length), max(2, int(lane.length / 2.0) + 1))
                    points.extend(np.asarray(lane.position(float(s), 0.0), dtype=float)[:2]
                                  for s in samples)
                target_point = min(points, key=lambda point: abs(float(point[0]) - desired_x))
                target_y = float(target_point[1])
            else:
                source = current_env._source_lane
                target = current_env._target_lane
                source_s, _ = source.local_coordinates(ego.position[:2])
                target_center = target.position(float(source_s), 0.0)
                _, target_y = source.local_coordinates(np.asarray(target_center, dtype=float)[:2])
            return PlanningContext(target_lateral_position=float(target_y), target_lateral_velocity=0.0)

        pipeline = SRAIRLClosedLoopPipeline(
            planner=planner, predictor=predictor, selector=selector, tracker=tracker,
            role_forecaster=forecast_rlv_roles, candidate_count=self.candidate_count,
            replan_steps=self.replan_steps, model_bundle=self.model_bundle,
            calibration_hash=self.calibration_hash, lane_state_mapper=map_to_tracker,
            planning_context_provider=planning_context,
        )
        pipeline.selector.lane_check = lane_check
        video = None
        old_callback = getattr(env, "frame_callback", None)
        if output_path:
            from studio.worker import EpisodeVideo, render_srairl_frame
            video = EpisodeVideo(Path(output_path).resolve().parent, episode=episode_index, fps=10)
            def capture(active, info):
                if old_callback:
                    old_callback(active, info)
                frame = render_srairl_frame(Path(output_path).resolve().parent, active, info)
                if frame is not None:
                    video.write(frame)
            env.frame_callback = capture
        try:
            result = pipeline.evaluate_episode(env, seed=seed, scenario_id=scenario_id)
        finally:
            if video is not None:
                video.close()
            if hasattr(env, "frame_callback"):
                env.frame_callback = old_callback
        result["calibration_hash"] = self.calibration_hash
        result["evaluation_protocol"] = "closed_loop_metadrive_receding_horizon_future_only_v2"
        if output_path:
            try:
                metadrive_version = importlib.metadata.version("metadrive-simulator")
            except importlib.metadata.PackageNotFoundError:
                metadrive_version = "unknown"
            write_evaluation_record(output_path, result["metrics"],
                                    bundle_manifest=self.model_bundle.to_manifest(),
                                    environment_metadata={"scenario_id": scenario_id,
                                                          "seed": seed,
                                                          "protocol": result["evaluation_protocol"],
                                                          "scenario_type": getattr(env, "scenario_type", None),
                                                          "scene_source": getattr(env, "scene_source", "procedural"),
                                                          "initialization_source": "metadrive_procedural" if getattr(env, "current_initial_state", None) else "canonical",
                                                          "initial_state_dataset_id": None if not getattr(env, "current_initial_state", None) else env.current_initial_state.get("dataset_id"),
                                                          "warmup_steps": result["warmup_steps"],
                                                          "warmup_duration_s": result["warmup_duration_s"],
                                                          "evaluation_start_states": result["evaluation_start_states"],
                                                          "candidate_time_semantics": "future_only_t_plus_dt_through_t_plus_horizon_dt",
                                                          "metadrive_version": metadrive_version,
                                                          "calibration_hash": self.calibration_hash})
        return result


__all__ = ["InteractionEvaluationTask", "SRAIRLModelBundle"]
