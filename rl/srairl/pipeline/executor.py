"""Receding-horizon closed-loop planner, selector, dual-tracker executor."""
import time
import inspect

import numpy as np

from env.tasks.srairl_trajectory_tracker import trajectory_tracking_metrics
from .candidate_builder import build_joint_candidates
from .evaluator import EpisodeMetrics
from .schema import CandidateBundle, PlanningContext


class RoleHistoryBuffer:
    def __init__(self, history_len=20):
        if history_len < 1:
            raise ValueError("history_len must be positive")
        self.history_len = int(history_len)
        self._values = {role: [] for role in ("elcv", "rfv", "rlv_current", "rlv_target")}

    def reset(self, roles):
        self._values = {role: [] for role in self._values}
        # Match the upstream discriminator's initial-frame replication while
        # keeping the first Planner decision valid at episode start.
        for _ in range(self.history_len):
            self.append(roles)

    def append(self, roles):
        if set(roles) != set(self._values):
            raise ValueError("role state mapping must contain all four explicit SR-AIRL roles")
        for role, state in roles.items():
            array = np.asarray(state, dtype=np.float64)
            if array.shape != (4,) or not np.isfinite(array).all():
                raise ValueError(f"{role} state must be finite [y,x,vx,vy]")
            self._values[role].append(array.copy())
            self._values[role] = self._values[role][-self.history_len:]

    def current(self):
        if any(len(values) != self.history_len for values in self._values.values()):
            raise RuntimeError(f"need {self.history_len} observed frames before planning")
        return {role: np.stack(values) for role, values in self._values.items()}


class SRAIRLClosedLoopPipeline:
    """Uses injected role forecasting and coordinate mapping to avoid guessing.

    ``role_forecaster(environment, history, horizon, dt)`` returns explicit
    RLV-current and RLV-target future TrajectoryCandidates. For recorded NGSIM
    these must be the recorded futures; for procedural scenes it must describe
    the actual scripted/physical motion source.
    """
    def __init__(self, *, planner, predictor, selector, tracker,
                 role_forecaster, candidate_count=16, replan_steps=5,
                 model_bundle=None, calibration_hash="", lane_state_mapper=None,
                 planning_context_provider=None):
        if candidate_count < 1 or replan_steps < 1:
            raise ValueError("candidate_count and replan_steps must be positive")
        self.planner = planner
        self.predictor = predictor
        self.selector = selector
        self.tracker = tracker
        self.role_forecaster = role_forecaster
        self.candidate_count = int(candidate_count)
        self.replan_steps = int(replan_steps)
        self.model_bundle = model_bundle
        self.calibration_hash = str(calibration_hash)
        self.lane_state_mapper = lane_state_mapper
        self.planning_context_provider = planning_context_provider

    @staticmethod
    def _environment_roles(env):
        if hasattr(env, "_roles_live"):
            return env._roles_live()
        if hasattr(env, "_roles"):
            return env._roles(float(env.time))
        raise TypeError("closed-loop SR-AIRL requires an environment exposing four live role states")

    @staticmethod
    def _initial_history(env, current_roles):
        if hasattr(env, "scenario_type") or not hasattr(getattr(env, "scenario", None), "tracks"):
            return [current_roles] * 20
        rows = []
        role_ids = {"elcv": env.scenario.elcv_id, "rfv": env.scenario.rfv_id,
                    "rlv_current": env.scenario.rlv_current_id,
                    "rlv_target": env.scenario.rlv_target_id}
        for index in range(19, -1, -1):
            t = float(env.time) - index * 0.1
            roles = {}
            for role, ident in role_ids.items():
                raw = env._sample(env.scenario.tracks[ident], t)
                if raw is None:
                    raise ValueError(f"NGSIM {role} track lacks the required 20-frame history at t={t:.2f}")
                x, y, vx, vy = raw
                roles[role] = np.asarray([y, x, vx, vy], dtype=np.float64)
            rows.append(roles)
        return rows

    def _track_state(self, role, physical_state, env):
        if self.lane_state_mapper is not None:
            return np.asarray(self.lane_state_mapper(role, physical_state, env), dtype=float)
        raise RuntimeError("a MetaDrive lane-state mapper is required for tracker execution")

    def _provenance(self):
        if self.model_bundle is None:
            return {}
        return self.model_bundle.to_manifest()

    def evaluate_episode(self, env, *, seed=0, scenario_id=None):
        if hasattr(self.planner, "reset_episode"):
            self.planner.reset_episode()
        reset_parameters = inspect.signature(env.reset).parameters
        reset_result = env.reset(seed=seed) if "seed" in reset_parameters else env.reset()
        roles = self._environment_roles(env)
        history = RoleHistoryBuffer(20)
        initial_rows = self._initial_history(env, roles)
        history.reset(initial_rows[0])
        for row in initial_rows[1:]:
            history.append(row)
        metrics = EpisodeMetrics()
        decisions = 0
        done = False
        last_info = {}
        while not done:
            state_history = history.current()
            if self.planning_context_provider is None:
                raise RuntimeError("a task target-lane planning context provider is required")
            planning_context = self.planning_context_provider(env, state_history)
            if not isinstance(planning_context, PlanningContext):
                raise TypeError("planning_context_provider must return PlanningContext")
            decision_started = time.perf_counter()
            started = time.perf_counter()
            proposals = self.planner.generate_candidates(
                state_history, planning_context, num_candidates=self.candidate_count,
                seed=int(seed) + decisions,
            )
            metrics.planning_latency_s.append(time.perf_counter() - started)
            started = time.perf_counter()
            rlv_current, rlv_target = self.role_forecaster(env, state_history, 100, 0.1)
            joint = build_joint_candidates(
                proposals, self.predictor, state_history["rfv"],
                rlv_current, rlv_target, dt=0.1,
            )
            metrics.predictor_latency_s.append(time.perf_counter() - started)
            selected, returns = self.selector.select(joint, history=state_history)
            metrics.candidate_returns.append({"planning_step": decisions,
                                              "candidate_returns": returns,
                                              "selected_candidate_id": selected.candidate_id})
            metrics.selected_candidate_ids.append(selected.candidate_id)
            metrics.selected_srairl_returns.append(selected.joint_return)
            metrics.candidate_count += len(joint)

            if self.model_bundle is not None:
                bundle = CandidateBundle(
                    schema_version="srairl-candidate-bundle-v1",
                    scenario_id=scenario_id or getattr(env.scenario, "scenario_id", "unknown"),
                    planning_step=decisions, dt=0.1, horizon=100,
                    history=state_history, candidates=joint,
                    planner_checkpoint=str(self.model_bundle.planner_diffusion_path),
                    reward_guide_checkpoint=str(self.model_bundle.reward_guide_path),
                    predictor_checkpoint=str(self.model_bundle.predictor_path),
                    srairl_checkpoint=str(self.model_bundle.reward_model_path),
                    calibration_hash=self.calibration_hash,
                    normalization_version="planner-goal-v1+reward-predictor-v1",
                    selected_candidate_id=selected.candidate_id,
                )
                env.last_candidate_bundle = bundle

            steps = min(self.replan_steps, getattr(env, "max_steps", self.replan_steps))
            for offset in range(steps):
                reference_index = min(offset, 99)
                controls = {}
                for role, trajectory in (("elcv", selected.elcv), ("rfv", selected.rfv)):
                    vehicle = env.world.role_vehicle(role)
                    lane = vehicle.lane if getattr(env, "scenario_type", None) == "merge" \
                        else getattr(env, "_source_lane", vehicle.lane)
                    current_lane = self.tracker.lane_state(vehicle, lane)
                    target_lane = self._track_state(role, trajectory.states[reference_index], env)
                    controls[role] = self.tracker.desired_acceleration(current_lane, target_lane)
                    tracked = trajectory_tracking_metrics(current_lane[None, :], target_lane[None, :])
                    for key, value in tracked.items():
                        metrics.tracker_errors.setdefault(f"{role}/{key}", []).append(value)
                _, last_info = env.step(controls)
                history.append(self._environment_roles(env))
                metrics.update_environment(last_info)
                done = bool(last_info.get("done", last_info.get("terminated", False)) or
                            last_info.get("truncated", False))
                if done:
                    break
            metrics.total_decision_latency_s.append(time.perf_counter() - decision_started)
            metrics.planning_steps += 1
            decisions += 1
        metrics.update_environment(last_info)
        return {"metrics": metrics, "last_info": last_info,
                "model_bundle": self._provenance(),
                "reset_info": reset_result[1] if isinstance(reset_result, tuple) and len(reset_result) > 1 else {}}
