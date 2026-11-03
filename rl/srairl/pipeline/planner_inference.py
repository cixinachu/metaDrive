"""Inference wrappers matching the local Step2 Planner training contract."""
from pathlib import Path

import numpy as np
import torch

from .normalization import PlannerGoalNormalizer
from .schema import PlanningContext, TrajectoryCandidate


class PlannerInference:
    def __init__(self, model_bundle, device="cpu"):
        self.bundle = model_bundle
        self.device = torch.device(device)
        root = Path(__file__).resolve().parents[1] / "upstream" / "Step2_Planner"
        import sys
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from rl.srairl.upstream.Step2_Planner.train_goal import GoalPredictor
        from rl.srairl.upstream.Step2_Planner.model.temporal import TemporalUnet
        from rl.srairl.upstream.Step2_Planner.model.diffusion import GaussianDiffusion

        goal_data = torch.load(self.bundle.planner_goal_path, map_location=self.device, weights_only=True)
        if not isinstance(goal_data, dict) or "state_dict" not in goal_data:
            raise ValueError("Goal Predictor checkpoint is legacy full-module format; retrain with the current checkpoint contract")
        architecture = goal_data.get("architecture", {})
        self.goal = GoalPredictor(architecture.get("context_dim", 128)).to(self.device)
        self.goal.load_state_dict(goal_data["state_dict"], strict=True)
        self.goal.eval()
        self.goal_min = torch.as_tensor(goal_data["bound_min"], dtype=torch.float32, device=self.device)
        self.goal_max = torch.as_tensor(goal_data["bound_max"], dtype=torch.float32, device=self.device)

        model = TemporalUnet(horizon=100, transition_dim=6, dim=64, cond_dim=4,
                             dim_mults=(1, 2, 4, 8), attention=False).to(self.device)
        self.diffusion = GaussianDiffusion(
            model, horizon=100, observation_dim=4, action_dim=2, n_timesteps=20,
            loss_type="l2", clip_denoised=False, predict_epsilon=False,
            action_weight=10, loss_discount=1,
        ).to(self.device)
        checkpoint = torch.load(self.bundle.planner_diffusion_path, map_location=self.device, weights_only=True)
        state = checkpoint.get("ema", checkpoint.get("model", checkpoint))
        self.diffusion.load_state_dict(state, strict=True)
        self.diffusion.eval()
        self._origin = None

    def reset_episode(self):
        self._origin = None

    @staticmethod
    def _validate_history(history):
        required = {"elcv", "rfv", "rlv_current", "rlv_target"}
        if set(history) != required:
            raise ValueError(f"history roles must be {sorted(required)}")
        checked = {}
        for role, values in history.items():
            value = np.asarray(values, dtype=np.float32)
            if value.shape != (20, 4) or not np.isfinite(value).all():
                raise ValueError(f"{role} history must be finite [20,4] [y,x,vx,vy]")
            checked[role] = value
        return checked

    @torch.no_grad()
    def predict_goal(self, history):
        history = self._validate_history(history)
        if self._origin is None:
            self._origin = history["elcv"][0, :2].astype(np.float32).copy()
        centered = {role: values.copy() for role, values in history.items()}
        for values in centered.values():
            values[:, :2] -= self._origin
        tensors = [torch.as_tensor(PlannerGoalNormalizer.transform(centered[k]),
                                   device=self.device).unsqueeze(0)
                   for k in ("elcv", "rfv", "rlv_current", "rlv_target")]
        normalized_delta = self.goal(*tensors)[0]
        return normalized_delta * (self.goal_max - self.goal_min) + self.goal_min

    @torch.no_grad()
    def generate_candidates(self, history, planning_context, num_candidates=16, seed=0):
        if num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        if not isinstance(planning_context, PlanningContext):
            raise TypeError("planning_context must be a PlanningContext with the task target lane")
        history = self._validate_history(history)
        goal = self.predict_goal(history)
        centered_current = history["elcv"][-1].copy()
        centered_current[:2] -= self._origin
        current = torch.as_tensor(centered_current, device=self.device)
        terminal = current.clone()
        # Goal Predictor is trained only for longitudinal displacement and
        # speed change. The lane-change task supplies the lateral terminal.
        terminal[0] = float(planning_context.target_lateral_position) - float(self._origin[0])
        terminal[1] += goal[0]
        terminal[2] += goal[1]
        terminal[3] = float(planning_context.target_lateral_velocity)
        conditions = {
            0: current[None, :].repeat(num_candidates, 1),
            99: terminal[None, :].repeat(num_candidates, 1),
        }
        # Isolate RNG changes so evaluation is repeatable and does not perturb
        # unrelated Torch sampling in the same process.
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            torch.manual_seed(int(seed))
            samples = self.diffusion.conditional_sample(conditions, verbose=False).trajectories
        array = samples.detach().cpu().numpy()
        candidates = []
        for row in array:
            sampled = row[:, 2:6].copy()
            actions = row[:, :2].copy()
            # The diffusion checkpoint is trained with the observation at
            # index 0. Runtime candidates are future-only, so retain t+0.1
            # through t+9.9 and append a kinematic t+10.0 state.
            states = sampled[1:].copy()
            final = sampled[-1].copy()
            ax, ay = actions[-1]
            final[0] += final[3] * 0.1 + 0.5 * ay * 0.1 ** 2
            final[1] += final[2] * 0.1 + 0.5 * ax * 0.1 ** 2
            final[2] += ax * 0.1
            final[3] += ay * 0.1
            states = np.concatenate([states, final[None, :]], axis=0)
            future_actions = np.concatenate([actions[1:], actions[-1:]], axis=0)
            states[:, :2] += self._origin
            candidates.append(TrajectoryCandidate(states=states, actions=future_actions, dt=0.1))
        return candidates
