"""Feasibility gate and the confirmed r_private - alpha*Phi reward selector."""
import numpy as np
import torch

from .normalization import RewardPredictorNormalizer


class CandidateSelector:
    def __init__(self, elcv_discriminator, rfv_discriminator, *, gamma=0.99,
                 validated_ax=(-1.5, 1.5), validated_ay=(-1.5, 1.5),
                 lane_check=None, overlap_distance=1.0):
        self.elcv_discriminator = elcv_discriminator
        self.rfv_discriminator = rfv_discriminator
        self.gamma = float(gamma)
        self.validated_ax = tuple(validated_ax)
        self.validated_ay = tuple(validated_ay)
        self.lane_check = lane_check
        self.overlap_distance = float(overlap_distance)
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0,1]")

    @staticmethod
    def _state(candidate):
        trajectories = [candidate.elcv.states, candidate.rfv.states,
                        candidate.rlv_current.states, candidate.rlv_target.states]
        e, r, c, t = [RewardPredictorNormalizer.transform(x) for x in trajectories]
        return np.concatenate([e, e-r, c-e, t-e], axis=-1)

    @staticmethod
    def _history_stack(states, history_len, history_states=None):
        if history_states is None:
            raise ValueError("candidate reward scoring requires observed four-vehicle history")
        history_states = np.asarray(history_states, dtype=np.float32)
        if history_states.ndim != 2 or history_states.shape[1] != 16 or len(history_states) < history_len - 1:
            raise ValueError(f"history states must contain at least {history_len - 1} finite 16D frames")
        if not np.isfinite(history_states).all():
            raise ValueError("history states must be finite")
        window = list(history_states[-(history_len - 1):]) if history_len > 1 else []
        result = []
        for state in states:
            window.append(state)
            result.append(np.concatenate(window[-history_len:]))
        return np.asarray(result, dtype=np.float32)

    def _feasibility(self, candidate):
        reasons = []
        for role in ("elcv", "rfv"):
            trajectory = getattr(candidate, role)
            if not np.isfinite(trajectory.states).all():
                reasons.append(f"{role}_nonfinite")
            if trajectory.actions is not None:
                ax, ay = trajectory.actions[:, 0], trajectory.actions[:, 1]
                if np.any((ax < self.validated_ax[0]) | (ax > self.validated_ax[1])):
                    reasons.append(f"{role}_longitudinal_outside_calibration")
                if np.any((ay < self.validated_ay[0]) | (ay > self.validated_ay[1])):
                    reasons.append(f"{role}_lateral_outside_calibration")
        roles = ("elcv", "rfv", "rlv_current", "rlv_target")
        for index, first in enumerate(roles):
            for second in roles[index + 1:]:
                gap = np.linalg.norm(getattr(candidate, first).states[:, :2] -
                                     getattr(candidate, second).states[:, :2], axis=1)
                if np.any(gap < self.overlap_distance):
                    reasons.append(f"{first}_{second}_overlap")
        if self.lane_check is not None:
            for role in ("elcv", "rfv"):
                if not bool(self.lane_check(role, getattr(candidate, role).states)):
                    reasons.append(f"{role}_off_road_or_lane_infeasible")
        return reasons

    @staticmethod
    def _reward(discriminator, state, history_states=None):
        device = next(discriminator.parameters()).device
        history_len = int(getattr(discriminator, "history_len", 11))
        stacked = CandidateSelector._history_stack(state, history_len, history_states)
        with torch.no_grad():
            tensor = torch.as_tensor(stacked, dtype=torch.float32, device=device)
            if hasattr(discriminator, "get_reward"):
                reward = discriminator.get_reward(tensor)[0]
            elif hasattr(discriminator, "calculate_reward"):
                reward = discriminator.calculate_reward(tensor)
            else:
                raise TypeError("SR-AIRL discriminator must expose get_reward or calculate_reward")
        return reward.reshape(-1).detach().cpu().numpy()

    @staticmethod
    def _observed_history_frames(history):
        if history is None:
            raise ValueError("candidate selection requires the observed four-vehicle history")
        required = {"elcv", "rfv", "rlv_current", "rlv_target"}
        if set(history) != required:
            raise ValueError(f"observed history roles must be exactly {sorted(required)}")
        arrays = {role: np.asarray(history[role], dtype=np.float32) for role in required}
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1 or any(value.ndim != 2 or value.shape[1] != 4 or not np.isfinite(value).all()
                                    for value in arrays.values()):
            raise ValueError("observed role histories must be aligned finite [N,4] [y,x,vx,vy]")
        e = RewardPredictorNormalizer.transform(arrays["elcv"])
        r = RewardPredictorNormalizer.transform(arrays["rfv"])
        c = RewardPredictorNormalizer.transform(arrays["rlv_current"])
        t = RewardPredictorNormalizer.transform(arrays["rlv_target"])
        return np.concatenate([e, e-r, c-e, t-e], axis=-1)

    def select(self, candidates, history=None):
        observed_frames = self._observed_history_frames(history)
        feasible = []
        diagnostics = []
        for candidate in candidates:
            reasons = self._feasibility(candidate)
            candidate.feasibility = {"feasible": not reasons, "reasons": reasons}
            if reasons:
                diagnostics.append({"candidate_id": candidate.candidate_id, "rejected": reasons})
                continue
            state = self._state(candidate)
            re = self._reward(self.elcv_discriminator, state, observed_frames)
            rr = self._reward(self.rfv_discriminator, state, observed_frames)
            weights = self.gamma ** np.arange(len(state))
            candidate.reward_elcv = float(weights @ re)
            candidate.reward_rfv = float(weights @ rr)
            candidate.joint_return = candidate.reward_elcv + candidate.reward_rfv
            feasible.append(candidate)
            diagnostics.append({"candidate_id": candidate.candidate_id,
                                "reward_elcv": candidate.reward_elcv,
                                "reward_rfv": candidate.reward_rfv,
                                "joint_return": candidate.joint_return})
        if not feasible:
            raise RuntimeError("all joint candidates failed hard feasibility checks")
        selected = max(feasible, key=lambda item: item.joint_return)
        return selected, diagnostics
