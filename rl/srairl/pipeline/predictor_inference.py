"""Reward-Guided Step4 response prediction with full-state reconstruction."""
import numpy as np
import torch

from .normalization import RewardPredictorNormalizer, reconstruct_predictor_deltas
from .schema import TrajectoryCandidate


def _with_acceleration(states, dt):
    states = np.asarray(states, dtype=np.float32)
    acceleration = np.zeros_like(states[:, 2:4])
    if len(states) > 1:
        acceleration[1:] = np.diff(states[:, 2:4], axis=0) / dt
        acceleration[0] = acceleration[1]
    return np.concatenate([states, acceleration], axis=-1)


class RewardGuidedResponsePredictor:
    def __init__(self, model_bundle, device="cpu", guide_scale=10.0):
        self.bundle = model_bundle
        self.device = torch.device(device)
        self.guide_scale = float(guide_scale)
        from rl.srairl.upstream.Step3_Reward_Guide.model.backbone import NoisyReward
        from rl.srairl.upstream.Step4_Predictor.model.backbone import TransformerConcatLinear
        from rl.srairl.upstream.Step4_Predictor.model.ddpm import GaussianDiffusionSampler

        guide_data = torch.load(bundle_path(model_bundle.reward_guide_path), map_location=self.device, weights_only=True)
        pred_data = torch.load(bundle_path(model_bundle.predictor_path), map_location=self.device, weights_only=True)
        self._validate_checkpoint(guide_data, "reward guide")
        self._validate_checkpoint(pred_data, "predictor")
        guide_arch = guide_data.get("architecture", {})
        pred_arch = pred_data.get("architecture", {})
        if guide_arch.get("context_dim", 64) != 64 or pred_arch.get("context_dim", 64) != 64:
            raise ValueError("only the locally trained context_dim=64 checkpoint contract is supported")
        self.guide = NoisyReward(64, 100).to(self.device)
        self.guide.load_state_dict(guide_data["state_dict"], strict=True)
        self.guide.eval()
        self.model = TransformerConcatLinear(64, 100).to(self.device)
        self.model.load_state_dict(pred_data["state_dict"], strict=True)
        self.model.eval()
        self.sampler = GaussianDiffusionSampler(self.model, beta_1=1e-4, beta_T=0.05, T=100).to(self.device)
        self.sampler.guide_scale = self.guide_scale
        self.sampler.eval()

    @staticmethod
    def _validate_checkpoint(data, name):
        if not isinstance(data, dict) or "state_dict" not in data:
            raise ValueError(f"{name} checkpoint is legacy full-module format; retrain with current checkpoint contract")
        metadata = data.get("metadata", {})
        if metadata and (metadata.get("obs_len", 20) != 20 or metadata.get("pred_len", 100) != 100):
            raise ValueError(f"{name} checkpoint dimensions are incompatible")

    @torch.no_grad()
    def predict(self, rfv_history, elcv_future, rlv_current_future, rlv_target_future, *, seed):
        history = np.asarray(rfv_history, dtype=np.float32)
        if history.shape != (20, 4) or not np.isfinite(history).all():
            raise ValueError("RFV history must be finite [20,4] [y,x,vx,vy]")
        elcv = elcv_future.states if isinstance(elcv_future, TrajectoryCandidate) else np.asarray(elcv_future)
        current = rlv_current_future.states if isinstance(rlv_current_future, TrajectoryCandidate) else np.asarray(rlv_current_future)
        target = rlv_target_future.states if isinstance(rlv_target_future, TrajectoryCandidate) else np.asarray(rlv_target_future)
        horizon = len(elcv)
        if any(x.shape != (horizon, 4) for x in (current, target)) or horizon != 100:
            raise ValueError("ELCV and RLV futures must each be [100,4]")
        hist6 = _with_acceleration(history, 0.1)
        cond6 = _with_acceleration(elcv, 0.1)
        cond6[:, :2] -= hist6[0:1, :2]
        hist_t = torch.as_tensor(hist6[None], device=self.device)
        cond_t = torch.as_tensor(cond6[None], device=self.device)
        context = self.model.context(hist_t, cond_t)
        n_lcv = torch.as_tensor(RewardPredictorNormalizer.transform(elcv[:, :4])[None], device=self.device)
        n_nlv = torch.as_tensor(RewardPredictorNormalizer.transform(current[:, :4])[None], device=self.device)
        n_olv = torch.as_tensor(RewardPredictorNormalizer.transform(target[:, :4])[None], device=self.device)
        state = {
            "rfv_history_last": torch.as_tensor(history[-1:, :], device=self.device),
            "n_lcv": n_lcv, "n_nlv": n_nlv, "n_olv": n_olv,
        }
        # The sampler remains differentiable only through its reward guide.
        # All model parameters are frozen by eval(); gradients are needed for
        # the candidate deltas during classifier guidance.
        device_index = (self.device.index if self.device.index is not None else torch.cuda.current_device()) \
            if self.device.type == "cuda" else None
        devices = [device_index] if device_index is not None else []
        with torch.random.fork_rng(devices=devices):
            seed = int(seed) % (2 ** 63 - 1)
            torch.random.default_generator.manual_seed(seed)
            if device_index is not None:
                torch.cuda.default_generators[device_index].manual_seed(seed)
            with torch.enable_grad():
                deltas, _ = self.sampler(node_loc=torch.zeros((1, 100, 4), device=self.device),
                                         context=context, state=state, guide=self.guide)
        reconstructed, accel = reconstruct_predictor_deltas(
            history, deltas[0].detach().cpu().numpy(), dt=0.1)
        return TrajectoryCandidate(reconstructed, actions=accel, dt=0.1)


def bundle_path(path):
    return path
