"""Adapters between local SR-AIRL checkpoints and upstream reward guidance."""
import json
from pathlib import Path

import torch


def load_reward_discriminator(model_dir, device="cpu"):
    """Rebuild the local FV discriminator from its state-dict checkpoint.

    Upstream Step 3 expected a pickled module from its own Step 1. Studio saves
    state dictionaries instead. This adapter makes that boundary explicit and
    rejects incomplete or incompatible model directories.
    """
    directory = Path(model_dir).resolve()
    metadata_path = directory / "config.json"
    checkpoint_path = directory / "discriminator_rfv.pt"
    if not metadata_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"SR-AIRL model must contain config.json and discriminator_rfv.pt: {directory}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("algorithm") != "SR-AIRL" or "srairl" not in metadata:
        raise ValueError(f"Not a compatible local SR-AIRL model: {directory}")

    from rl.srairl.config import SRAIRLConfig
    from rl.srairl.discriminator import AnisotropicInteractionPotential, SocialAIRLDiscrim

    cfg = SRAIRLConfig.from_mapping(metadata["srairl"])
    state_dim = cfg.frame_state_dim * cfg.history_len
    potential = AnisotropicInteractionPotential(
        cfg.potential_longitudinal_scale,
        cfg.potential_lateral_scale,
        cfg.direction_lambda,
        pos_lat_scale=cfg.pos_lat_scale,
        pos_long_scale=cfg.pos_long_scale,
        vel_scale=cfg.vel_scale,
        vel_offset=cfg.vel_offset,
    )
    discriminator = SocialAIRLDiscrim(
        state_dim,
        cfg.gamma,
        potential,
        cfg.alpha_init,
        role="fv",
        history_len=cfg.history_len,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    discriminator.load_state_dict(state, strict=True)
    discriminator.to(device).eval()
    return discriminator
