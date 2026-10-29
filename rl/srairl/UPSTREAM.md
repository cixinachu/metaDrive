# SR-AIRL source provenance and compatibility notes

Source: [cixinachu/SR-AIRL](https://github.com/cixinachu/SR-AIRL), inspected at commit `a762fd1c4099ce27eda1fa4f26372c007fab0129` (MIT; copyright Kehua Chen, 2024). Upstream license is retained in `LICENSE.upstream`.

The upstream `Step1_Reward_Recovery/main_helbing.py` imports `reward_config.py`, but that file is absent from the public repository tree and history. The study supplied these constants in the current session; they are now recorded in `reward_config.py` and used by `SRAIRLConfig`, the observation adapter, reward potential and trainer.

Verified upstream facts:

- `data_process.py` maps the `id_x` trajectory to `lcv`, `id_y` to `fv`, the unsuffixed `id` to `nlv`, and `id_z` to `olv`. These symbolic names are preserved; mapping them to ELCV/RFV requires a source scenario's explicit role assignment. The repository does not contain the originating pair table needed to prove all role semantics for a given imported clip.
- Each role frame is normalized as `[lateral position, longitudinal position, longitudinal speed, lateral speed]`, relative features are `[lcv, lcv-fv, nlv-lcv, olv-lcv]`, giving 16 values.
- `NGSIMEnv.step` uses `dt=0.1`, updates velocity first, then position with the updated velocity, for both controlled vehicles. NLV/OLV are read from the next recorded frame.
- The public `SocialAIRLDiscrim` reward is `private_g(state) - softplus(alpha_raw) * interaction_potential(state)`. Its AIRL energy is `reward + gamma*(1-done)*h(next_state) - h(state)`. It is not the potential-difference formula described in the platform design draft.
- The public trainer gives both PPO policies the same joint state, concatenates their two 2D accelerations, and uses separate LCV/FV discriminators.

The local `ppo.py`, `discriminator.py` and `trainer.py` now provide a legacy-kinematic dual-policy training path. The trainer keeps the upstream alternating schedule and PPO/AIRL objective; however, it has not been numerically compared to upstream because the original role-pair source table and expert pickle are not present. Explicit roles are required by the CLI manifest. MetaDrive-physics closed-loop transfer remains gated on a validated two-vehicle synchronization bridge.


`ExpertDatasetBuilder` writes canonical dual-actor transitions from explicitly assigned clip tracks. Its acceleration labels are backward differences of the canonical clip velocity and require one frame of pre-roll. Since this repository's general importer derives velocity with `numpy.gradient` while the original SR-AIRL preprocessor uses framewise differences, these exported transitions are not claimed numerically identical to the original `expert_data.pkl` until both are built from the same raw trajectories and preprocessing path.
