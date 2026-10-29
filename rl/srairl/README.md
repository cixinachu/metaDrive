# SR-AIRL training and evaluation

Study constants are in `reward_config.py`. The local trainer uses two continuous PPO policies and two Social-AIRL discriminators with a shared anisotropic interaction potential. It preserves the upstream 0.1-second velocity-first kinematic update and action scale 5. The potential reconstructs physical velocities using the supplied offset (`v = normalized * VEL_SCALE + VEL_OFFSET`).

Training consumes an explicit role manifest because the general clip importer cannot infer ELCV, RFV, current-lane leader and target-lane leader. A CLI manifest looks like:

```json
{
  "scenarios": [{
    "dataset_id": "0123456789abcdef01234567",
    "roles": {
      "elcv_id": "101:0",
      "rfv_id": "107:0",
      "rlv_current_id": "98:0",
      "rlv_target_id": "113:0"
    }
  }]
}
```

The Studio task selector dispatches SR-AIRL training and deterministic actor evaluation. Choose an imported NGSIM `train` clip for training, bind all four role IDs, and set episodes/PPO epochs. For evaluation, select an NGSIM `eval` clip and a trained SR-AIRL checkpoint. Training outputs are stored in `models/<job-id>/`; evaluation records are stored in `results/<job-id>/`. The Studio world is a MetaDrive NGSIM reconstruction: ELCV and RFV are controlled by separate policy networks; other recorded vehicles remain anchored to their NGSIM reference paths. By default, a bounded IDM-like correction slows a replay vehicle when a policy-controlled car closes in from behind in the same lane. This response is a reproducible interaction baseline, not a calibrated human driver. Disable “background response” to use fixed replay. Both policy actions still use the paper-compatible legacy kinematic update and are synchronized into MetaDrive bodies; this is not MetaDrive vehicle-physics control.

The CLI remains available:

```bash
python scripts/train_srairl.py --scenarios path/to/srairl_train_scenarios.json --episodes 110001 --device auto
```

Pass `--episodes 1` for a smoke test only. Controlled cars use the upstream kinematic update. Evaluation reports tracking error and axis-aligned proxy clearance/collision plus MetaDrive contact flags; it does not establish calibrated vehicle-physics interaction safety. The upstream expert role-pair table and processed expert pickle are absent, so exact numerical parity to that expert file is not claimed.
