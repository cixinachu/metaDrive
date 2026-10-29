"""Train SR-AIRL from explicitly role-mapped canonical NGSIM clips.

Scenario manifest schema:
{
  "scenarios": [{"dataset_id":"<24-hex-id>", "roles": {
    "elcv_id":"<track-id>", "rfv_id":"<track-id>",
    "rlv_current_id":"<track-id>", "rlv_target_id":"<track-id>"}}
  ]
}
Only training-split clips are accepted. Role mappings are mandatory.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.importer import load_clip
from datasets.srairl.expert_builder import ExpertDatasetBuilder
from env.core.types import InteractionScenario
from env.tasks.srairl_env import SRAIRLEnv
from rl.srairl.config import SRAIRLConfig
from rl.srairl.trainer import SRAIRLTrainer, SRAIRLTrainingConfig


def build_training_inputs(path, config):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = manifest.get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise ValueError("scenario manifest must contain a non-empty 'scenarios' list")
    scenarios, expert = [], []
    builder = ExpertDatasetBuilder(config=config, dt=0.1)
    seen = set()
    for row in rows:
        clip, source_manifest = load_clip(row["dataset_id"])
        if source_manifest.get("split") != "train":
            raise ValueError(f"SR-AIRL training cannot consume split={source_manifest.get('split')}")
        if clip.get("source") != "ngsim":
            raise ValueError("the current SR-AIRL experiment is defined on NGSIM expert trajectories")
        if source_manifest["id"] in seen:
            raise ValueError("duplicate clip IDs do not create independent scenarios")
        seen.add(source_manifest["id"])
        roles = row.get("roles")
        if not isinstance(roles, dict):
            raise ValueError("each scenario row must provide explicit roles mapping")
        scenario = InteractionScenario.from_clip(clip, source_manifest, **roles,
            scenario_id=row.get("scenario_id", source_manifest["id"]))
        scenarios.append(SRAIRLEnv(scenario, config=config))
        expert.extend(builder.build(scenario))
    if not expert:
        raise ValueError("no expert transitions could be constructed")
    return scenarios, expert


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", required=True, type=Path, help="JSON with clip IDs and explicit ELCV/RFV/leader IDs")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "srairl")
    parser.add_argument("--episodes", type=int, default=110001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    config = SRAIRLConfig.from_mapping()
    scenarios, expert = build_training_inputs(args.scenarios, config)
    training = SRAIRLTrainingConfig(episodes=args.episodes, seed=args.seed, device=args.device)
    trainer = SRAIRLTrainer(scenarios, expert, config=config, training_config=training, output_dir=args.output)
    history = trainer.train(callback=lambda row: print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True))
    print(f"Training finished: episodes={trainer.episode}, checkpoints={args.output}")


if __name__ == "__main__":
    main()
