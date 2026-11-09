"""Train SR-AIRL on current native four-vehicle MetaDrive expert archives."""
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
    parser.add_argument("--expert-archives", required=True, nargs="+", type=Path, help="Current native MetaDrive expert archives")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "srairl")
    parser.add_argument("--episodes", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument('--ppo-epochs',type=int,default=80)
    parser.add_argument('--discriminator-every',type=int,default=150)
    args = parser.parse_args(argv)
    config = SRAIRLConfig.from_mapping()
    from datasets.srairl.expert_builder import load_transitions
    from datasets.srairl.native_windows import native_dataset_info
    from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv
    from env.tasks.srairl_interaction import SRAIRLInteractionScenario
    native_dataset_info(args.expert_archives,require_current=True)
    expert=[];scenarios=[];seen_scenes=set()
    for path in args.expert_archives:
        rows, metadata=load_transitions(path,split='train')
        expert.extend(rows)
        for scene in metadata.get('scenarios') or [metadata.get('scenario')]:
            if scene not in ('straight','curve','merge'):raise ValueError('archive has no supported MetaDrive scenario')
            if scene in seen_scenes:continue
            seen_scenes.add(scene)
            scenarios.append(SRAIRLProceduralEnv(scene,lane_num=3,
                traffic_density=float(metadata.get('traffic_density_by_scenario',{}).get(scene,metadata.get('traffic_density',0))),
                seed=args.seed,horizon=1000))
    if args.output.exists():
        raise ValueError('output directory already exists; choose a new run directory')
    training = SRAIRLTrainingConfig(episodes=args.episodes, seed=args.seed, device=args.device,
        ppo_epochs=args.ppo_epochs,discriminator_every_episodes=args.discriminator_every)
    trainer = SRAIRLTrainer(scenarios, expert, config=config, training_config=training, output_dir=args.output)
    import hashlib
    digest=hashlib.sha256()
    for path in sorted(args.expert_archives,key=lambda item:item.name):
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode('ascii'))
    trainer.source_dataset_revision=digest.hexdigest()
    try:
        history = trainer.train(callback=lambda row: print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True))
    finally:
        for env in scenarios:env.close()
    print(f"Training finished: episodes={trainer.episode}, checkpoints={args.output}")


if __name__ == "__main__":
    main()
