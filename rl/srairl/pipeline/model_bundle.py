"""Model references and compatibility checks; checkpoint bytes are not copied."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "obs_len": 20, "pred_len": 100, "horizon": 100, "dt": 0.1,
    "state_dim": 16,
}


def sha256(path):
    digest = hashlib.sha256()
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for item in files:
        if path.is_dir():
            digest.update(str(item.relative_to(path)).encode("utf-8"))
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SRAIRLModelBundle:
    planner_goal_path: Path
    planner_diffusion_path: Path
    reward_guide_path: Path
    predictor_path: Path
    reward_model_path: Path
    planner_metadata: dict
    predictor_metadata: dict
    normalization: dict

    def __post_init__(self):
        for name in ("planner_goal_path", "planner_diffusion_path", "reward_guide_path", "predictor_path"):
            path = Path(getattr(self, name)).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{name} checkpoint is missing: {path}")
            object.__setattr__(self, name, path)
        reward_path = Path(self.reward_model_path).expanduser().resolve()
        reward_valid = (reward_path.is_dir() and (reward_path / "config.json").is_file()) \
            or reward_path.is_file()
        if not reward_valid:
            raise FileNotFoundError(f"SR-AIRL reward checkpoint directory/file is missing: {reward_path}")
        object.__setattr__(self, "reward_model_path", reward_path)
        self.validate_metadata()

    def validate_metadata(self):
        combined = {**self.planner_metadata, **self.predictor_metadata}
        errors = []
        for key, expected in EXPECTED.items():
            if key in combined and combined[key] != expected:
                errors.append(f"{key}={combined[key]!r}, expected {expected!r}")
        planner_revision = self.planner_metadata.get("source_dataset_revision")
        predictor_revision = self.predictor_metadata.get("source_dataset_revision")
        if planner_revision and predictor_revision and planner_revision != predictor_revision:
            errors.append("planner and predictor source dataset revisions differ")
        planner_split=self.planner_metadata.get("split_version")
        predictor_split=self.predictor_metadata.get("split_version")
        if planner_split and predictor_split and planner_split!=predictor_split:
            errors.append("planner and predictor split versions differ")
        if self.planner_metadata.get("normalization_version") not in (None, "planner-goal-v1"):
            errors.append("Planner checkpoint normalization does not match planner-goal-v1")
        if self.predictor_metadata.get("normalization_version") not in (None, "reward-predictor-v1"):
            errors.append("Predictor checkpoint normalization does not match reward-predictor-v1")
        expected_normalizers = {
            "planner_goal": "planner-goal-v1",
            "reward_predictor": "reward-predictor-v1",
        }
        for name, expected in expected_normalizers.items():
            value = self.normalization.get(name)
            if value and value != expected:
                errors.append(f"normalization {name}={value!r}, expected {expected!r}")
        for metadata in (self.planner_metadata, self.predictor_metadata):
            version = metadata.get("normalization_version")
            if version and version not in expected_normalizers.values():
                errors.append(f"unsupported checkpoint normalization_version={version!r}")
        if errors:
            raise ValueError("incompatible SR-AIRL model bundle: " + "; ".join(errors))

    def to_manifest(self):
        refs = {
            "planner_goal": self.planner_goal_path,
            "planner": self.planner_diffusion_path,
            "reward_guide": self.reward_guide_path,
            "predictor": self.predictor_path,
            "srairl_reward": self.reward_model_path,
        }
        return {
            "schema_version": "srairl-model-bundle-v1",
            "checkpoints": {key: {"path": str(path), "sha256": sha256(path)} for key, path in refs.items()},
            "planner_metadata": self.planner_metadata,
            "predictor_metadata": self.predictor_metadata,
            "normalization": self.normalization,
            **EXPECTED,
        }

    def write_manifest(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_manifest(), indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def from_checkpoints(cls, *, planner_goal_path, planner_diffusion_path,
                         reward_guide_path, predictor_path, reward_model_path):
        """Build only from metadata-bearing training outputs and matching data."""
        paths = [Path(planner_goal_path), Path(planner_diffusion_path),
                 Path(reward_guide_path), Path(predictor_path)]
        payloads = [torch_load_metadata(path) for path in paths]
        goal_meta, planner_meta, guide_meta, predictor_meta = [p.get("metadata", {}) for p in payloads]
        requirements = (
            ("Goal Predictor", goal_meta, {"obs_len", "pred_len", "horizon", "dt","source_dataset_revision","component_dataset_revision","split_version"}),
            ("Planner", planner_meta, {"obs_len", "pred_len", "horizon", "dt","source_dataset_revision","component_dataset_revision","split_version"}),
            ("Reward Guide", guide_meta, {"obs_len", "pred_len", "dt", "state_dim", "history_len","source_dataset_revision","component_dataset_revision","split_version"}),
            ("Predictor", predictor_meta, {"obs_len", "pred_len", "dt", "state_dim", "history_len","source_dataset_revision","component_dataset_revision","split_version"}),
        )
        for name, metadata, required in requirements:
            missing = required - set(metadata)
            if missing:
                raise ValueError(f"{name} checkpoint lacks compatibility metadata: {sorted(missing)}")
        source_revisions={meta['source_dataset_revision'] for _,meta,_ in requirements}
        split_versions={meta['split_version'] for _,meta,_ in requirements}
        if len(source_revisions)!=1:
            raise ValueError("all generative components must use the same source MetaDrive dataset revision")
        if len(split_versions)!=1:
            raise ValueError("all generative components must use the same episode split version")
        goal_arch = payloads[0].get("architecture", {})
        if goal_arch.get("context_dim") != 128 or goal_arch.get("input_size") != 2:
            raise ValueError("Goal Predictor architecture must use context_dim=128 and input_size=2")
        if planner_meta.get("transition_dim") != 6 or planner_meta.get("n_diffusion_steps") != 20:
            raise ValueError("Planner must use [ax,ay,y,x,vx,vy] and 20 denoising steps")
        for name, payload in (("Reward Guide", payloads[2]), ("Predictor", payloads[3])):
            architecture = payload.get("architecture", {})
            if architecture.get("context_dim") != 64 or architecture.get("T") != 100:
                raise ValueError(f"{name} architecture must use context_dim=64 and T=100")
        reward_dir = Path(reward_model_path).resolve()
        if reward_dir.is_file():
            reward_dir = reward_dir.parent
        reward_config = json.loads((reward_dir / "config.json").read_text(encoding="utf-8"))
        reward_settings = reward_config.get("srairl", {})
        if reward_config.get("reward_version")!="temporal-dynamic-v2" or reward_settings.get("frame_state_dim") != 16 or reward_settings.get("history_len") != 11:
            raise ValueError("SR-AIRL reward model must use 16D frames and history_len=11")
        action_scale=reward_config.get("training",{}).get("action_scale")
        if action_scale is None or action_scale>1.5+1e-8:
            raise ValueError("SR-AIRL policy action scale must remain inside the calibrated ±1.5 m/s² domain")
        reward_revision=reward_config.get("source_dataset_revision")
        if reward_revision and reward_revision not in source_revisions:
            raise ValueError("SR-AIRL reward checkpoint uses a different MetaDrive source dataset revision")
        planner_contract = {**planner_meta, "goal_architecture": goal_arch,
                            "planner_contract": {"transition_order": ["ax", "ay", "y", "x", "vx", "vy"],
                                                 "n_diffusion_steps": 20}}
        predictor_contract = {**predictor_meta,
                              "reward_guide_architecture": payloads[2].get("architecture", {}),
                              "predictor_architecture": payloads[3].get("architecture", {}),
                              "reward_guide_metadata": guide_meta}
        # The reward discriminator intentionally comes from SR-AIRL training on
        # expert transitions and may have a different expert-data revision.
        return cls(Path(planner_goal_path), Path(planner_diffusion_path),
                   Path(reward_guide_path), Path(predictor_path), Path(reward_model_path),
                   planner_contract, predictor_contract,
                   {"planner_goal": "planner-goal-v1",
                    "reward_predictor": "reward-predictor-v1"})

    @classmethod
    def from_manifest(cls, path):
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "srairl-model-bundle-v1":
            raise ValueError("unsupported model bundle schema")
        refs = manifest["checkpoints"]
        for key, entry in refs.items():
            if sha256(entry["path"]) != entry["sha256"]:
                raise ValueError(f"checkpoint hash changed: {key}")
        return cls(
            Path(refs["planner_goal"]["path"]), Path(refs["planner"]["path"]),
            Path(refs["reward_guide"]["path"]), Path(refs["predictor"]["path"]),
            Path(refs["srairl_reward"]["path"]), manifest["planner_metadata"],
            manifest["predictor_metadata"], manifest["normalization"],
        )


def torch_load_metadata(path):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must be a metadata-bearing mapping: {path}")
    return payload
