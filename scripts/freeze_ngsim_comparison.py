"""Freeze the v1+v2 NGSIM lists as one auditable comparison protocol."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    source_paths = [ROOT / f"datasets/experiments/ngsim-mainline-v{i}/selected.json" for i in (1, 2)]
    source_lists = [json.loads(p.read_text()) for p in source_paths]
    seen, clips = set(), []
    for path, source in zip(source_paths, source_lists):
        for original in source["clips"]:
            entry = dict(original)
            key = (entry["location"], entry["origin"], entry["ego_id"], entry["start_ms"])
            if key in seen:
                raise ValueError(f"Duplicate NGSIM window across source lists: {key}")
            seen.add(key)
            clip_manifest = json.loads((ROOT / "datasets/processed" / entry["clip_id"] / "manifest.json").read_text())
            entry.update(source_experiment=source["version"], source_selection_manifest_sha256=sha(path),
                         selection_config_sha256=clip_manifest["selection_config_sha256"],
                         selection_code_sha256=clip_manifest["selection_code_sha256"],
                         source_sha256=clip_manifest["source_sha256"]["trajectories.csv"],
                         clip_sha256=clip_manifest["clip_sha256"],
                         recording_cluster=clip_manifest["recording_group"])
            clips.append(entry)
    # Small, frozen, class-balanced checkpoint-selection panel; final validation uses every validation clip.
    validation_selection = []
    classes = sorted({c["event_class"] for c in clips})
    for event_class in classes:
        candidates = [c for c in clips if c["split"] == "validation" and c["event_class"] == event_class]
        candidates.sort(key=lambda c: hashlib.sha256(f"ngsim-comparison-v3-validation-panel:{c['clip_id']}".encode()).hexdigest())
        validation_selection.extend(c["clip_id"] for c in candidates[:10])
    config = {
        "version": "ngsim-comparison-v3",
        "source_manifests": {s["version"]: {"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p, s in zip(source_paths, source_lists)},
        "full_source_sha256": source_lists[0]["source_sha256"],
        "membership_sha256": canonical_sha([c["clip_id"] for c in clips]),
        "split_counts": dict(sorted(Counter(c["split"] for c in clips).items())),
        "class_counts": {split: dict(sorted(Counter(c["event_class"] for c in clips if c["split"] == split).items())) for split in ("train", "validation", "test_ood")},
        "validation_checkpoint_panel": {"clips_per_class": 10, "selection_rule": "Sort validation clip IDs by SHA256('ngsim-comparison-v3-validation-panel:'+clip_id), take first 10 per class.", "clip_ids": validation_selection},
        "training": {"algorithms": ["sac", "sac_lagrangian", "wcsac", "wcsac_iqn"], "common_timesteps": 100000, "seeds": [0, 1, 2, 3, 4], "clip_sampling": "uniform event-class draw, then uniform clip draw with replacement; RNG seeded by run seed, so episode-index scenario sequence is shared across algorithms"},
        "checkpoint_selection": {"validation_split": "validation", "constraint": "mean discounted episode cost <= cost_limit from the common training settings", "priority": ["minimize collision_rate", "minimize out_of_road_rate", "minimize discounted_cost_cvar", "maximize success_rate", "maximize mean route_completion"], "fallback": "If no checkpoint satisfies the cost limit, minimize mean discounted episode cost, then minimize collision rate, then maximize success_rate."},
        "evaluation": {"validation_final": "all listed validation clips once per selected model", "test": "all listed test_ood clips once per selected model; no test-driven selection", "metrics": ["success_rate", "collision_rate", "out_of_road_rate", "mean_cost", "mean_route_completion", "discounted_cost_cvar"], "cluster": "recording_cluster; cluster bootstrap, never IID clip/frame bootstrap", "tail_fraction": 0.1},
        "interpretation": "固定 NGSIM 轨迹回放下的策略比较。Background traffic is recorded open-loop replay and does not react to ego actions; do not claim closed-loop interactive traffic safety.",
    }
    config["config_sha256"] = canonical_sha(config)
    out_dir = ROOT / "datasets/experiments/ngsim-comparison-v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "configs/datasets/ngsim_comparison_v3.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    manifest = {"version": config["version"], "config_sha256": config["config_sha256"], "source_sha256": config["full_source_sha256"], "clips": clips, "validation_selection": validation_selection, "source_manifests": config["source_manifests"], "membership_sha256": config["membership_sha256"]}
    (out_dir / "selected.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "validation_selection.json").write_text(json.dumps(validation_selection, indent=2) + "\n")
    print(json.dumps({"version": config["version"], "clips": len(clips), "split_counts": config["split_counts"], "class_counts": config["class_counts"], "validation_panel": len(validation_selection), "config_sha256": config["config_sha256"], "membership_sha256": config["membership_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
