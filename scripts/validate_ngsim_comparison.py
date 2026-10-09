"""Audit the frozen NGSIM comparison membership and every materialized clip hash."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets/experiments/ngsim-comparison-v3/selected.json"
CONFIG = ROOT / "configs/datasets/ngsim_comparison_v3.json"


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    manifest, config = json.loads(MANIFEST.read_text()), json.loads(CONFIG.read_text())
    config_without_hash = dict(config)
    recorded_config_sha = config_without_hash.pop("config_sha256")
    assert canonical_sha(config_without_hash) == recorded_config_sha == manifest["config_sha256"]
    ids = [c["clip_id"] for c in manifest["clips"]]
    assert len(ids) == len(set(ids)) == 1058
    assert canonical_sha(ids) == manifest["membership_sha256"] == config["membership_sha256"]
    split_recordings = defaultdict(set)
    split_counts = Counter(c["split"] for c in manifest["clips"])
    for c in manifest["clips"]:
        split_recordings[c["split"]].add(c["recording_cluster"])
        d = ROOT / "datasets/processed" / c["clip_id"]
        m = json.loads((d / "manifest.json").read_text())
        assert file_sha(d / "clip.json") == c["clip_sha256"] == m["clip_sha256"]
        assert m["experiment_split"] == c["split"]
        assert m["event_class"] == c["event_class"]
        assert m["recording_group"] == c["recording_cluster"]
        assert m["selection_config_sha256"] == c["selection_config_sha256"]
    assert not (split_recordings["train"] & split_recordings["validation"])
    assert not (split_recordings["train"] & split_recordings["test_ood"])
    assert not (split_recordings["validation"] & split_recordings["test_ood"])
    panel = manifest["validation_selection"]
    panel_counts = Counter(next(c["event_class"] for c in manifest["clips"] if c["clip_id"] == i) for i in panel)
    assert len(panel) == 40 and set(panel_counts.values()) == {10}
    print(json.dumps({"status": "passed", "clips": len(ids), "split_counts": dict(split_counts),
                      "recording_blocks_by_split": {k: len(v) for k, v in split_recordings.items()},
                      "validation_panel": len(panel), "config_sha256": recorded_config_sha,
                      "membership_sha256": manifest["membership_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
