"""Run the pre-registered four-algorithm NGSIM comparison.

Default behavior is a dry run. Pass --execute to launch training and evaluation.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "datasets/experiments/ngsim-comparison-v3/selected.json"
RUNNER = ROOT / "scripts/run_ngsim_experiment.py"
CONFIG = ROOT / "configs/datasets/ngsim_comparison_v3.json"
ALGORITHMS = ("sac", "sac_lagrangian", "wcsac", "wcsac_iqn")


def call(args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run([str(x) for x in args], cwd=ROOT, check=True)


def eval_model(algorithm, seed, model, split, out, panel=None):
    cmd = [sys.executable, RUNNER, "eval", "--algorithm", algorithm, "--seed", seed,
           "--model", model, "--split", split, "--output", out, "--experiment", EXPERIMENT]
    if panel is not None:
        cmd += ["--clip-ids-file", panel]
    call(cmd)
    return json.loads((Path(out) / "summary.json").read_text())


def rank_checkpoint(summary, cost_limit):
    return (-summary["collision_rate"], -summary["out_of_road_rate"],
            -summary["discounted_cost_cvar"], summary["success_rate"], summary["mean_route_completion"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--timesteps", type=int, default=100000)
    p.add_argument("--checkpoint-every", type=int, default=25000)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--settings", type=Path, help="One common JSON settings override for all four algorithms")
    p.add_argument("--output", type=Path, default=ROOT / "results/ngsim-comparison-v3")
    p.add_argument("--execute", action="store_true", help="Actually run the full training and evaluation plan")
    args = p.parse_args()
    manifest = json.loads(EXPERIMENT.read_text())
    config = json.loads(CONFIG.read_text())
    if args.timesteps <= 0 or args.checkpoint_every <= 0:
        p.error("timesteps and checkpoint interval must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        p.error("seeds must be unique")
    panel_path = ROOT / "datasets/experiments/ngsim-comparison-v3/validation_selection.json"
    plan = {"protocol": manifest["version"], "membership_sha256": manifest["membership_sha256"],
            "config_sha256": manifest["config_sha256"], "algorithms": list(ALGORITHMS), "seeds": args.seeds,
            "timesteps_per_run": args.timesteps, "checkpoint_every": args.checkpoint_every,
            "train_clips": len([x for x in manifest["clips"] if x["split"] == "train"]),
            "validation_panel_clips_per_checkpoint": len(manifest["validation_selection"]),
            "validation_final_clips_per_selected_model": len([x for x in manifest["clips"] if x["split"] == "validation"]),
            "test_clips_once_per_selected_model": len([x for x in manifest["clips"] if x["split"] == "test_ood"]),
            "execute": args.execute}
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        print("Dry run only. Review the fixed manifest and rerun with --execute to launch the experiment.")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    all_rows = []
    for algorithm in ALGORITHMS:
        for seed in args.seeds:
            base = args.output / algorithm / f"seed-{seed}"
            train_dir = ROOT / "models/ngsim-comparison-v3" / algorithm / f"seed-{seed}"
            train_dir.parent.mkdir(parents=True, exist_ok=True)
            train_cmd = [sys.executable, RUNNER, "train", "--algorithm", algorithm, "--seed", seed,
                         "--timesteps", args.timesteps, "--checkpoint-every", args.checkpoint_every,
                         "--device", args.device, "--output", train_dir, "--experiment", EXPERIMENT]
            if args.settings:
                train_cmd += ["--settings", args.settings]
            call(train_cmd)
            candidates = sorted(train_dir.glob("checkpoint_*.zip"))
            if not candidates or max(int(x.stem.rsplit("_", 1)[1]) for x in candidates) < args.timesteps:
                candidates.append(train_dir / "model.zip")
            validation = []
            for i, model in enumerate(candidates):
                panel_out = base / "checkpoint-validation" / f"{i:03d}"
                score = eval_model(algorithm, seed, model, "validation", panel_out, panel_path)
                validation.append({"model": str(model), "sha256": score["model_sha256"], "summary": score})
            cost_limit = float(json.loads((train_dir / "settings.json").read_text()).get("cost_limit", 5.0))
            feasible = [x for x in validation if x["summary"]["mean_discounted_cost"] <= cost_limit]
            if feasible:
                selected = max(feasible, key=lambda x: rank_checkpoint(x["summary"], cost_limit))
                selection_reason = f"Validation mean discounted cost <= {cost_limit}; minimize collision, out-of-road, then CVaR; maximize success and route completion as tie-breakers."
            else:
                selected = min(validation, key=lambda x: (x["summary"]["mean_discounted_cost"], x["summary"]["collision_rate"], -x["summary"]["success_rate"]))
                selection_reason = f"No checkpoint meets validation mean discounted-cost limit {cost_limit}; minimize mean discounted cost, then collision, then maximize success."
            (base / "checkpoint_selection.json").parent.mkdir(parents=True, exist_ok=True)
            (base / "checkpoint_selection.json").write_text(json.dumps({"selected": selected, "reason": selection_reason,
                "cost_limit": cost_limit, "all_validation_panel_results": validation}, ensure_ascii=False, indent=2) + "\n")
            final_validation = eval_model(algorithm, seed, selected["model"], "validation", base / "validation-final")
            final_test = eval_model(algorithm, seed, selected["model"], "test_ood", base / "test-once")
            all_rows.append({"algorithm": algorithm, "seed": seed, "model": selected["model"], "model_sha256": selected["sha256"],
                             "validation": final_validation, "test_ood": final_test})
            (base / "selected_model.json").write_text(json.dumps({"algorithm": algorithm, "seed": seed,
                "model": selected["model"], "model_sha256": selected["sha256"], "validation": final_validation,
                "test_ood": final_test, "protocol": manifest["version"], "selection_reason": selection_reason}, ensure_ascii=False, indent=2) + "\n")
    result_path = args.output / "comparison_results.json"
    result_path.write_text(json.dumps({"protocol": manifest["version"], "membership_sha256": manifest["membership_sha256"],
                                       "runs": all_rows}, ensure_ascii=False, indent=2) + "\n")
    with (args.output / "comparison_results.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["algorithm", "seed", "model_sha256", "split", "success_rate", "collision_rate", "out_of_road_rate", "mean_cost", "mean_discounted_cost", "mean_route_completion", "discounted_cost_cvar", "episodes", "recording_clusters"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for run in all_rows:
            for split in ("validation", "test_ood"):
                s = run[split]
                w.writerow({"algorithm": run["algorithm"], "seed": run["seed"], "model_sha256": run["model_sha256"], "split": split,
                            **{k: s[k] for k in fields if k in s}, "recording_clusters": ";".join(s["evaluation_recording_clusters"])})
    print(result_path)


if __name__ == "__main__":
    main()
