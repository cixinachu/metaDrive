"""Small, explicit ID/OOD matrix using one checkpoint and isolated engines."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / f"matrix_{time.time_ns()}")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [("id", "procedural", .10), ("low_density", "procedural", .05),
             ("high_density", "procedural", .20), ("stress", "procedural", .30),
             ("straight", "straight", .10), ("intersection", "intersection", .10),
             ("merge", "merge", .10), ("t_junction", "t_junction", .10), ("roundabout", "roundabout", .10)]
    summaries = {}
    for name, scenario, density in cases:
        output = args.output / name
        subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_env.py"), "--policy", "sac", "--model", args.model,
                        "--device", args.device,
                        "--episodes", str(args.episodes), "--scenario", scenario, "--traffic-density", str(density),
                        "--output", str(output)], check=True, cwd=ROOT)
        summaries[name] = json.loads((output / "summary.json").read_text())
        (args.output / "matrix_summary.json").write_text(json.dumps(summaries, indent=2))
