"""Reproducible random, IDM and SAC evaluation without action shields."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.research_config import load_config, merge_config, validate_config
from env.safe_metadrive_env import SafeMetaDriveEnv


def arguments(default_policy="random"):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/env/eval_id.yaml"))
    p.add_argument("--policy", choices=("random", "idm", "sac"), default=default_policy)
    p.add_argument("--model")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--scenario", choices=("procedural", "straight", "merge", "intersection", "t_junction", "roundabout"))
    p.add_argument("--traffic-density", type=float)
    p.add_argument("--seed-start", type=int)
    p.add_argument("--num-scenarios", type=int)
    p.add_argument("--horizon", type=int)
    p.add_argument("--traffic-seed", type=int)
    p.add_argument("--traffic-config", help="JSON file with traffic configuration overrides")
    p.add_argument("--reward-mode", choices=("cmdp", "metadrive"))
    p.add_argument("--cost-mode", choices=("binary", "continuous"))
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / f"evaluation_{time.time_ns()}")
    p.add_argument("--steps-csv", action="store_true")
    return p.parse_args()


def resolve_config(args):
    c = load_config(args.config)
    if args.scenario and args.scenario != "procedural":
        c = merge_config(c, json.loads((ROOT / f"configs/scenarios/{args.scenario}.yaml").read_text()))
    elif args.scenario == "procedural":
        c["environment"]["map"] = load_config()["environment"]["map"]
        c["scenario"]["type"] = "procedural"
    for arg, group, name in (("traffic_density", "environment", "traffic_density"), ("horizon", "environment", "horizon"),
                             ("seed_start", "scenario", "start_seed"), ("num_scenarios", "scenario", "num_scenarios"),
                             ("traffic_seed", "seeds", "traffic_seed"), ("reward_mode", "environment", "reward_mode"),
                             ("cost_mode", "safety", "cost_mode")):
        if getattr(args, arg) is not None:
            c[group][name] = getattr(args, arg)
    if args.traffic_config:
        c = merge_config(c, {"traffic": json.loads(Path(args.traffic_config).read_text())})
    validate_config(c)
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    return c


def summarize(rows):
    result = {"episodes": len(rows)}
    for name, field in (("success_rate", "success"), ("collision_rate", "collision"), ("out_of_road_rate", "out_of_road"),
                        ("mean_episode_cost", "episode_risk_sum"), ("mean_risk_exposure", "risk_exposure"),
                        ("mean_episode_length", "episode_length"), ("mean_reward", "episode_reward"),
                        ("mean_route_completion", "route_completion"), ("mean_speed", "mean_speed")):
        result[name] = float(np.mean([r[field] for r in rows]))
    result["min_ttc_distribution"] = [r["min_ttc"] for r in rows]
    result["mean_step_risk"] = sum(r["episode_risk_sum"] for r in rows) / sum(r["episode_length"] for r in rows)
    return result


def run(args):
    c = resolve_config(args)
    args.output.mkdir(parents=True, exist_ok=False)
    versions = {p: importlib.metadata.version(p) for p in ("metadrive-simulator", "stable-baselines3", "gymnasium", "numpy")}
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "env").glob("*.py")}
    metadata = {"config": c, "versions": versions, "source_sha256": sources, "policy": args.policy,
                "model": args.model, "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest() if args.model else None,
                "episodes_requested": args.episodes}
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2))
    model = None
    if args.policy == "sac":
        if not args.model:
            raise ValueError("SAC evaluation requires --model")
        from stable_baselines3 import SAC
        model = SAC.load(args.model, device=args.device)
        metadata["policy_device"] = str(model.device)
        (args.output / "config.json").write_text(json.dumps(metadata, indent=2))
    env = SafeMetaDriveEnv(research_config=c)
    rows, step_rows = [], []
    try:
        print("observation_space:", env.observation_space, "action_space:", env.action_space, flush=True)
        for episode in range(args.episodes):
            seed = c["scenario"]["start_seed"] + episode % c["scenario"]["num_scenarios"]
            obs, _ = env.reset(seed=seed)
            env.action_space.seed(int(np.random.SeedSequence([c["seeds"]["evaluation_seed"], seed]).generate_state(1)[0]))
            policy = None
            if args.policy == "idm":
                from metadrive.policy.idm_policy import IDMPolicy
                policy = IDMPolicy(env.agent, c["seeds"]["evaluation_seed"])
            try:
                while True:
                    action = policy.act() if policy else model.predict(obs, deterministic=True)[0] if model else env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    if args.steps_csv:
                        keys = ("scene_seed", "dt", "cost", "binary_cost", "continuous_cost", "vehicle_risk", "road_risk", "min_ttc", "min_distance",
                                "min_predicted_distance", "time_to_closest_approach", "collision", "out_of_road", "critical_vehicle_id",
                                "critical_vehicle_distance", "critical_vehicle_ttc", "critical_vehicle_risk", "road_clearance", "active_traffic_count")
                        step_rows.append({"episode": episode, "step": env._research_steps, **{k: info[k] for k in keys}})
                    if terminated or truncated:
                        rows.append({"episode": episode, **info["episode_safety"]})
                        break
            finally:
                if policy:
                    policy.destroy()
            with (args.output / "episodes.jsonl").open("a") as f:
                f.write(json.dumps(rows[-1], allow_nan=False) + "\n")
            print(f"episode {episode+1}/{args.episodes}: steps={rows[-1]['episode_length']} success={rows[-1]['success']} cost={rows[-1]['episode_risk_sum']:.3f}", flush=True)
    finally:
        env.close()
    for filename, data in (("episodes.csv", rows), ("steps.csv", step_rows)):
        if data:
            with (args.output / filename).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    summary = summarize(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2), flush=True)
    return args.output


if __name__ == "__main__":
    run(arguments())
