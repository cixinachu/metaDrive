"""Check ray sampling convergence while a rule policy drives each road type."""
import json
import numpy as np
from evaluate_env import ROOT
from env.research_config import load_config
from env.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy

if __name__ == "__main__":
    rows = []
    for scenario in ("straight", "merge", "intersection", "t_junction", "roundabout"):
        c = load_config(ROOT / f"configs/scenarios/{scenario}.yaml")
        env = SafeMetaDriveEnv(research_config=c)
        try:
            env.reset(seed=0)
            policy = IDMPolicy(env.agent, c["seeds"]["evaluation_seed"])
            try:
                for step in range(300):
                    _, _, terminated, truncated, _ = env.step(policy.act())
                    if step % 10 == 0:
                        coarse = env.road_clearance()
                        env.research_config["safety"]["road_sensor_rays"] = 720
                        fine = env.road_clearance()
                        env.research_config["safety"]["road_sensor_rays"] = c["safety"]["road_sensor_rays"]
                        rows.append({"scenario": scenario, "step": step, "coarse": coarse, "fine": fine,
                                     "absolute_difference": abs(coarse-fine) if coarse is not None and fine is not None else None})
                    if terminated or truncated:
                        break
            finally:
                policy.destroy()
        finally:
            env.close()
    differences = [r["absolute_difference"] for r in rows if r["absolute_difference"] is not None]
    def risk(margin):
        return max(0, 1-max(margin, 0)/c["safety"]["road_margin"])**2 if margin is not None else 0
    risk_differences = [abs(risk(r["coarse"])-risk(r["fine"])) for r in rows]
    output = {"samples": rows, "max_absolute_difference_m": max(differences),
              "p95_absolute_difference_m": float(np.percentile(differences, 95)),
              "max_road_risk_difference": max(risk_differences),
              "p95_road_risk_difference": float(np.percentile(risk_differences, 95)),
              "scope": "seed 0, 300 IDM decisions per diagnostic map; empirical sampling check, not an exact-boundary guarantee"}
    (ROOT / "outputs/boundary_sampling.json").write_text(json.dumps(output, indent=2))
    print({key: value for key, value in output.items() if key != "samples"})
