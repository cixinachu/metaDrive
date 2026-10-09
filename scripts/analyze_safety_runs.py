"""Compare task/safety outcomes, and retain controlled sweeps beside observations."""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from diagnose_safety_cost import plot_diagnostics
from evaluate_env import ROOT
from env.research_config import load_config
from env.risk_utils import compute_ttc_risk, compute_distance_risk

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--random", type=Path, required=True)
    p.add_argument("--rule", type=Path, required=True)
    args = p.parse_args()
    result = {}
    for name, path in (("random", args.random), ("idm", args.rule)):
        summary = json.loads((path / "summary.json").read_text())
        steps = pd.read_csv(path / "steps.csv")
        result[name] = {**summary, "steps_with_active_traffic": int((steps.active_traffic_count > 0).sum()),
                        "max_active_traffic": int(steps.active_traffic_count.max()),
                        "cost_min": float(steps.cost.min()), "cost_max": float(steps.cost.max()),
                        "finite_ttc_steps": int(steps.min_ttc.notna().sum())}
        assert steps.cost.between(0, 1).all()
        assert (steps.loc[steps.collision | steps.out_of_road, "cost"] == 1).all()
        episodes = pd.read_csv(path / "episodes.csv").set_index("episode")
        totals = steps.groupby("episode").cost.sum()
        exposures = (steps.cost * steps.dt).groupby(steps.episode).sum()
        np.testing.assert_allclose(totals, episodes.loc[totals.index, "episode_risk_sum"], rtol=1e-10)
        np.testing.assert_allclose(exposures, episodes.loc[exposures.index, "risk_exposure"], rtol=1e-10)
        collision_any = steps.groupby("episode").collision.any()
        assert (collision_any == episodes.loc[collision_any.index, "collision"]).all()
        result[name]["step_episode_consistency_passed"] = True
        plot_diagnostics(path)
    c = load_config()["safety"]
    pd.DataFrame({"ttc": np.linspace(0, 6, 121), "ttc_risk": [compute_ttc_risk(t, c["ttc_threshold"]) for t in np.linspace(0, 6, 121)]}).to_csv(ROOT / "outputs/controlled_ttc_sweep.csv", index=False)
    distances = np.linspace(0, 30, 121)
    pd.DataFrame({"distance": distances, "equal_speed_10m_s_risk": [compute_distance_risk(d, 10, 10, c) for d in distances]}).to_csv(ROOT / "outputs/controlled_distance_sweep.csv", index=False)
    (ROOT / "outputs/baseline_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
