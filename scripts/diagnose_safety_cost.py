"""Plot observed safety signals and analytical controlled risk sweeps."""
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluate_env import arguments, run, ROOT
from env.research_config import load_config
from env.risk_utils import compute_ttc_risk, compute_distance_risk


def plot_diagnostics(directory):
    data = pd.read_csv(directory / "steps.csv")
    c = load_config()["safety"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(data.min_ttc, data.cost, s=3, alpha=.4, label="observed total cost")
    ttc = np.linspace(0, 6, 200)
    axes[0].plot(ttc, [compute_ttc_risk(t, c["ttc_threshold"]) for t in ttc], color="red", label="isolated TTC risk")
    axes[0].set(xlabel="TTC (s)", ylabel="Cost / risk")
    axes[0].set_xlim(0, c["ttc_threshold"] * 2)
    axes[0].legend()
    axes[1].scatter(data.min_distance, data.cost, s=3, alpha=.4)
    axes[1].set(xlabel="Vehicle rectangle clearance (m)", ylabel="Total cost")
    for episode, group in data.groupby("episode"):
        if episode < 5:
            axes[2].plot(group.step * group.dt, group.cost, label=str(episode))
    axes[2].set(xlabel="Episode time (s)", ylabel="Cost")
    axes[2].legend(title="Episode")
    fig.tight_layout()
    fig.savefig(directory / "cost_diagnostics.png", dpi=160)
    plt.close(fig)
    shutil.copyfile(directory / "steps.csv", ROOT / "outputs/cost_diagnostics.csv")


if __name__ == "__main__":
    args = arguments("idm")
    args.steps_csv = True
    plot_diagnostics(run(args))
