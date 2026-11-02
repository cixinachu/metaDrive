"""Measure desired-to-actual acceleration mapping in the MetaDrive SR-AIRL world."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.adapters.srairl_action import SRAIRLActionAdapter
from env.tasks.srairl_procedural_env import SRAIRLProceduralEnv


SPEEDS = (5., 10., 15., 20., 25., 30.)
TARGETS = (-1., 0., 1.)
CONTROLLED_ROLE = "elcv"
SETTLING_STEPS = 5
DT = 0.1


def measure(env, axis, speed, desired):
    env.reset()
    vehicle = env.world.role_vehicle(CONTROLLED_ROLE)
    lane = vehicle.lane
    longitudinal, _ = lane.local_coordinates(vehicle.position[:2])
    heading = float(lane.heading_theta_at(longitudinal))
    tangent = np.asarray([np.cos(heading), np.sin(heading)])
    normal = np.asarray([-tangent[1], tangent[0]])
    vehicle.set_heading_theta(heading)
    vehicle.set_velocity(tangent * speed)
    initial_velocity = np.asarray(vehicle.velocity[:2], dtype=float)
    action = [desired, 0.] if axis == "longitudinal" else [0., desired]
    for _ in range(SETTLING_STEPS):
        env.step({"elcv": action, "rfv": [0., 0.]})
    final_velocity = np.asarray(vehicle.velocity[:2], dtype=float)
    basis = tangent if axis == "longitudinal" else normal
    actual = float((final_velocity - initial_velocity) @ basis / (SETTLING_STEPS * DT))
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/calibration/srairl_action_adapter"))
    parser.add_argument("--seed", type=int, default=2718)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = SRAIRLProceduralEnv("straight", lane_num=2, traffic_density=0., seed=args.seed,
                              horizon=SETTLING_STEPS + 2, traffic_seed=args.seed + 1_000_003)
    rows = []
    try:
        for speed in SPEEDS:
            for axis in ("longitudinal", "lateral"):
                for desired in TARGETS:
                    actual = measure(env, axis, speed, desired)
                    rows.append({"initial_speed_mps": speed, "axis": axis,
                                 "desired_accel_mps2": desired, "actual_accel_mps2": actual,
                                 "error_mps2": actual - desired,
                                 "absolute_error_mps2": abs(actual - desired)})
    finally:
        env.close()

    csv_path = args.output / "measurements.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for column, axis_name in enumerate(("longitudinal", "lateral")):
        for desired in TARGETS:
            selected = [row for row in rows if row["axis"] == axis_name and
                        row["desired_accel_mps2"] == desired]
            axes[0, column].plot([r["initial_speed_mps"] for r in selected],
                [r["actual_accel_mps2"] for r in selected], marker="o", label=f"desired {desired:+g}")
            axes[1, column].plot([r["initial_speed_mps"] for r in selected],
                [r["absolute_error_mps2"] for r in selected], marker="o", label=f"desired {desired:+g}")
        axes[0, column].axhline(0., color="#64748b", linewidth=.8)
        axes[0, column].set_title(f"{axis_name.title()} acceleration")
        axes[0, column].set_ylabel("Actual acceleration (m/s²)")
        axes[1, column].set_ylabel("Absolute error (m/s²)")
        axes[1, column].set_xlabel("Initial speed (m/s)")
        axes[0, column].grid(alpha=.25); axes[1, column].grid(alpha=.25)
        axes[0, column].legend(); axes[1, column].legend()
    fig.suptitle("SR-AIRL MetaDrive action adapter calibration")
    fig.tight_layout()
    plot_path = args.output / "error_curves.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    by_axis = {}
    for axis_name in ("longitudinal", "lateral"):
        selected = [row["absolute_error_mps2"] for row in rows if row["axis"] == axis_name]
        by_axis[axis_name] = {"mae_mps2": float(np.mean(selected)),
                              "max_abs_error_mps2": float(np.max(selected))}
    try:
        version = importlib.metadata.version("metadrive-simulator")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    summary = {"simulator": "MetaDrive", "simulator_version": version, "seed": args.seed,
        "map": "straight", "lane_num": 2, "traffic_density": 0., "dt_s": DT,
        "settling_steps": SETTLING_STEPS, "speeds_mps": list(SPEEDS),
        "desired_accelerations_mps2": list(TARGETS), "vehicle_role": CONTROLLED_ROLE,
        "adapter_config": SRAIRLActionAdapter.from_yaml().__dict__,
        "equivalence_status": "not_validated",
        "decision": "Do not use this mapping for claims of upstream-equivalent MetaDrive physics or closed-loop candidate evaluation.",
        "metrics": by_axis, "measurements_csv": csv_path.name, "error_plot": plot_path.name}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
