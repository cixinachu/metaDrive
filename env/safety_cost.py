"""Safety aggregation, independent of reward and RL discounting."""
from .risk_utils import bounded, compute_pairwise_risk
import math


def minimum(values):
    values = [v for v in values if v is not None]
    return min(values) if values else None


def compute_safety_cost(ego, others, road_clearance, collision, out_of_road, config):
    ego.validate()
    if road_clearance is not None and not math.isfinite(road_clearance):
        raise ValueError("Non-finite road clearance")
    pairs = [compute_pairwise_risk(ego, other, config) for other in others]
    critical = max(pairs, key=lambda x: (x["risk"], -x["distance"]), default=None)
    vehicle = critical["risk"] if critical else 0.0
    road = bounded(max(0.0, 1 - max(road_clearance, 0.0) / config["road_margin"]) ** 2) if road_clearance is not None else 0.0
    if out_of_road:
        road = 1.0
    binary = float(bool(collision or out_of_road))
    continuous = max(vehicle, road, binary)
    return {
        "cost": bounded(binary if config["cost_mode"] == "binary" else continuous),
        "continuous_cost": continuous, "binary_cost": binary, "vehicle_risk": vehicle, "road_risk": road,
        "min_ttc": minimum(p["ttc"] for p in pairs), "min_distance": minimum(p["distance"] for p in pairs),
        "min_predicted_distance": minimum(p["predicted_distance"] for p in pairs),
        "time_to_closest_approach": critical["time_to_closest_approach"] if critical else None,
        "collision": bool(collision), "out_of_road": bool(out_of_road),
        "critical_vehicle_id": critical["id"] if critical else None,
        "critical_vehicle_distance": critical["distance"] if critical else None,
        "critical_vehicle_ttc": critical["ttc"] if critical else None,
        "critical_vehicle_risk": critical["risk"] if critical else None,
        "road_clearance": road_clearance, "active_traffic_count": len(others),
    }
