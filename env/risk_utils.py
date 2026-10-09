"""Pure safety geometry; metres, seconds, metres/second throughout."""
from dataclasses import dataclass
import math
import numpy as np


def bounded(value):
    if not math.isfinite(value):
        raise ValueError("Non-finite safety signal")
    return float(np.clip(value, 0.0, 1.0))


def compute_ttc_risk(ttc, threshold):
    if ttc is None:
        return 0.0
    if not math.isfinite(ttc) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid TTC or threshold")
    return bounded(max(0.0, 1.0 - max(ttc, 0.0) / threshold) ** 2)


def compute_distance_risk(distance, ego_speed, other_speed, config):
    if not np.isfinite([distance, ego_speed, other_speed]).all():
        raise ValueError("Non-finite distance or speed")
    safe = config["min_distance"] + config["time_headway"] * max(ego_speed, 0.0)
    safe += max(ego_speed - other_speed, 0.0) ** 2 / (2 * config["comfortable_deceleration"])
    return bounded(max(0.0, 1.0 - max(distance, 0.0) / max(safe, config["epsilon"])) ** 2)


@dataclass(frozen=True)
class VehicleState:
    identifier: str
    position: tuple
    velocity: tuple
    heading: float
    length: float
    width: float

    def validate(self):
        if not np.isfinite([*self.position, *self.velocity, self.heading, self.length, self.width]).all():
            raise ValueError("Non-finite vehicle state")
        if self.length <= 0 or self.width <= 0:
            raise ValueError("Invalid vehicle dimensions")


def axes(v):
    forward = np.array([math.cos(v.heading), math.sin(v.heading)])
    return forward, np.array([-forward[1], forward[0]])


def support(v, axis):
    f, side = axes(v)
    return abs(float(f @ axis)) * v.length / 2 + abs(float(side @ axis)) * v.width / 2


def corners(v, position=None):
    f, side = axes(v)
    p = np.asarray(v.position if position is None else position)
    return np.array([p + x * v.length / 2 * f + y * v.width / 2 * side for x, y in ((1,1), (1,-1), (-1,-1), (-1,1))])


def rectangle_distance(a, b, tau=0.0):
    """Exact clearance of constant-heading oriented rectangles at time tau."""
    pa = np.asarray(a.position) + tau * np.asarray(a.velocity)
    pb = np.asarray(b.position) + tau * np.asarray(b.velocity)
    ca, cb = corners(a, pa), corners(b, pb)
    if all(abs(float((pb-pa) @ axis)) <= support(a, axis) + support(b, axis) for axis in (*axes(a), *axes(b))):
        return 0.0
    distances = []
    for points, edges in ((ca, cb), (cb, ca)):
        delta = np.roll(edges, -1, axis=0) - edges
        offset = points[:, None, :] - edges[None, :, :]
        t = np.clip(np.sum(offset * delta, axis=-1) / np.sum(delta * delta, axis=-1), 0, 1)
        distances.append(float(np.min(np.linalg.norm(offset - t[..., None] * delta, axis=-1))))
    return min(distances)


def compute_closest_approach_risk(ego, other, config):
    relative = np.asarray(other.position) - ego.position
    velocity = np.asarray(other.velocity) - ego.velocity
    tau = float(np.clip(-float(relative @ velocity) / (float(velocity @ velocity) + config["epsilon"]), 0, config["prediction_horizon"]))
    distance = rectangle_distance(ego, other, tau)
    spatial = max(0.0, 1 - distance / config["encounter_margin"]) ** 2
    temporal = max(0.0, 1 - tau / config["prediction_horizon"]) ** 2
    return bounded(spatial * temporal), tau, distance


def compute_pairwise_risk(ego, other, config):
    ego.validate()
    other.validate()
    forward, side = axes(ego)
    relative = np.asarray(other.position) - ego.position
    ev, ov = np.asarray(ego.velocity), np.asarray(other.velocity)
    longitudinal = float(relative @ forward)
    lateral_overlap = abs(float(relative @ side)) < support(ego, side) + support(other, side)
    following = longitudinal > 0 and lateral_overlap and float(forward @ axes(other)[0]) >= config["same_direction_cosine"]
    gap = max(0.0, longitudinal - support(ego, forward) - support(other, forward))
    closing = float((ev - ov) @ forward)
    ttc = None
    ttc_risk = distance_risk = 0.0
    if following:
        if gap <= 0:
            ttc = 0.0
        elif closing > config["epsilon"]:
            ttc = gap / closing
        ttc_risk = compute_ttc_risk(ttc, config["ttc_threshold"])
        distance_risk = compute_distance_risk(gap, float(ev @ forward), float(ov @ forward), config)
    encounter, tau, predicted = compute_closest_approach_risk(ego, other, config)
    return {"id": other.identifier, "risk": max(ttc_risk, distance_risk, encounter), "ttc": ttc,
            "distance": rectangle_distance(ego, other), "predicted_distance": predicted,
            "time_to_closest_approach": tau, "closing_speed": closing,
            "ttc_risk": ttc_risk, "distance_risk": distance_risk, "encounter_risk": encounter}
