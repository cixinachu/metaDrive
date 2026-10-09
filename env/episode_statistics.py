"""Undiscounted episode metrics. Collision events are contact-onset transitions."""
import numpy as np
from .safety_cost import minimum


class EpisodeStatistics:
    def __init__(self, dt, critical_threshold):
        self.dt, self.threshold = dt, critical_threshold
        self.steps = self.cost = self.reward = self.speed = 0
        self.unsafe_steps = self.critical_steps = self.collision_events = 0
        self.collision_any = self.out_any = self.previous_collision = False
        self.first_unsafe = self.first_critical = None
        self.ttcs, self.distances, self.predicted = [], [], []

    def update(self, info, reward, speed):
        self.steps += 1
        self.cost += info["cost"]
        self.reward += reward
        self.speed += speed
        self.unsafe_steps += int(info["binary_cost"] > 0)
        critical = info["continuous_cost"] >= self.threshold
        self.critical_steps += int(critical)
        self.collision_events += int(info["collision"] and not self.previous_collision)
        self.previous_collision = info["collision"]
        self.collision_any |= info["collision"]
        self.out_any |= info["out_of_road"]
        if info["binary_cost"] and self.first_unsafe is None:
            self.first_unsafe = self.steps * self.dt
        if critical and self.first_critical is None:
            self.first_critical = self.steps * self.dt
        for key, target in (("min_ttc", self.ttcs), ("min_distance", self.distances), ("min_predicted_distance", self.predicted)):
            if info[key] is not None:
                target.append(info[key])

    def result(self, info):
        return {"episode_reward": self.reward, "episode_risk_sum": self.cost,
                "risk_exposure": self.cost * self.dt, "mean_step_risk": self.cost / max(self.steps, 1),
                "binary_unsafe_steps": self.unsafe_steps, "collision_events": self.collision_events,
                "collision": self.collision_any, "out_of_road": self.out_any,
                "min_ttc": minimum(self.ttcs), "ttc_p05": float(np.percentile(self.ttcs, 5)) if self.ttcs else None,
                "min_distance": minimum(self.distances), "min_predicted_distance": minimum(self.predicted),
                "time_to_first_unsafe_event": self.first_unsafe, "time_to_first_critical_event": self.first_critical,
                "safety_critical_steps": self.critical_steps, "episode_length": self.steps,
                "success": bool(info.get("arrive_dest", False)), "route_completion": float(info.get("route_completion", 0)),
                "mean_speed": self.speed / max(self.steps, 1), "travel_time": self.steps * self.dt}
