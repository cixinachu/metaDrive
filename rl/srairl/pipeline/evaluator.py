"""Episode metrics and provenance for SR-AIRL candidate execution."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np


@dataclass
class EpisodeMetrics:
    completion_time_s: float | None = None
    controls: dict[str,list[dict]] = field(default_factory=dict)
    failure_type: str | None = None
    failure_details: list[dict] = field(default_factory=list)
    lane_change_success: bool = False
    time_to_lane_change_s: float | None = None
    collision: bool = False
    out_of_road: bool = False
    minimum_ttc_s: float | None = None
    minimum_distance_m: float | None = None
    route_completion: float = 0.0
    rfv_delta_v_mps: float = 0.0
    target_gap_change_m: float = 0.0
    planning_steps: int = 0
    candidate_count: int = 0
    selected_candidate_ids: list[int] = field(default_factory=list)
    selected_srairl_returns: list[float] = field(default_factory=list)
    candidate_returns: list[dict] = field(default_factory=list)
    planning_latency_s: list[float] = field(default_factory=list)
    predictor_latency_s: list[float] = field(default_factory=list)
    total_decision_latency_s: list[float] = field(default_factory=list)
    tracker_errors: dict[str, list[float]] = field(default_factory=dict)

    def update_environment(self, info):
        if info.get('time') is not None:self.completion_time_s=float(info['time'])
        for role,control in info.get('control_telemetry',{}).items():
            history=self.controls.setdefault(role,[])
            if history and history[-1].get('time_s')==info.get('time'):continue
            measured=np.asarray(control['measured_acceleration'],dtype=float)
            jerk=None if not history else ((measured-np.asarray(history[-1]['measured_acceleration']))/float(info['dt_s'])).tolist()
            history.append({**control,'time_s':info.get('time'),'jerk_mps3':jerk,
                'steering_saturated':abs(control['metadrive_action'][0])>=.99,
                'throttle_saturated':abs(control['metadrive_action'][1])>=.99})
        self.lane_change_success = self.lane_change_success or bool(info.get("lane_change_success", info.get("merge_success", False)))
        if info.get("time_to_lane_change") is not None:
            self.time_to_lane_change_s = float(info["time_to_lane_change"])
        self.collision = self.collision or bool(info.get("metadrive_primary_collision", info.get("collision_proxy_any", False)))
        self.out_of_road = self.out_of_road or bool(info.get("metadrive_out_of_road", False))
        if self.collision:self.failure_type='collision'
        elif self.out_of_road:self.failure_type='out_of_road'
        for key, sources in (("minimum_ttc_s", ("minimum_ttc_s", "min_interaction_ttc_s", "controlled_ttc_s")),
                             ("minimum_distance_m", ("minimum_distance_m", "min_interaction_clearance_m",
                                                      "controlled_min_clearance_m"))):
            value = next((info[source] for source in sources if info.get(source) is not None), None)
            if value is not None and np.isfinite(value):
                old = getattr(self, key)
                setattr(self, key, float(value) if old is None else min(old, float(value)))
        self.route_completion = float(info.get("route_completion", self.route_completion))
        self.rfv_delta_v_mps = float(info.get("delta_v_rfv_mps", self.rfv_delta_v_mps))
        self.target_gap_change_m = float(info.get("target_gap_change_m", self.target_gap_change_m))

    def to_dict(self):
        return asdict(self)


def write_evaluation_record(path, metrics, *, bundle_manifest, environment_metadata):
    record = {"metrics": metrics.to_dict(), "model_bundle": bundle_manifest,
              "environment": environment_metadata}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)
    return target
