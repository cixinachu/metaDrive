"""Versioned, framework-independent contracts for SR-AIRL planning."""
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


@dataclass(frozen=True)
class PlanningContext:
    """Terminal lane target in the same lateral frame as Planner history."""
    target_lateral_position: float
    target_lateral_velocity: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.target_lateral_position) or not np.isfinite(self.target_lateral_velocity):
            raise ValueError("planning context lateral target must be finite")


@dataclass(frozen=True)
class TrajectoryCandidate:
    """Future-only states [y, x, vx, vy] at t+dt ... t+T*dt; actions [ax, ay]."""
    states: np.ndarray
    actions: np.ndarray | None = None
    dt: float = 0.1

    def __post_init__(self):
        states = np.asarray(self.states, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != 4 or states.shape[0] < 1:
            raise ValueError("states must be a non-empty [T,4] array ordered [y,x,vx,vy]")
        if not np.isfinite(states).all():
            raise ValueError("states must contain only finite values")
        object.__setattr__(self, "states", states.copy())
        if self.actions is not None:
            object.__setattr__(self, "actions", _finite_array(self.actions, (len(states), 2), "actions"))
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be a positive finite number")


@dataclass
class JointCandidate:
    candidate_id: int
    elcv: TrajectoryCandidate
    rfv: TrajectoryCandidate
    rlv_current: TrajectoryCandidate
    rlv_target: TrajectoryCandidate
    feasibility: dict[str, Any] = field(default_factory=dict)
    reward_elcv: float = float("nan")
    reward_rfv: float = float("nan")
    joint_return: float = float("nan")

    def __post_init__(self):
        lengths = {len(t.states) for t in (self.elcv, self.rfv, self.rlv_current, self.rlv_target)}
        timesteps = {t.dt for t in (self.elcv, self.rfv, self.rlv_current, self.rlv_target)}
        if len(lengths) != 1 or len(timesteps) != 1:
            raise ValueError("all four vehicle trajectories must share horizon and dt")


@dataclass
class CandidateBundle:
    """Versioned bundle whose candidates begin at t+dt, never at the observation time."""
    schema_version: str
    time_semantics: str
    scenario_id: str
    planning_step: int
    dt: float
    horizon: int
    history: dict[str, np.ndarray]
    candidates: list[JointCandidate]
    planner_checkpoint: str
    reward_guide_checkpoint: str
    predictor_checkpoint: str
    srairl_checkpoint: str
    calibration_hash: str
    normalization_version: str
    selected_candidate_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.schema_version != "srairl-candidate-bundle-v2":
            raise ValueError(f"unsupported CandidateBundle schema: {self.schema_version}")
        if self.time_semantics != "future_only_t_plus_dt_through_t_plus_horizon_dt":
            raise ValueError(f"unsupported CandidateBundle time semantics: {self.time_semantics}")
        if self.horizon < 1 or self.dt <= 0 or not np.isfinite(self.dt):
            raise ValueError("horizon and dt must be positive")
        required = {"elcv", "rfv", "rlv_current", "rlv_target"}
        if set(self.history) != required:
            raise ValueError(f"history must contain exactly {sorted(required)}")
        normalized = {}
        for role, value in self.history.items():
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 4 or not len(arr) or not np.isfinite(arr).all():
                raise ValueError(f"history[{role}] must be finite [N,4] [y,x,vx,vy]")
            normalized[role] = arr.copy()
        self.history = normalized
        if not self.candidates:
            raise ValueError("CandidateBundle must contain at least one joint candidate")
        for candidate in self.candidates:
            if len(candidate.elcv.states) != self.horizon or candidate.elcv.dt != self.dt:
                raise ValueError("candidate horizon/dt does not match bundle")
        if self.selected_candidate_id is not None and self.selected_candidate_id not in {
                candidate.candidate_id for candidate in self.candidates}:
            raise ValueError("selected_candidate_id must identify a candidate in the bundle")
