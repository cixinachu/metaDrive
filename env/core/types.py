"""Canonical typed descriptions shared by interaction tasks and datasets."""
from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class InteractionScenario:
    """One NGSIM interaction window with explicit, auditable vehicle roles.

    `tracks` uses the importer schema: each state is [t, x, y, vx, vy].
    Role IDs are never guessed from arbitrary clip order.
    """
    scenario_id: str
    elcv_id: str
    rfv_id: str
    tracks: Mapping[str, dict]
    lane_num: int
    lane_width: float
    road_length: float
    split: str
    source: str
    rlv_current_id: Optional[str] = None
    rlv_target_id: Optional[str] = None
    lane_change_start: Optional[float] = None
    lane_change_end: Optional[float] = None
    start_time: float = 0.0

    def __post_init__(self):
        if not self.scenario_id or not self.source:
            raise ValueError("scenario_id and source are required")
        if self.split not in {"train", "validation", "test", "eval"}:
            raise ValueError("split must be train, validation, test, or eval")
        required = [self.elcv_id, self.rfv_id]
        if self.rlv_current_id:
            required.append(self.rlv_current_id)
        if self.rlv_target_id:
            required.append(self.rlv_target_id)
        missing = sorted(set(required) - set(self.tracks))
        if missing:
            raise ValueError(f"role tracks missing from scenario: {missing}")
        if len(set(required)) != len(required):
            raise ValueError("controlled and leader roles must have distinct track IDs")
        if self.lane_num < 1 or self.lane_width <= 0 or self.road_length <= 0:
            raise ValueError("invalid road geometry")
        if (self.lane_change_start is None) != (self.lane_change_end is None):
            raise ValueError("lane-change start and end must be supplied together")
        if self.lane_change_start is not None and self.lane_change_end < self.lane_change_start:
            raise ValueError("lane-change end precedes start")

    @classmethod
    def from_clip(cls, clip, manifest, *, elcv_id, rfv_id, rlv_current_id=None,
                  rlv_target_id=None, scenario_id=None, lane_change_start=None,
                  lane_change_end=None):
        """Bind explicit role IDs to a validated importer clip and its manifest."""
        tracks = {str(t["id"]): t for t in clip.get("tracks", [])}
        return cls(
            scenario_id=scenario_id or manifest["id"], elcv_id=str(elcv_id),
            rfv_id=str(rfv_id), rlv_current_id=None if rlv_current_id is None else str(rlv_current_id),
            rlv_target_id=None if rlv_target_id is None else str(rlv_target_id), tracks=tracks,
            lane_num=int(clip["lane_num"]), lane_width=float(clip["lane_width"]),
            road_length=float(clip["road_length"]), split=manifest["split"],
            source=manifest["source"], lane_change_start=lane_change_start,
            lane_change_end=lane_change_end,
        )
