"""Explicit initial-state contract for a procedural SR-AIRL lane-change game."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SRAIRLInteractionScenario:
    source_lane: int = 0
    target_lane: int = 1
    elcv_x: float = 30.
    elcv_speed: float = 20.
    rfv_x: float = 15.
    rfv_speed: float = 19.
    rlv_current_x: float = 65.
    rlv_current_speed: float = 18.
    rlv_target_x: float = 55.
    rlv_target_speed: float = 18.
    interaction_start_x: float = 40.
    interaction_end_x: float = 120.
    success_lateral_tolerance_m: float = .4
    success_stable_steps: int = 10

    @property
    def target_gap_m(self):
        return float(self.rlv_target_x-self.rfv_x)

    @classmethod
    def from_ngsim_initial_state(cls,state):
        """Map a validated straight-road NGSIM gap sample to a two-lane task."""
        elcv_x=30.
        source_lane,target_lane=(0,1) if int(state.get('target_lane',1))>int(state.get('source_lane',0)) else (1,0)
        return cls(source_lane=source_lane,target_lane=target_lane,elcv_x=elcv_x,
            elcv_speed=float(state['elcv_speed_mps']),
            rfv_x=elcv_x-float(state['elcv_rfv_gap_m']),
            rfv_speed=float(state['rfv_speed_mps']),
            rlv_current_x=elcv_x+float(state['rlv_current_gap_m']),
            rlv_current_speed=float(state['rlv_current_speed_mps']),
            rlv_target_x=elcv_x+float(state['rlv_target_gap_m']),
            rlv_target_speed=float(state['rlv_target_speed_mps']))

    def validate(self, lane_num):
        numeric=[self.elcv_x,self.elcv_speed,self.rfv_x,self.rfv_speed,self.rlv_current_x,
                 self.rlv_current_speed,self.rlv_target_x,self.rlv_target_speed,
                 self.interaction_start_x,self.interaction_end_x,self.success_lateral_tolerance_m]
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError('interaction template values must be finite')
        if lane_num < 2 or self.source_lane == self.target_lane:
            raise ValueError('lane-change game requires distinct lanes and at least two lanes')
        if abs(self.source_lane-self.target_lane)!=1:
            raise ValueError('target lane must be adjacent to the source lane')
        if not (0 <= self.source_lane < lane_num and 0 <= self.target_lane < lane_num):
            raise ValueError('source/target lane index is outside the generated road')
        if not self.elcv_x < self.rlv_current_x:
            raise ValueError('RLV_current must be ahead of ELCV in the source lane')
        if not self.rfv_x < self.elcv_x < self.rlv_target_x:
            raise ValueError('ELCV must start inside the RFV/RLV_target gap')
        if self.rlv_target_x-self.rfv_x<=0:
            raise ValueError('target-lane gap must have positive length')
        if self.rlv_current_x<=self.elcv_x or min(self.elcv_speed,self.rfv_speed,self.rlv_current_speed,self.rlv_target_speed)<=0:
            raise ValueError('all four actors require positive speeds and RLV_current must be ahead of ELCV')
        if self.interaction_start_x >= self.interaction_end_x:
            raise ValueError('interaction zone must have positive length')
        if self.success_stable_steps < 1 or self.success_lateral_tolerance_m <= 0:
            raise ValueError('lane-change completion criteria must be positive')
        return self
