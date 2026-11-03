"""Shared inference and execution interfaces for the SR-AIRL closed loop."""

from .schema import CandidateBundle, JointCandidate, TrajectoryCandidate

__all__ = ["CandidateBundle", "JointCandidate", "TrajectoryCandidate"]
