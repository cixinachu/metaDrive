"""Task-level routing; SafeRL algorithms remain in rl.algorithms.REGISTRY."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskDescriptor:
    key: str
    label: str
    status: str
    description: str


TASK_REGISTRY = {
    "safe_rl": TaskDescriptor("safe_rl", "Safe RL", "available", "Single-ego CMDP; select an algorithm from rl.algorithms.REGISTRY."),
    "srairl": TaskDescriptor("srairl", "SR-AIRL", "available", "Studio dispatches dual-policy Social-AIRL training/evaluation in a MetaDrive NGSIM world using explicit role IDs, legacy kinematic actor updates and recorded background traffic."),
    "planner": TaskDescriptor("planner", "Diffusion Planner", "not_integrated", "Planner code has not been integrated into this repository."),
    "predictor": TaskDescriptor("predictor", "Response Predictor", "not_integrated", "Predictor code has not been integrated into this repository."),
    "evaluation": TaskDescriptor("evaluation", "Interaction Evaluation", "partial", "Shared evaluation metrics are being added; SR-AIRL candidate pipeline is not yet integrated."),
}


def get_task(key):
    try:
        return TASK_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"unknown task: {key}") from exc
