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
    "srairl": TaskDescriptor("srairl", "SR-AIRL", "available", "Two learned policy actors (ELCV/RFV) train from accepted MetaDrive expert demonstrations and interact in MetaDrive-generated straight, curve and ramp-merge scenes with optional IDM traffic."),
    "reward_guide": TaskDescriptor("reward_guide", "SR-AIRL Reward Guide", "available", "Upstream diffusion reward-guide training uses the selected episode-grouped MetaDrive expert dataset and a compatible local SR-AIRL discriminator."),
    "predictor": TaskDescriptor("predictor", "SR-AIRL Response Predictor", "available", "Upstream response-predictor diffusion training uses the selected episode-grouped MetaDrive expert dataset and a compatible reward-guide checkpoint."),
    "planner": TaskDescriptor("planner", "Diffusion Planner", "available", "Goal predictor and DPPO diffusion planner train sequentially on the selected episode-grouped MetaDrive expert dataset."),
    "evaluation": TaskDescriptor("evaluation", "Interaction Evaluation", "partial", "A versioned CandidateBundle, Planner/Reward-Guided Predictor inference, SR-AIRL reward selector, dual Tracker, calibrated-domain filter and 0.5-second receding-horizon MetaDrive task are implemented. It remains partial until compatible trained component checkpoints exist and full episodes pass on straight, curve and merge maps."),
}


def get_task(key):
    try:
        return TASK_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"unknown task: {key}") from exc
