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
    "srairl": TaskDescriptor("srairl", "SR-AIRL", "available", "Two learned policy actors (ELCV/RFV) can train from explicitly assigned NGSIM demonstrations and run either on NGSIM replay or generated MetaDrive straight/curve/merge maps with multiple IDM traffic vehicles. Background traffic is not an additional learned SR-AIRL policy; procedural-map training uses NGSIM as the expert transition source."),
    "reward_guide": TaskDescriptor("reward_guide", "SR-AIRL Reward Guide", "available", "Upstream diffusion reward-guide training initialized from a selected local SR-AIRL discriminator. The pinned NGSIM pair subset has 203 records: 150 training and 53 used for checkpoint selection; it has no independent test split."),
    "predictor": TaskDescriptor("predictor", "SR-AIRL Response Predictor", "available", "Upstream response-predictor diffusion training initialized from a selected reward-guide checkpoint. It uses the same 150/53 pinned NGSIM pair split; the 53 records select checkpoints and are not an independent test set."),
    "planner": TaskDescriptor("planner", "Diffusion Planner", "available", "Upstream goal predictor and DPPO diffusion planner train sequentially on the pinned NGSIM pair subset: 150 training and 53 checkpoint-selection records, with no independent test split."),
    "evaluation": TaskDescriptor("evaluation", "Interaction Evaluation", "partial", "Shared evaluation metrics are being added; SR-AIRL candidate pipeline is not yet integrated."),
}


def get_task(key):
    try:
        return TASK_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"unknown task: {key}") from exc
