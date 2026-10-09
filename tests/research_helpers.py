from env.research_config import load_config


def config(**environment):
    return load_config("configs/env/eval_id.yaml", {"environment": {"map": "S", "horizon": 12, **environment}})
