"""Local SAC entry: uses the original SB3 SAC optimization."""
from stable_baselines3 import SAC as SB3SAC
from rl.common import sac_options


class SAC(SB3SAC):
    """Reward-only SAC; optimization is inherited without modifications."""


def create_model(settings, env):
    return SAC('MlpPolicy', env, **sac_options(settings))
