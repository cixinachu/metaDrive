"""WCSAC-IQN: quantile safety critic and sampled upper-tail CVaR."""
from rl.constrained_sac import ConstrainedSAC
from rl.common import constrained_options


class WCSACIQN(ConstrainedSAC):
    safety_kind = 'iqn'


def create_model(settings, env):
    return WCSACIQN('MlpPolicy', env, **constrained_options(settings))
