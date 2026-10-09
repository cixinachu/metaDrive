"""Gaussian WCSAC: mean/variance safety critic with upper-tail CVaR."""
from rl.constrained_sac import ConstrainedSAC
from rl.common import constrained_options


class WCSAC(ConstrainedSAC):
    safety_kind = 'gaussian'


def create_model(settings, env):
    return WCSAC('MlpPolicy', env, **constrained_options(settings))
