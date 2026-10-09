"""SAC-Lagrangian: expected discounted cost and an adaptive multiplier."""
from rl.constrained_sac import ConstrainedSAC
from rl.common import constrained_options


class SACLagrangian(ConstrainedSAC):
    safety_kind = 'mean'


def create_model(settings, env):
    return SACLagrangian('MlpPolicy', env, **constrained_options(settings))
