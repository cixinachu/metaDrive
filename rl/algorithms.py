"""Single algorithm registry for model creation, loading and Studio metadata.

Add a local module and one Algorithm entry; do not add branches to the worker.
See rl/README.md for the model/factory contract.
"""
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Algorithm:
    name: str
    module: str
    class_name: str
    description: str
    uses_cost: bool = False

    @property
    def model_class(self):
        return getattr(import_module(self.module), self.class_name)

    def create(self, settings, env):
        return import_module(self.module).create_model(settings, env)


REGISTRY = {
    'sac': Algorithm('SAC', 'rl.sac', 'SAC', '本地 SAC：优化任务 reward，cost 仅记录。'),
    'sac_lagrangian': Algorithm('SAC-Lagrangian', 'rl.sac_lagrangian', 'SACLagrangian',
        '期望 cost critic + 自适应约束乘子。', True),
    'wcsac': Algorithm('WCSAC', 'rl.wcsac', 'WCSAC', '高斯 safety critic：学习均值与方差，以 CVaR 约束风险。', True),
    'wcsac_iqn': Algorithm('WCSAC-IQN', 'rl.wcsac_iqn', 'WCSACIQN', 'IQN safety critic：学习 cost 分位数分布，估计尾部风险。', True),
}


def metadata():
    return {key:dict(name=entry.name, module=entry.module, description=entry.description, uses_cost=entry.uses_cost)
            for key,entry in REGISTRY.items()}


def create_model(settings, env):
    return REGISTRY[settings['algorithm']].create(settings, env)


def load_model(algorithm, path, device='auto'):
    return REGISTRY[algorithm].model_class.load(path, device=device)


# Backward-compatible import; new code should use REGISTRY or the factory functions.
def __getattr__(name):
    if name == 'ALGORITHMS':
        return {key:entry.model_class for key,entry in REGISTRY.items()}
    raise AttributeError(name)
