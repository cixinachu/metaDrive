"""Task-specific environments, loaded lazily to keep task adapters isolated."""

def __getattr__(name):
    if name == "SafeRLEnv":
        from .safe_rl_env import SafeRLEnv
        return SafeRLEnv
    if name == "SRAIRLEnv":
        from .srairl_env import SRAIRLEnv
        return SRAIRLEnv
    raise AttributeError(name)
