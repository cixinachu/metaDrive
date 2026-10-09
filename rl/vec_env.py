"""Research vector environment: RL RNG must not select training-map seeds."""
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env.safe_metadrive_env import SafeMetaDriveEnv


class ResearchVecEnv(DummyVecEnv):
    def seed(self, seed=None):
        self._seeds = [None] * self.num_envs
        return self._seeds


def build_env(research_config):
    return ResearchVecEnv([lambda: Monitor(SafeMetaDriveEnv(research_config=research_config))])
