"""Interface-only test; no SAC model or training is run here."""
import unittest
from env.research_config import load_config
from rl.vec_env import build_env


class TrainingInterfaceTests(unittest.TestCase):
    def test_rl_seed_does_not_select_training_map(self):
        c = load_config("configs/env/train.yaml", {"environment": {"map": "S"}})
        env = build_env(c)
        try:
            env.seed(0)
            obs = env.reset()
            self.assertEqual(obs.shape, (1,259))
            scene_seed = env.envs[0].unwrapped.current_seed
            self.assertTrue(1000 <= scene_seed < 2000)
        finally:
            env.close()
