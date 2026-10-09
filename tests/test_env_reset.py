import unittest
from env.safe_metadrive_env import SafeMetaDriveEnv
from research_helpers import config


class ResetTests(unittest.TestCase):
    def test_reset(self):
        env = SafeMetaDriveEnv(research_config=config())
        try:
            obs, info = env.reset(seed=0)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertEqual(env.action_space.shape, (2,))
            self.assertEqual(info["scene_seed"], 0)
            self.assertAlmostEqual(info["dt"], .1)
        finally:
            env.close()
