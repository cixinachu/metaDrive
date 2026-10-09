import unittest
from env.safe_metadrive_env import SafeMetaDriveEnv
from research_helpers import config


class RewardModeTests(unittest.TestCase):
    def test_collision_does_not_replace_cmdp_progress(self):
        for mode in ("cmdp", "metadrive"):
            env = SafeMetaDriveEnv(research_config=config(reward_mode=mode, horizon=50))
            try:
                env.reset(seed=0)
                for _ in range(10):
                    env.step([0, .5])
                env.agent.crash_vehicle = True
                reward, info = env.reward_function(env.agent.id if env.agent.id in env.agents else "default_agent")
                if mode == "cmdp":
                    self.assertEqual(reward, info["step_reward"])
                    self.assertGreater(reward, 0)
                else:
                    self.assertEqual(reward, -env.config["crash_vehicle_penalty"])
                done, _ = env.done_function("default_agent")
                self.assertFalse(done)
            finally:
                env.close()
