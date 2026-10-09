import unittest
import numpy as np
from env.safe_metadrive_env import SafeMetaDriveEnv
from research_helpers import config


class StepTests(unittest.TestCase):
    def test_world_velocity_matches_position_change(self):
        env = SafeMetaDriveEnv(research_config=config(traffic_density=0))
        try:
            env.reset(seed=0)
            lane = env.agent.navigation.current_ref_lanes[0]
            env.agent.set_position(lane.position(10, lane.width))
            env.agent.set_heading_theta(np.pi / 4)
            env.agent.set_velocity([1, 1], 5)
            # Settle the artificial teleport/velocity intervention before
            # measuring an ordinary simulator decision interval.
            env.step([0, 0])
            before = np.asarray(tuple(env.agent.position), dtype=float)
            velocity_before = np.asarray(tuple(env.agent.velocity), dtype=float)
            env.step([0, 0])
            displacement = np.asarray(tuple(env.agent.position), dtype=float) - before
            expected = (velocity_before + env.agent.velocity) * env.dt / 2
            self.assertTrue((displacement > 0).all())
            np.testing.assert_allclose(displacement, expected, atol=.03)
        finally:
            env.close()

    def test_binary_mode_and_real_out_of_road(self):
        c = config()
        c["safety"]["cost_mode"] = "binary"
        env = SafeMetaDriveEnv(research_config=c)
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step([0, .1])
            self.assertEqual(info["cost"], info["binary_cost"])
            env.agent.set_position((10000, 10000))
            _, _, terminated, _, info = env.step([0, 0])
            self.assertTrue(info["out_of_road"])
            self.assertTrue(terminated)
            self.assertEqual(info["cost"], 1)
            self.assertEqual(info["episode_safety"]["binary_unsafe_steps"], 1)
        finally:
            env.close()

    def test_episodes(self):
        env = SafeMetaDriveEnv(research_config=config())
        try:
            for seed in range(3):
                env.reset(seed=seed)
                while True:
                    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                    self.assertTrue(env.observation_space.contains(obs))
                    self.assertTrue(0 <= info["cost"] <= 1)
                    if terminated or truncated:
                        self.assertIn("episode_safety", info)
                        if not terminated:
                            self.assertEqual(info["episode_safety"]["episode_length"], 12)
                        break
        finally:
            env.close()
