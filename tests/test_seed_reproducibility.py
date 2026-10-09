import unittest
import numpy as np
from env.safe_metadrive_env import SafeMetaDriveEnv
from env.research_config import load_config
from research_helpers import config


class SeedTests(unittest.TestCase):
    def test_complete_merge_rollout_repeats(self):
        from metadrive.policy.idm_policy import IDMPolicy
        c = load_config("configs/scenarios/merge.yaml")
        env = SafeMetaDriveEnv(research_config=c)
        traces = []
        try:
            self.assertTrue(env.config["force_destroy"])
            for _ in range(2):
                env.reset(seed=0)
                policy = IDMPolicy(env.agent, 44)
                trace = []
                try:
                    while True:
                        _, _, terminated, truncated, info = env.step(policy.act())
                        trace.append([*env.agent.position, info["cost"], info["collision"]])
                        if terminated or truncated:
                            break
                finally:
                    policy.destroy()
                traces.append(trace)
            np.testing.assert_allclose(traces[0], traces[1], atol=1e-6)
        finally:
            env.close()

    def test_dynamic_traffic_repeats(self):
        from metadrive.policy.idm_policy import IDMPolicy
        env = SafeMetaDriveEnv(research_config=config(horizon=400))
        traces = []
        try:
            for _ in range(2):
                env.reset(seed=0)
                policy = IDMPolicy(env.agent, 44)
                trace = []
                try:
                    for step in range(180):
                        obs, reward, terminated, truncated, info = env.step(policy.act())
                        trace.append([*env.agent.position, info["cost"], info["active_traffic_count"]])
                        for vehicle in env.engine.traffic_manager.traffic_vehicles:
                            self.assertIsInstance(env.engine.get_policy(vehicle.id), IDMPolicy)
                        if terminated or truncated:
                            break
                finally:
                    policy.destroy()
                traces.append(trace)
            self.assertGreater(max(row[-1] for row in traces[0]), 0)
            np.testing.assert_allclose(traces[0], traces[1], atol=1e-6)
        finally:
            env.close()

    def test_repeat_trajectory(self):
        env = SafeMetaDriveEnv(research_config=config())
        try:
            traces = []
            for _ in range(2):
                obs, _ = env.reset(seed=3)
                trace = [obs.copy()]
                for _ in range(5):
                    obs, reward, _, _, info = env.step([0,.3])
                    trace.append(np.concatenate([obs, [reward, info["cost"]]]))
                traces.append(trace)
            for a, b in zip(*traces):
                np.testing.assert_allclose(a, b, atol=1e-6)
        finally:
            env.close()

    def test_diagnostic_maps(self):
        for name in ("straight", "merge", "intersection", "t_junction", "roundabout"):
            c = load_config(f"configs/scenarios/{name}.yaml")
            env = SafeMetaDriveEnv(research_config=c)
            try:
                obs1, _ = env.reset(seed=0)
                obs2, _ = env.reset(seed=0)
                np.testing.assert_allclose(obs1, obs2)
            finally:
                env.close()
