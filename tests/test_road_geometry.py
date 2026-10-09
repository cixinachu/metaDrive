import unittest
from env.safe_metadrive_env import SafeMetaDriveEnv
from research_helpers import config


class RoadGeometryTests(unittest.TestCase):
    def test_straight_boundary_clearance(self):
        env = SafeMetaDriveEnv(research_config=config(traffic_density=0))
        try:
            env.reset(seed=0)
            lane = env.agent.navigation.current_ref_lanes[0]
            env.agent.set_heading_theta(lane.heading_theta_at(20))
            measured = []
            for margin in (1.0, .6, .3):
                lateral = -lane.width / 2 + env.agent.WIDTH / 2 + margin
                env.agent.set_position(lane.position(20, lateral))
                measured.append(env.road_clearance())
            self.assertGreater(measured[0], measured[1])
            self.assertGreater(measured[1], measured[2])
            # Lane-line collision bodies have finite thickness; allow that offset.
            for observed, expected in zip(measured, (1.0, .6, .3)):
                self.assertAlmostEqual(observed, expected, delta=.2)
        finally:
            env.close()
