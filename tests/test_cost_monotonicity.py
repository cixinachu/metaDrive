import unittest
import math
from env.research_config import load_config
from env.risk_utils import VehicleState, compute_pairwise_risk, compute_ttc_risk, compute_distance_risk, rectangle_distance


class RiskTests(unittest.TestCase):
    def setUp(self):
        self.c = load_config()["safety"]
        self.ego = VehicleState("ego", (0, 0), (10, 0), 0, 4, 2)

    def test_ttc_monotonic(self):
        values = [compute_ttc_risk(t, 4) for t in (6, 4, 3, 2, 1, 0)]
        self.assertEqual(values, sorted(values))
        self.assertEqual(values[-1], 1)

    def test_distance_monotonic(self):
        values = [compute_distance_risk(d, 10, 10, self.c) for d in (30, 15, 8, 3, 0)]
        self.assertEqual(values, sorted(values))

    def test_equal_speed_close_following(self):
        near = compute_pairwise_risk(self.ego, VehicleState("j", (5, 0), (10, 0), 0, 4, 2), self.c)
        self.assertIsNone(near["ttc"])
        self.assertGreater(near["distance_risk"], .8)

    def test_crossing(self):
        near = VehicleState("j", (10, -10), (0, 10), math.pi/2, 4, 2)
        result = compute_pairwise_risk(self.ego, near, self.c)
        self.assertGreater(result["encounter_risk"], .5)
        self.assertLess(result["predicted_distance"], .001)
        self.assertIsNone(result["ttc"])

    def test_parallel_lanes_and_receding(self):
        result = compute_pairwise_risk(self.ego, VehicleState("j", (5, 4), (10, 0), 0, 4, 2), self.c)
        self.assertEqual(result["risk"], 0)
        result = compute_pairwise_risk(self.ego, VehicleState("j", (40, 0), (20, 0), 0, 4, 2), self.c)
        self.assertEqual(result["risk"], 0)

    def test_rectangle_geometry(self):
        self.assertAlmostEqual(rectangle_distance(self.ego, VehicleState("j", (10, 0), (0,0), 0,4,2)), 6)
        self.assertAlmostEqual(rectangle_distance(self.ego, VehicleState("j", (0, 5), (0,0), 0,4,2)), 3)

    def test_cut_in_and_rotation(self):
        side_by_side = VehicleState("j", (5, 5), (10, -2), 0, 4, 2)
        result = compute_pairwise_risk(self.ego, side_by_side, self.c)
        self.assertGreater(result["encounter_risk"], 0)
        # Rotating the entire scene must not change physical risk.
        ego_rotated = VehicleState("ego", (0,0), (0,10), math.pi/2, 4,2)
        other_rotated = VehicleState("j", (-5,5), (2,10), math.pi/2,4,2)
        rotated = compute_pairwise_risk(ego_rotated, other_rotated, self.c)
        self.assertAlmostEqual(result["risk"], rotated["risk"])
