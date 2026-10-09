import unittest
import numpy as np
from env.research_config import load_config
from env.risk_utils import VehicleState
from env.safety_cost import compute_safety_cost


class CostRangeTests(unittest.TestCase):
    def test_range_and_overrides(self):
        c = load_config()["safety"]
        ego = VehicleState("ego", (0,0), (10,0), 0,4,2)
        rng = np.random.default_rng(123)
        for _ in range(80):
            other = VehicleState("j", tuple(rng.normal(0,10,2)), tuple(rng.normal(0,10,2)), float(rng.uniform(-3,3)),4,2)
            for mode in ("binary", "continuous"):
                c["cost_mode"] = mode
                for collision, out in ((False, False), (True, False), (False, True)):
                    info = compute_safety_cost(ego, [other], float(rng.uniform(-1,4)), collision, out, c)
                    self.assertTrue(0 <= info["cost"] <= 1)
                    if collision or out:
                        self.assertEqual(info["cost"], 1)

    def test_nonfinite_rejected(self):
        c = load_config()["safety"]
        ego = VehicleState("ego", (0,0), (10,0),0,4,2)
        bad = VehicleState("j", (float("nan"),0), (0,0),0,4,2)
        with self.assertRaises(ValueError):
            compute_safety_cost(ego, [bad], 2, False, False, c)
        with self.assertRaises(ValueError):
            compute_safety_cost(ego, [], float("nan"), False, False, c)
        with self.assertRaises(ValueError):
            compute_safety_cost(bad, [], 2, False, False, c)
