import unittest
from env.research_config import load_config, SeedManager


class SplitTests(unittest.TestCase):
    def test_disjoint(self):
        train = load_config("configs/env/train.yaml")["scenario"]
        test = load_config("configs/env/eval_id.yaml")["scenario"]
        self.assertFalse(set(range(train["start_seed"], train["start_seed"]+train["num_scenarios"])) & set(range(test["start_seed"], test["start_seed"]+test["num_scenarios"])))
        with self.assertRaises(ValueError):
            load_config("configs/env/eval_id.yaml", {"scenario": {"start_seed": 999, "num_scenarios": 2}})
        with self.assertRaises(ValueError):
            SeedManager(load_config()).scene(0)
