"""Run only the environment acceptance tests, excluding legacy shield tests."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

if __name__ == "__main__":
    modules = ["test_env_reset", "test_env_step", "test_cost_range", "test_cost_monotonicity",
               "test_seed_reproducibility", "test_train_test_split", "test_termination", "test_road_geometry", "test_sac_env_interface", "test_reward_modes"]
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    path = ROOT / "outputs/unit_tests.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"tests": result.testsRun, "failures": len(result.failures),
                                "errors": len(result.errors), "successful": result.wasSuccessful()}, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
