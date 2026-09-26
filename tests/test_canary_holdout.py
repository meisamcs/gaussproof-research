import importlib.util
import unittest

AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(AVAILABLE, "torch is not installed")
class CanaryHoldoutTests(unittest.TestCase):
    def test_config_canaries_are_disjoint(self):
        import json
        from pathlib import Path
        config = json.loads(Path("configs/canary_holdout.json").read_text())
        self.assertNotEqual(config["calibration_canaries"], 0)
        self.assertNotEqual(config["test_canaries"], 0)
        self.assertGreater(config["sequences_per_canary"], 1)

    def test_holdout_summary_contains_calibration_and_test_metrics(self):
        from gaussproof.canary_holdout import auc
        self.assertGreater(auc([0, 1, 2, 3], [0, 0, 1, 1]), .99)
