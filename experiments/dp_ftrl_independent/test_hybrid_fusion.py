"""Checks that trajectory fusion is fair and calibration-only."""

import copy
import unittest

from hybrid_fusion import apply_hybrids
from run_pilot import cross_identity_auc


def fixture():
    records = []
    for index in range(8):
        base = index * .08
        records.append({
            "positive": {"lira_calibrated": base + .15,
                         "rmia_calibrated": base + .11,
                         "gaussproof_mean": (index % 3) * .05 + .02},
            "negative": {"lira_calibrated": base,
                         "rmia_calibrated": base,
                         "gaussproof_mean": (index % 3) * .05},
        })
    return records


class HybridFusionTests(unittest.TestCase):
    def test_holdout_labels_do_not_select_weight_or_normalization(self):
        original = fixture()
        modified = copy.deepcopy(original)
        for record in modified[4:]:
            record["positive"], record["negative"] = (
                record["negative"], record["positive"])
        selected_a = apply_hybrids(original, 4)
        selected_b = apply_hybrids(modified, 4)
        for row_a, row_b in zip(selected_a, selected_b):
            self.assertEqual(row_a["selected_trajectory_weight"],
                             row_b["selected_trajectory_weight"])
            self.assertEqual(row_a["calibration_auc"],
                             row_b["calibration_auc"])
        for a, b in zip(original[:4], modified[:4]):
            for world in ("positive", "negative"):
                for family in ("lira", "rmia"):
                    self.assertEqual(a[world][f"hybrid_equal_{family}"],
                                     b[world][f"hybrid_equal_{family}"])

    def test_zero_weight_falls_back_to_endpoint_when_trajectory_constant(self):
        records = fixture()
        for record in records:
            for world in ("positive", "negative"):
                record[world]["gaussproof_mean"] = 1.
        choices = apply_hybrids(records, 4)
        for choice in choices:
            family = choice["family"]
            self.assertEqual(choice["selected_trajectory_weight"], 0.)
            self.assertAlmostEqual(
                cross_identity_auc(records[4:],
                                   f"hybrid_calibrated_{family}"),
                cross_identity_auc(records[4:],
                                   f"{family}_calibrated"))


if __name__ == "__main__":
    unittest.main()
