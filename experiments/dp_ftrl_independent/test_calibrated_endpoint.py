"""Check endpoint tuning does not inspect held-out membership outcomes."""

import copy
import unittest

import numpy as np

from calibrated_endpoint import (apply_calibration, score_variants,
                                 variant_names)
from endpoint_baselines import (offline_lira_fixed_record_scores,
                                offline_rmia_record_scores)


class CalibratedEndpointTests(unittest.TestCase):
    def test_variants_match_record_equations(self):
        x = np.array([.2, .4, .6, .8])
        xr = np.array([[.25, .35, .55, .65],
                       [.30, .30, .50, .70],
                       [.20, .40, .60, .75]])
        z = np.array([.2, .3, .4, .5, .6, .7])
        zr = np.stack([z * .9, z, z * 1.1])
        raw = np.array([-.3, .1, .5, .9])
        raw_ref = np.stack([raw-.2, raw-.1, raw+.1])
        scores = score_variants(x, raw, xr, raw_ref, z, zr, 1.2)
        lira, rmia = variant_names()
        self.assertEqual(len(lira), 10)
        self.assertEqual(len(rmia), 48)
        self.assertEqual(set(scores), set(lira + rmia))
        self.assertTrue(np.isfinite(list(scores.values())).all())
        self.assertAlmostEqual(
            scores["lira_tune_original_fixed_mean"],
            np.mean(offline_lira_fixed_record_scores(x, xr, 1.2)))
        record_rmia = offline_rmia_record_scores(
            x, xr, z, zr, a=.5, gamma=1, population_correction=True)
        self.assertAlmostEqual(scores["rmia_tune_a050_g100_paper_mean"],
                               np.mean(record_rmia))

    def test_calibration_selection_ignores_holdout_labels(self):
        lira, rmia = variant_names()
        names = ["endpoint_loss", *lira, *rmia]
        records = []
        for index in range(7):
            positive = {name: 0. for name in names}
            negative = {name: 0. for name in names}
            if index < 4:
                positive[lira[0]] = 2.
                negative[lira[0]] = -2.
                positive[rmia[0]] = 3.
                negative[rmia[0]] = -3.
            else:
                positive[lira[1]] = 1000.
                negative[lira[1]] = -1000.
                positive[rmia[1]] = 1000.
                negative[rmia[1]] = -1000.
            records.append({"positive": positive, "negative": negative})
        selections = apply_calibration(records, 4)
        by_family = {r["family"]: r for r in selections}
        self.assertEqual(by_family["lira_calibrated"]["selected_variant"], lira[0])
        self.assertEqual(by_family["rmia_calibrated"]["selected_variant"], rmia[0])
        self.assertEqual(by_family["endpoint_best_calibrated"]["selected_variant"],
                         lira[0])
        changed = copy.deepcopy(records)
        for record in changed[4:]:
            record["positive"][lira[1]] = -1000.
            record["negative"][lira[1]] = 1000.
            record["positive"][rmia[1]] = -1000.
            record["negative"][rmia[1]] = 1000.
        again = apply_calibration(changed, 4)
        self.assertEqual([(x["selected_variant"], x["score_sign"])
                          for x in selections],
                         [(x["selected_variant"], x["score_sign"])
                          for x in again])


if __name__ == "__main__":
    unittest.main()
