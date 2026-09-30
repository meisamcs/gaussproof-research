import unittest

import numpy as np

from gaussproof.trajectory_access_robustness import (
    access_scores, configurations, summarize,
)


class TrajectoryAccessRobustnessTest(unittest.TestCase):
    def setUp(self):
        self.run = dict(
            observations=np.array([[2., 0.], [0., 1.], [3., 0.], [0., 2.]]),
            fingerprints=np.array([[1., 0.], [0., 1.], [2., 0.], [0., 2.]]),
            backgrounds=np.zeros((4, 2)),
        )

    def test_access_is_deterministic_and_stale_changes_fingerprint(self):
        full = access_scores(self.run, .5, 2., 1., 2)
        self.assertEqual(full[2], 4)
        self.assertAlmostEqual(full[1], 13.)
        self.assertEqual(full, access_scores(self.run, .5, 2., 1., 2, stale=1,
                                             assumed_sigma=2.))
        sparse = access_scores(self.run, .5, 2., 1., 2, cadence=2)
        self.assertEqual(sparse[2], 2)
        self.assertAlmostEqual(sparse[1], 5.)
        stale = access_scores(self.run, .5, 2., 1., 2, stale=2)
        self.assertAlmostEqual(stale[1], 8.)
        self.assertNotEqual(stale[0], full[0])

    def test_predeclared_scenarios_have_unique_names(self):
        scenarios = configurations(dict(sigma=4., cadences=[1, 4, 16],
                                        stale_intervals=[1, 4, 16],
                                        assumed_sigmas=[3., 4., 5.]))
        self.assertEqual(len(scenarios), 7)
        self.assertEqual(len({s['name'] for s in scenarios}), 7)

    def test_summary_uses_holdout_and_reports_achieved_fpr(self):
        rows = []
        for identity in range(8):
            for world in (0, 1):
                rows.append(dict(identity=identity, world=world, scenario='full',
                    visible_releases=4, gaussproof=world + identity*.001,
                    alignment=world + identity*.001,
                    endpoint_loss=world + identity*.001,
                    public_accuracy=.7, appearances=2*world))
        cfg = dict(seed=9, bootstrap_repetitions=40, calibration_fpr=.05)
        result = summarize(rows, [dict(name='full')], list(range(4)),
                           list(range(4, 8)), cfg)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(row['auc'] == 1. for row in result))
        self.assertTrue(all(row['gain_vs_alignment'] == 0. for row in result))
        self.assertTrue(all(0 <= row['achieved_fpr'] <= 1 for row in result))


if __name__ == '__main__':
    unittest.main()
