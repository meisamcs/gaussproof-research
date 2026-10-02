import unittest
import csv
import tempfile
from pathlib import Path

import numpy as np


class CanaryQSensitivityTests(unittest.TestCase):
    def test_stratified_identities_are_balanced_and_fixed_order(self):
        from gaussproof.canary_q_sensitivity import stratified_identities
        labels = np.tile(np.arange(10), 3)
        pool = list(range(30))
        chosen = stratified_identities(pool, labels, 2)
        self.assertEqual(len(chosen), 20)
        self.assertEqual(np.bincount(labels[chosen], minlength=10).tolist(), [2] * 10)

    def test_identity_offset_creates_disjoint_fixed_holdout(self):
        from gaussproof.canary_q_sensitivity import stratified_identities
        labels = np.tile(np.arange(10), 10)
        pool = list(range(100))
        pilot = stratified_identities(pool, labels, 2)
        holdout = stratified_identities(pool, labels, 8, offset_per_class=2)
        self.assertEqual(len(holdout), 80)
        self.assertTrue(set(pilot).isdisjoint(holdout))

    def test_prefix_scores_use_checkpoint_loss_at_requested_prefix(self):
        from gaussproof.canary_q_sensitivity import prefix_scores
        rng = np.random.default_rng(4)
        run = dict(
            observations=rng.normal(size=(4, 7)).astype("float32"),
            fingerprints=rng.normal(size=(4, 7)).astype("float32"),
            backgrounds=rng.normal(size=(4, 7)).astype("float32"),
            included=np.asarray([0, 1, 0, 1]),
            endpoint_losses=np.asarray([3.0, 2.0, 1.0, .5]),
            endpoint_confidences=np.asarray([.1, .2, .4, .7]),
        )
        result = prefix_scores(run, sigma=4.0, clip=1.0, batch_size=8,
                               probability=.1, lengths=[2, 4])
        self.assertEqual(result[0]["endpoint_neg_loss"], -2.0)
        self.assertEqual(result[1]["endpoint_neg_loss"], -.5)
        self.assertEqual(result[0]["endpoint_confidence"], .2)

    def test_resume_accepts_only_complete_q_arms(self):
        from gaussproof.canary_q_sensitivity import load_completed_q_rows
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            row = dict(q=0.5, sigma=4.0, canary=7, digit=1, label=0,
                       sequence=0, inclusion_rate=0.0, steps=16,
                       sequence_llr=0.1, mixture_llr=0.2,
                       endpoint_neg_loss=-1.0, endpoint_confidence=0.4)
            rows = [dict(row, label=label, steps=step)
                    for label in (0, 1) for step in (16, 32)]
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader(); writer.writerows(rows)
            recovered, completed = load_completed_q_rows(
                path, [0.5, 0.1], [16, 32], [7], 4.0, 1)
            self.assertEqual(len(recovered), 4)
            self.assertEqual(completed, {0.5})
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader(); writer.writerows(rows[:-1])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_completed_q_rows(path, [0.5, 0.1], [16, 32], [7], 4.0, 1)


if __name__ == "__main__":
    unittest.main()
