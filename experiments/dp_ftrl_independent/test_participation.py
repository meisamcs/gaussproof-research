"""Checks for repeated-client B-of-N participation and replacement worlds."""

import unittest

import numpy as np

from endpoint_baselines import train_out_reference
from run_distributions import replacement_schedule
from run_pilot import simulate, tree_matrix


class RepeatedParticipationTests(unittest.TestCase):
    def test_sampling_probability_and_paired_replacement(self):
        rng = np.random.default_rng(84)
        cohort = np.array([f"b{i}" for i in range(16)], dtype=object)
        schedule, selected = replacement_schedule(cohort, 2000, 8, rng)
        self.assertEqual(schedule.shape, (2000, 8))
        self.assertTrue(all(len(set(row)) == 8 for row in schedule))
        self.assertTrue(all(schedule[t, -1] == cohort[-1] for t in selected))
        self.assertTrue(all(cohort[-1] not in schedule[t]
                            for t in range(2000) if t not in selected))
        self.assertAlmostEqual(len(selected) / 2000, .5, delta=.03)

    def test_absence_matches_out_reference_and_positive_changes_only_when_sampled(self):
        rng = np.random.default_rng(31)
        clients = {}
        for client_id in range(30):
            x = rng.normal(size=(4, 50))
            x[:, -1] = 1.
            clients[client_id] = (x, rng.integers(0, 10, size=4))

        class Data:
            def get(self, key):
                return clients[key]

        data = Data()
        schedule, selected = replacement_schedule(
            np.asarray(range(1, 17), dtype=object), 8, 8,
            np.random.default_rng(55))
        common = dict(sigma=2., clip=.25, client_lr=.2, server_lr=.005,
                      momentum=.9, clients_per_round=8, seed=12)
        reference = train_out_reference(data, schedule, **common)
        settings = dict(data=data, candidate_id=0, background_ids=schedule,
                        public_ids=[17, 18, 19], validation=[20, 21, 22],
                        rounds=8, position=2, matrix=tree_matrix(8),
                        penalty=.0001, sparse_iterations=10,
                        capture_final=True, replace_background=True,
                        **common)
        negative = simulate(member=False, participation_rounds=selected,
                            **settings)
        positive = simulate(member=True, participation_rounds=selected,
                            **settings)
        np.testing.assert_allclose(reference, negative["final_weights"],
                                   atol=1e-14)
        self.assertFalse(np.allclose(positive["final_weights"],
                                     negative["final_weights"]))
        never = simulate(member=True, participation_rounds=(), **settings)
        np.testing.assert_allclose(reference, never["final_weights"],
                                   atol=1e-14)
        # The selected-round mask is private: an absence-world public score
        # must not change if the diagnostic mask is altered.
        absent_other_mask = simulate(member=False,
                                     participation_rounds=(), **settings)
        for name in ("gaussproof_mean", "rero_mean", "public_gls_max",
                     "endpoint_loss"):
            self.assertEqual(negative[name], absent_other_mask[name])
        # Replacing a client with its own identical update changes nothing.
        identical = simulate(member=True, participation_rounds=selected,
                             **(settings | {"candidate_id": 16}))
        np.testing.assert_allclose(reference, identical["final_weights"],
                                   atol=1e-14)


if __name__ == "__main__":
    unittest.main()
