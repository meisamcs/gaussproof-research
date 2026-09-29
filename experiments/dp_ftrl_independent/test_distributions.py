"""Source-formula, role-separation, and mechanism checks for new baselines."""

import unittest

import numpy as np

from endpoint_baselines import (offline_lira_fixed_record_scores,
                                offline_rmia_record_scores,
                                pierre_offline_lira_cdf,
                                probability_logit, reference_global_std,
                                train_out_reference,
                                true_label_logits, true_label_probabilities)
from run_distributions import materialize_distribution
from run_pilot import simulate, tree_matrix


class EndpointBaselineTests(unittest.TestCase):
    def test_rmia_matches_explicit_offline_equation_including_ties(self):
        target = np.array([.5, .8, .25])
        references = np.array([[.4, .5, .3], [.6, .7, .2]])
        population_target = np.array([.5, .2, .7])
        population_refs = np.array([[.4, .3, .6], [.6, .2, .8]])
        a = .5
        gamma = 1.1
        px = ((1+a)*references.mean(axis=0)+(1-a))/2
        pz = ((1+a)*population_refs.mean(axis=0)+(1-a))/2
        expected = np.mean((target / px)[:, None] >=
                           gamma * (population_target / pz)[None, :], axis=1)
        np.testing.assert_allclose(offline_rmia_record_scores(
            target, references, population_target, population_refs,
            a=a, gamma=gamma), expected)
        ties = offline_rmia_record_scores(
            np.array([.2]), np.array([[.2], [.2]]),
            np.array([.2]), np.array([[.2], [.2]]), a=1, gamma=1)
        np.testing.assert_array_equal(ties, [1.])

    def test_pierre_rmia_population_uses_uncorrected_out_mean(self):
        x = np.array([.6, .8])
        xr = np.array([[.3, .4], [.4, .5]])
        z = np.array([.4, .5])
        zr = np.array([[.2, .3], [.3, .4]])
        a = .5
        x_ratio = x / (.5*((1+a)*xr.mean(axis=0)+(1-a)))
        z_ratio = z / zr.mean(axis=0)
        expected = np.mean(x_ratio[:, None] >= z_ratio[None, :], axis=1)
        np.testing.assert_allclose(offline_rmia_record_scores(
            x, xr, z, zr, a=a, gamma=1,
            population_correction=False), expected)

    def test_pierre_lira_cdf_matches_raw_logit_normal(self):
        from scipy.stats import norm

        target = np.array([.1, -.2])
        references = np.array([[.0, -.3], [.2, -.1], [.1, -.4]])
        expected = norm.cdf(target, references.mean(axis=0),
                            references.std(axis=0, ddof=1)+2e-9)
        np.testing.assert_allclose(
            pierre_offline_lira_cdf(target, references), expected)

    def test_lira_matches_negative_out_log_density(self):
        from scipy.stats import norm

        target = np.array([.4, .8])
        refs = np.array([[.2, .5], [.3, .6], [.4, .7]])
        std = reference_global_std(refs)
        scores = offline_lira_fixed_record_scores(target, refs, std)
        mu = np.median(probability_logit(refs), axis=0)
        expected = (.5*((probability_logit(target)-mu)/std)**2)
        np.testing.assert_allclose(scores, expected, rtol=1e-14)
        official_density = -norm.logpdf(probability_logit(target), mu, std)
        np.testing.assert_allclose(scores, official_density
                                   - np.log(std) - .5*np.log(2*np.pi))
        self.assertGreater(float(scores[1]), 0)

    def test_true_label_probability(self):
        weights = np.zeros((10, 50))
        client = (np.ones((3, 50)), np.array([0, 5, 9]))
        np.testing.assert_allclose(true_label_probabilities(weights, client), .1)
        np.testing.assert_allclose(true_label_logits(weights, client), 0.)

    def test_virtual_client_records_are_conserved_and_partitions_separate(self):
        clients = {}
        for i in range(12):
            x = np.zeros((4, 50))
            x[:, 0] = np.arange(4) + 4*i
            x[:, -1] = 1
            y = (np.arange(4) + i) % 3
            clients[i] = (x, y)

        class Data:
            def get(self, key):
                return clients[key]

        partitions = dict(candidate=list(range(4)), public=list(range(4, 8)),
                          population=list(range(8, 10)), background=list(range(10, 12)))
        for mode in ("natural_writer", "iid_mixed", "label_sorted"):
            mixed, purity = materialize_distribution(Data(), partitions,
                                                      mode, 4, seed=17)
            self.assertEqual(set(mixed.items), set(clients))
            for role, ids in partitions.items():
                old = sorted(clients[i][0][j, 0] for i in ids for j in range(4))
                new = sorted(mixed.get(i)[0][j, 0] for i in ids for j in range(4))
                self.assertEqual(old, new)
                self.assertTrue(0 < purity[role] <= 1)

    def test_out_reference_matches_absence_world_mechanism(self):
        rng = np.random.default_rng(4)
        ids = list(range(37))
        clients = {}
        for client_id in ids:
            x = rng.normal(size=(4, 50))
            x[:, -1] = 1
            clients[client_id] = (x, rng.integers(0, 10, size=4))

        class Data:
            def get(self, key):
                return clients[key]

        data = Data()
        schedule = np.asarray(ids[1:29]).reshape(4, 7)
        common = dict(sigma=2., clip=.25, client_lr=.2, server_lr=.005,
                      momentum=.9, clients_per_round=8, seed=12)
        reference = train_out_reference(data, schedule, **common)
        baseline = simulate(data, ids[0], schedule, ids[29:32], ids[32:35],
                            member=False, rounds=4, position=2,
                            matrix=tree_matrix(4), penalty=.0001,
                            sparse_iterations=10, capture_final=True, **common)
        np.testing.assert_allclose(reference, baseline["final_weights"], atol=1e-14)


if __name__ == "__main__":
    unittest.main()
