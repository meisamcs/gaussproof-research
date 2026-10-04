import unittest

import numpy as np

from gaussproof.interference_audit import select_overlap_cohorts, overlap_stats


class InterferenceSelectionTests(unittest.TestCase):
    def test_matched_classes_and_high_low_overlap(self):
        ids = np.arange(12)
        labels = np.repeat([1, 7], 6)
        vectors = np.array([[1, 0], [1, .01], [1, .02], [0, 1], [.01, 1], [.02, 1]] * 2,
                           dtype=float)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        cohorts = select_overlap_cohorts(ids, labels, vectors, [1, 7], 3, 6, 41)
        for positions in cohorts.values():
            np.testing.assert_array_equal(np.bincount(labels[positions], minlength=8)[[1, 7]],
                                          [3, 3])
        high = overlap_stats(cohorts['high'], labels, vectors)
        low = overlap_stats(cohorts['low'], labels, vectors)
        self.assertGreater(high['within_abs_cosine'], low['within_abs_cosine'])


if __name__ == '__main__':
    unittest.main()
