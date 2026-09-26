"""Negative controls report observed behavior; never require a favored attack to win."""
import numpy as np
from .attacks import trajectory_scores
from .metrics import auc


def permutation_control(scores, labels, repeats, seed):
    rng = np.random.default_rng(seed)
    values = [auc(scores, rng.permutation(labels)) for _ in range(repeats)]
    return dict(mean_auc=float(np.mean(values)), std_auc=float(np.std(values)), repeats=repeats)


def synthetic_controls(seed=19, repeats=12):
    rows = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed+repeat)
        n, dim, steps = 32, 40, 32
        bank = rng.normal(size=(n, dim))
        bank /= np.linalg.norm(bank, axis=1, keepdims=True)
        dictionary = np.repeat(bank[None], steps, axis=0)
        labels = np.r_[np.ones(n//2), np.zeros(n//2)]
        z = np.zeros((steps, n))
        for t in range(steps):
            z[t, rng.choice(n//2, 3, replace=False)] = 1
        signal = z @ bank
        noise = rng.normal(size=signal.shape)
        for name, releases in [('structured', signal+0.05*noise),
                               ('pure_noise', noise),
                               ('nonmember_only', np.roll(z, n//2, axis=1)@bank+0.05*noise),
                               ('high_noise', signal+4*noise)]:
            scores = trajectory_scores(dictionary, releases, penalty=0.08, iterations=80)
            for attack, score in scores.items():
                rows.append(dict(repeat=repeat, control=name, attack=attack, auc=auc(score, labels)))
    return rows
