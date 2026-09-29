"""Audited offline record-level RMIA and LiRA, aggregated to client scores.

These attacks use only the final target checkpoint. Reference models are OUT
for every candidate and population record. The client-level score is the mean
of its record-level scores; this aggregation is an adaptation, not a claim
that the original papers studied client-level DP-FTRLM.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

from run_pilot import EfficientTree, clipped_client_delta, server_step


def true_label_probabilities(weights: np.ndarray, client) -> np.ndarray:
    x, labels = client
    logits = np.asarray(x, np.float64) @ np.asarray(weights, np.float64).T
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    labels = np.asarray(labels, np.int64)
    if labels.shape != (len(x),) or np.any((labels < 0) | (labels >= 10)):
        raise ValueError("Invalid digit labels")
    return probabilities[np.arange(len(labels)), labels]


def true_label_logits(weights: np.ndarray, client) -> np.ndarray:
    """Raw correct-class logits for the Pierre-Joly CDF variant."""
    x, labels = client
    logits = np.asarray(x, np.float64) @ np.asarray(weights, np.float64).T
    labels = np.asarray(labels, np.int64)
    if labels.shape != (len(x),) or np.any((labels < 0) | (labels >= 10)):
        raise ValueError("Invalid digit labels")
    return logits[np.arange(len(labels)), labels]


def train_out_reference(data, schedule, *, sigma: float, clip: float,
                        client_lr: float, server_lr: float, momentum: float,
                        clients_per_round: int, seed: int) -> np.ndarray:
    """Train an OUT model with the same fixed-slot DP-FTRLM server recurrence.

    The scheduled clients fill B-1 slots, leaving one zero slot, exactly as
    in the target's absence world. The caller must use a public pool disjoint
    from candidate and RMIA population examples.
    """
    if schedule.ndim != 2 or schedule.shape[1] != clients_per_round - 1:
        raise ValueError("Expected one empty client slot in every round")
    if sigma <= 0 or clip <= 0 or clients_per_round < 2:
        raise ValueError("Invalid mechanism parameters")
    initial = np.zeros((10, 50), np.float64)
    weights = initial.copy()
    velocity = initial.copy()
    cumulative = initial.copy()
    rng = np.random.default_rng(seed)
    std = sigma * clip / clients_per_round
    tree = EfficientTree(lambda: rng.normal(0, std, initial.size).reshape(initial.shape))
    for round_ids in schedule:
        update = np.zeros_like(weights)
        for client_id in round_ids:
            update -= clipped_client_delta(weights, data.get(client_id),
                                           client_lr, clip) / clients_per_round
        cumulative += update
        weights, velocity = server_step(initial, velocity, cumulative,
                                        tree.next(), server_lr, momentum)
    return weights


def offline_rmia_record_scores(target: np.ndarray, references: np.ndarray,
                               population_target: np.ndarray,
                               population_references: np.ndarray,
                               *, a: float, gamma: float,
                               population_correction: bool = True) -> np.ndarray:
    """Equation (5) and offline Eq. (10) of Zarifzadeh et al. (ICML 2024).

    p_hat(x)=((1+a)*mean_OUT(p(x|theta'))+(1-a))/2. Return the fraction of
    independent population records z for which r(x)>=gamma*r(z). Equality
    is included, as in the paper. No target membership labels are inputs.
    """
    x = np.asarray(target, np.float64)
    xr = np.asarray(references, np.float64)
    z = np.asarray(population_target, np.float64)
    zr = np.asarray(population_references, np.float64)
    if (x.ndim != 1 or z.ndim != 1 or not len(z) or xr.ndim != 2 or
            zr.ndim != 2 or xr.shape[1] != len(x) or zr.shape[1] != len(z) or
            xr.shape[0] == 0 or zr.shape[0] != xr.shape[0] or
            not 0 <= a <= 1 or gamma < 1):
        raise ValueError("Invalid offline RMIA shapes or parameters")
    if not all(np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()
               for v in (x, xr, z, zr)):
        raise ValueError("RMIA expects finite true-label probabilities")

    def ratio(target_probs, reference_probs):
        p_hat = ((1 + a) * reference_probs.mean(axis=0) + (1 - a)) / 2
        return target_probs / np.maximum(p_hat, 1e-15)

    if population_correction:
        z_ratio = np.sort(ratio(z, zr))
    else:
        # Pierre-Joly/Membership-Inference-Attacks@9182ed8 uses the OUT
        # reference mean directly for z, but the offline a-correction for x.
        z_ratio = np.sort(z / np.maximum(zr.mean(axis=0), 1e-15))
    x_ratio = ratio(x, xr)
    return np.searchsorted(z_ratio, x_ratio / gamma, side="right") / len(z_ratio)


def offline_lira_fixed_record_scores(target: np.ndarray,
                                     references: np.ndarray,
                                     global_std: float) -> np.ndarray:
    """Offline fixed-variance LiRA: minus Gaussian OUT log-density.

    This follows the original `mi_lira_2021/plot.py`: logit(true-label
    probability), per-record median OUT mean, and one global OUT standard
    deviation. The additive Gaussian normalizer is common to all records and
    is omitted. Higher scores mean stronger membership evidence.
    """
    x = np.asarray(target, np.float64)
    refs = np.asarray(references, np.float64)
    if x.ndim != 1 or refs.ndim != 2 or refs.shape[1] != len(x) or not len(refs):
        raise ValueError("Invalid LiRA probability shapes")
    if not np.isfinite(global_std) or global_std <= 0:
        raise ValueError("Global OUT standard deviation must be positive")
    if not all(np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()
               for v in (x, refs)):
        raise ValueError("LiRA expects finite true-label probabilities")
    x_logit = probability_logit(x)
    ref_logit = probability_logit(refs)
    mean_out = np.median(ref_logit, axis=0)
    return .5 * ((x_logit - mean_out) / global_std) ** 2


def probability_logit(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, np.float64), 1e-15, 1 - 1e-15)
    return np.log(p) - np.log1p(-p)


def reference_global_std(reference_probs: np.ndarray) -> float:
    """Fixed LiRA variance estimated only from OUT reference predictions."""
    values = probability_logit(reference_probs)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("At least two OUT reference models are required")
    std = float(np.std(values))
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Degenerate OUT reference distribution")
    return std


def pierre_offline_lira_cdf(target_logits: np.ndarray,
                            reference_logits: np.ndarray) -> np.ndarray:
    """Pierre-Joly's offline LiRA variant on raw correct-class logits.

    It returns the OUT Gaussian CDF with per-record sample mean/std, matching
    `attacks/offline_lira.py` at commit 9182ed8. This is different from the
    original Carlini offline fixed-variance negative-density score.
    """
    x = np.asarray(target_logits, np.float64)
    refs = np.asarray(reference_logits, np.float64)
    if x.ndim != 1 or refs.ndim != 2 or refs.shape[1] != len(x) or len(refs) < 2:
        raise ValueError("Pierre-Joly LiRA requires at least two OUT references")
    if not np.isfinite(x).all() or not np.isfinite(refs).all():
        raise ValueError("Nonfinite logits")
    mean = refs.mean(axis=0)
    # torch.std(dim=0) in the linked repository uses the sample correction.
    std = refs.std(axis=0, ddof=1) + 1e-9
    # Its `gaussian_cdf` adds a second epsilon in the denominator.
    return ndtr((x - mean) / (std + 1e-9))
