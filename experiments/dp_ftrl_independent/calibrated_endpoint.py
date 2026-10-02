"""Generous, calibration-only LiRA/RMIA endpoint adaptation for clients.

The original attacks are record-level. This module tries prespecified record
scores and mean/max bag aggregation, then uses only calibration identities to
select each family member and score direction. Held-out identities select
nothing. The paper-faithful scores remain separate in run_distributions.py.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

from endpoint_baselines import (offline_lira_fixed_record_scores,
                                offline_rmia_record_scores,
                                pierre_offline_lira_cdf,
                                probability_logit)
from run_pilot import cross_identity_auc


RMIA_A = (0., .5, 1.)
RMIA_GAMMA = (1., 1.05, 1.2, 2.)
AGGREGATIONS = ("mean", "max")
LIRA_NAMES = ("original_fixed", "original_variable", "scaled_cdf_fixed",
              "scaled_cdf_variable", "pierre_raw_cdf")


def _aggregate(values: np.ndarray, name: str) -> float:
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Empty or nonfinite record scores")
    return float(np.mean(values) if name == "mean" else np.max(values))


def variant_names():
    lira = [f"lira_tune_{name}_{agg}" for name in LIRA_NAMES
            for agg in AGGREGATIONS]
    rmia = [
        f"rmia_tune_a{int(100*a):03d}_g{int(100*gamma):03d}_{denom}_{agg}"
        for a in RMIA_A for gamma in RMIA_GAMMA
        for denom in ("paper", "pierre") for agg in AGGREGATIONS]
    return lira, rmia


def score_variants(target: np.ndarray, target_raw: np.ndarray,
                   reference: np.ndarray, reference_raw: np.ndarray,
                   population_target: np.ndarray,
                   population_reference: np.ndarray,
                   fixed_std: float) -> dict[str, float]:
    """All inputs are endpoint-only; no member labels enter this function."""
    logit = probability_logit(target)
    out_logit = probability_logit(reference)
    out_mean = np.median(out_logit, axis=0)
    out_std = np.maximum(out_logit.std(axis=0), 1e-12)
    lira_record = {
        "original_fixed": offline_lira_fixed_record_scores(
            target, reference, fixed_std),
        "original_variable": -norm.logpdf(logit, out_mean, out_std),
        # One-sided OUT-CDF adaptations are labeled separately from the
        # original two-sided offline negative-density score.
        "scaled_cdf_fixed": ndtr((logit - out_mean) / fixed_std),
        "scaled_cdf_variable": ndtr((logit - out_mean) / out_std),
        "pierre_raw_cdf": pierre_offline_lira_cdf(target_raw, reference_raw),
    }
    scores = {}
    for name, values in lira_record.items():
        for agg in AGGREGATIONS:
            scores[f"lira_tune_{name}_{agg}"] = _aggregate(values, agg)
    for a in RMIA_A:
        for gamma in RMIA_GAMMA:
            for denom in ("paper", "pierre"):
                record = offline_rmia_record_scores(
                    target, reference, population_target,
                    population_reference, a=a, gamma=gamma,
                    population_correction=(denom == "paper"))
                prefix = (f"rmia_tune_a{int(100*a):03d}_"
                          f"g{int(100*gamma):03d}_{denom}")
                for agg in AGGREGATIONS:
                    scores[f"{prefix}_{agg}"] = _aggregate(record, agg)
    expected = set(sum((list(group) for group in variant_names()), []))
    if set(scores) != expected:
        raise AssertionError("Endpoint variant list disagrees with scores")
    return scores


def select_family(calibration: list[dict], names: list[str] | tuple[str, ...]):
    """Pick one method/sign using calibration pairs, with stable tie order."""
    if len(calibration) < 2 or not names:
        raise ValueError("Need paired calibration identities and candidates")
    selected = None
    for name in names:
        auc = cross_identity_auc(calibration, name)
        # AUC(-score)=1-AUC(score), including tie-aware cross-identity AUC.
        sign = 1 if auc >= .5 else -1
        oriented = max(auc, 1 - auc)
        if selected is None or oriented > selected[2] + 1e-12:
            selected = (name, sign, oriented)
    return selected


def apply_calibration(records: list[dict], calibration_count: int):
    """Add three selected endpoint scores without inspecting holdout labels."""
    if not 2 <= calibration_count < len(records) - 1:
        raise ValueError("Need separate calibration and holdout identities")
    lira, rmia = variant_names()
    families = {
        "lira_calibrated": lira,
        "rmia_calibrated": rmia,
        "endpoint_best_calibrated": ["endpoint_loss", *lira, *rmia],
    }
    selection = []
    for family, names in families.items():
        chosen, sign, calibration_auc = select_family(
            records[:calibration_count], names)
        for record in records:
            for world in ("positive", "negative"):
                record[world][family] = sign * record[world][chosen]
        selection.append(dict(family=family, selected_variant=chosen,
                              score_sign=sign, calibration_auc=calibration_auc,
                              holdout_auc=cross_identity_auc(
                                  records[calibration_count:], family),
                              candidate_variants=len(names),
                              calibration_clients=calibration_count,
                              holdout_clients=len(records)-calibration_count))
    return selection
