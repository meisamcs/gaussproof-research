"""Calibration-only fusion of endpoint MIA and GAUSSPROOF trajectory scores.

The two scores are standardized using calibration identities only. We report
both a predeclared equal-weight fusion and a weight selected from a small,
fixed grid on calibration identities. Zero trajectory weight is included so
the selected method can fall back to the endpoint. No held-out labels enter
normalization or selection.
"""

from __future__ import annotations

import numpy as np

from run_pilot import cross_identity_auc


TRAJECTORY = "gaussproof_mean"
ENDPOINTS = {"lira": "lira_calibrated", "rmia": "rmia_calibrated"}
TRAJECTORY_WEIGHTS = (0., .25, .5, .75, 1.)


def apply_hybrids(records: list[dict], calibration_count: int):
    """Add equal-weight and calibration-selected scores to paired records."""
    if not 2 <= calibration_count < len(records) - 1:
        raise ValueError("Need separate calibration and holdout identities")
    calibration = records[:calibration_count]
    heldout = records[calibration_count:]
    selections = []
    for family, endpoint in ENDPOINTS.items():
        center = {}
        scale = {}
        for feature in (endpoint, TRAJECTORY):
            values = np.asarray([record[world][feature]
                                 for record in calibration
                                 for world in ("positive", "negative")],
                                dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite calibration score: {feature}")
            center[feature] = float(values.mean())
            sd = float(values.std())
            scale[feature] = sd if sd > 1e-12 else 1.
        for record in records:
            for world in ("positive", "negative"):
                item = record[world]
                endpoint_z = (item[endpoint] - center[endpoint]) / scale[endpoint]
                trajectory_z = (item[TRAJECTORY] - center[TRAJECTORY]) / scale[TRAJECTORY]
                for weight in TRAJECTORY_WEIGHTS:
                    item[f"_hybrid_{family}_{weight:g}"] = (
                        (1 - weight) * endpoint_z + weight * trajectory_z)
                item[f"hybrid_equal_{family}"] = item[f"_hybrid_{family}_0.5"]
        best_weight = TRAJECTORY_WEIGHTS[0]
        best_auc = -np.inf
        for weight in TRAJECTORY_WEIGHTS:
            score = f"_hybrid_{family}_{weight:g}"
            auc = cross_identity_auc(calibration, score)
            if auc > best_auc + 1e-12:
                best_weight, best_auc = weight, auc
        selected_key = f"_hybrid_{family}_{best_weight:g}"
        for record in records:
            for world in ("positive", "negative"):
                item = record[world]
                item[f"hybrid_calibrated_{family}"] = item[selected_key]
                for weight in TRAJECTORY_WEIGHTS:
                    del item[f"_hybrid_{family}_{weight:g}"]
        selections.append(dict(
            family=family, endpoint_method=endpoint,
            trajectory_method=TRAJECTORY,
            selected_trajectory_weight=best_weight,
            selected_endpoint_weight=1 - best_weight,
            calibration_auc=float(best_auc),
            heldout_auc=float(cross_identity_auc(
                heldout, f"hybrid_calibrated_{family}")),
            equal_weight_heldout_auc=float(cross_identity_auc(
                heldout, f"hybrid_equal_{family}")),
            endpoint_heldout_auc=float(cross_identity_auc(heldout, endpoint)),
            calibration_clients=calibration_count,
            heldout_clients=len(heldout),
        ))
    return selections
