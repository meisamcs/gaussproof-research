"""Tie-aware empirical ROC and independent calibration thresholds."""
import numpy as np


def checked(scores, labels):
    s, y = np.asarray(scores, dtype=float), np.asarray(labels)
    if s.ndim != 1 or y.shape != s.shape or not np.isfinite(s).all():
        raise ValueError("Scores must be finite vectors matching labels")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both binary membership classes are required")
    return s, y.astype(int)


def roc(scores, labels):
    s, y = checked(scores, labels)
    order = np.argsort(-s, kind='stable')
    s, y = s[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.r_[0, np.cumsum(y)[ends]] / y.sum()
    fp = np.r_[0, np.cumsum(1 - y)[ends]] / (1 - y).sum()
    return fp, tp, np.r_[np.inf, s[ends]]


def auc(scores, labels):
    fpr, tpr, _ = roc(scores, labels)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def threshold_at_fpr(scores, labels, level):
    fpr, _, thresholds = roc(scores, labels)
    return float(thresholds[np.flatnonzero(fpr <= level)[-1]])


def rates(scores, labels, threshold):
    s, y = checked(scores, labels)
    pred = s >= threshold
    return float(pred[y == 1].mean()), float(pred[y == 0].mean())


def evaluate(scores, labels, cal_scores, cal_labels, bootstrap=200, seed=0):
    s, y = checked(scores, labels)
    fpr, tpr, _ = roc(s, y)
    result = dict(auc=auc(s, y), max_advantage=float(np.max(tpr - fpr)),
                  n_member=int(y.sum()), n_nonmember=int((1-y).sum()),
                  fpr_resolution=float(1 / (1-y).sum()))
    cf, ct, thresholds = roc(cal_scores, cal_labels)
    best = thresholds[np.argmax(ct - cf)]
    et, ef = rates(s, y, best)
    result.update(balanced_accuracy=0.5*(et+1-ef), calibrated_advantage=et-ef)
    for level, tag in [(0.01, '1pct'), (0.05, '5pct')]:
        # Empirical step ROC: no extrapolation/interpolation between tied scores.
        result['tpr_at_' + tag + '_fpr'] = float(np.max(tpr[fpr <= level]))
        th = threshold_at_fpr(cal_scores, cal_labels, level)
        et, ef = rates(s, y, th)
        result['calibrated_tpr_' + tag] = et
        result['calibrated_fpr_' + tag] = ef
        result['threshold_' + tag] = th if np.isfinite(th) else 'inf'
    rng = np.random.default_rng(seed)
    pos, neg = s[y == 1], s[y == 0]
    samples = [auc(np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))],
                   np.r_[np.ones(len(pos)), np.zeros(len(neg))]) for _ in range(bootstrap)]
    result['auc_ci_low'], result['auc_ci_high'] = map(float, np.quantile(samples, [.025, .975]))
    return result
