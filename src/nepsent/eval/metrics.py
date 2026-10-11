"""Classification metrics, bootstrap intervals and paired tests."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support

from nepsent.data.clean import LABELS


def macro_f1(y_true, y_pred, labels=LABELS) -> float:
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def fast_macro_f1(y_true, y_pred, labels=LABELS) -> float:
    """Same value as macro_f1; accepts label strings or integer codes (index into labels)."""
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    t, p = np.asarray(y_true), np.asarray(y_pred)
    if t.dtype.kind not in "iu":
        t = np.fromiter((idx[v] for v in t), int, len(t))
    if p.dtype.kind not in "iu":
        p = np.fromiter((idx[v] for v in p), int, len(p))
    cm = np.bincount(t * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1.mean())


def report(y_true, y_pred, labels=LABELS) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(labels), zero_division=0)
    return {
        "n": int(len(y_true)),
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                            "support": int(s[i])} for i, lab in enumerate(labels)},
        "confusion_matrix": {"labels": list(labels),
                             "matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist()},
    }


def bootstrap_ci(y_true, y_pred, metric=macro_f1, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap interval over test items."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.array([metric(y_true[idx], y_pred[idx]) for idx in rng.integers(0, n, size=(n_boot, n))])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def paired_bootstrap(y_true, pred_a, pred_b, metric=macro_f1, n_boot=2000, seed=0) -> dict:
    """Test H0: metric(A) == metric(B) on the same items.

    Returns the observed difference (A - B), a percentile 95% interval and a
    two-sided p-value from the centred bootstrap distribution.
    """
    y, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(y)
    obs = metric(y, a) - metric(y, b)
    diffs = np.empty(n_boot)
    for k, idx in enumerate(rng.integers(0, n, size=(n_boot, n))):
        diffs[k] = metric(y[idx], a[idx]) - metric(y[idx], b[idx])
    centred = diffs - diffs.mean()
    p = float((np.sum(np.abs(centred) >= abs(obs)) + 1) / (n_boot + 1))
    return {"diff": float(obs), "ci_low": float(np.quantile(diffs, 0.025)),
            "ci_high": float(np.quantile(diffs, 0.975)), "p_value": p}


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test of H0: P(A correct, B wrong) == P(A wrong, B correct).

    This evaluates disagreement in correctness (accuracy), not macro F1.
    """
    y, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    ca, cb = a == y, b == y
    n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    p = 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": float(p)}


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for one family of tests."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def agreement(labels_a, labels_b) -> dict:
    """Raw agreement, Cohen's kappa and each annotator's label distribution."""
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    cats = sorted(set(a) | set(b))
    return {"n": int(len(a)), "raw_agreement": float(np.mean(a == b)),
            "cohen_kappa": float(cohen_kappa_score(a, b, labels=cats)),
            "label_dist_a": {c: int(np.sum(a == c)) for c in cats},
            "label_dist_b": {c: int(np.sum(b == c)) for c in cats}}
