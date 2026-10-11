"""Unit tests for data handling and evaluation assumptions.

Fixtures are small hand-written strings and label vectors used only to
exercise code paths. They are not research data.
"""
import numpy as np
import pandas as pd
import pytest

from nepsent.data.clean import dedup_key, mask_pii, near_duplicate_pairs, normalize, stable_id
from nepsent.data.split import grouped_split, ids_hash, stratified_subsample
from nepsent.eval.metrics import agreement, bootstrap_ci, holm, macro_f1, mcnemar_exact, paired_bootstrap, report


def test_normalize_nfc_and_whitespace():
    decomposed = "क़"  # Devanagari ka + nukta
    assert normalize(f"  {decomposed}   x ") == normalize("क़ x")


def test_mask_pii():
    s = mask_pii("call 9841234567 or mail a.b@example.com @someone https://x.y/z")
    assert "9841234567" not in s and "example.com" not in s and "@someone" not in s
    assert "<PHONE>" in s and "<EMAIL>" in s and "<USER>" in s and "<URL>" in s


def test_stable_id_deterministic_and_opaque():
    a = stable_id("yt", "abc123")
    assert a == stable_id("yt", "abc123") and a != stable_id("yt", "abc124")
    assert "abc123" not in a and a.startswith("ns-")


def test_duplicates():
    assert dedup_key("Ramro  CHA") == dedup_key("ramro cha")
    pairs = near_duplicate_pairs(["phone ekdam ramro cha", "phone ekdam ramro cha!", "worst delivery ever"])
    assert pairs == [(0, 1)]


def _frame(n=300, groups=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"label": rng.choice(["negative", "neutral", "positive"], n, p=[.3, .3, .4]),
                         "group": rng.integers(0, groups, n)})


def test_split_groups_disjoint_and_reproducible():
    df = _frame()
    s1, s2 = grouped_split(df), grouped_split(df)
    assert s1.equals(s2)
    sets = {k: set(df.loc[s1 == k, "group"]) for k in ("train", "validation", "test")}
    assert not (sets["train"] & sets["test"]) and not (sets["train"] & sets["validation"])
    assert not (sets["validation"] & sets["test"])
    assert 0.1 < (s1 == "test").mean() < 0.3


def test_ids_hash_order_invariant():
    assert ids_hash(["b", "a"]) == ids_hash(["a", "b"])


def test_stratified_subsample():
    df = _frame(1000)
    sub = stratified_subsample(df, 100, seed=1)
    assert len(sub) == 100 and sub.index.is_unique
    full = df["label"].value_counts(normalize=True)
    assert (sub["label"].value_counts(normalize=True) - full).abs().max() < 0.02
    assert sub.index.equals(stratified_subsample(df, 100, seed=1).index)


def test_metrics_basic():
    y = ["positive", "negative", "neutral", "positive"]
    assert macro_f1(y, y) == 1.0
    r = report(y, ["positive"] * 4)
    assert r["per_class"]["negative"]["f1"] == 0.0 and r["accuracy"] == 0.5
    assert sum(map(sum, r["confusion_matrix"]["matrix"])) == 4


def test_bootstrap_and_paired():
    rng = np.random.default_rng(0)
    y = rng.choice(["negative", "neutral", "positive"], 400)
    good = np.where(rng.random(400) < 0.8, y, "neutral")
    bad = np.where(rng.random(400) < 0.4, y, "neutral")
    lo, hi = bootstrap_ci(y, good, n_boot=300)
    assert lo <= macro_f1(y, good) <= hi
    res = paired_bootstrap(y, good, bad, n_boot=300)
    assert res["diff"] > 0 and res["ci_low"] > 0 and res["p_value"] < 0.05
    same = paired_bootstrap(y, good, good, n_boot=300)
    assert same["diff"] == 0 and same["p_value"] == 1.0


def test_mcnemar_and_holm():
    y = np.array(["positive"] * 20)
    a = y.copy()
    b = np.array(["negative"] * 10 + ["positive"] * 10)
    r = mcnemar_exact(y, a, b)
    assert r["a_only_correct"] == 10 and r["b_only_correct"] == 0 and r["p_value"] < 0.01
    adj = holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adj["x"] == pytest.approx(0.03) and adj["z"] == pytest.approx(0.06) and adj["y"] == pytest.approx(0.06)


def test_agreement_kappa():
    a = ["positive", "negative", "neutral", "positive"]
    r = agreement(a, a)
    assert r["raw_agreement"] == 1.0 and r["cohen_kappa"] == 1.0


def test_fast_macro_f1_matches_sklearn():
    from nepsent.eval.metrics import fast_macro_f1
    rng = np.random.default_rng(3)
    labs = np.array(["negative", "neutral", "positive"])
    for _ in range(20):
        y, p = rng.choice(labs, 200), rng.choice(labs[:2], 200)
        assert fast_macro_f1(y, p) == pytest.approx(macro_f1(y, p))
