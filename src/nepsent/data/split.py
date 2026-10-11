"""Grouped, sentiment-stratified train/validation/test split."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def grouped_split(df: pd.DataFrame, label_col="label", group_col="group",
                  test_frac=0.2, val_frac=0.1, seed=13) -> pd.Series:
    """Return a Series of 'train'/'validation'/'test' aligned with df.index.

    Groups never cross splits. Fractions are approximate because whole
    groups are assigned.
    """
    split = pd.Series("train", index=df.index)
    k_test = max(2, round(1 / test_frac))
    sgkf = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(df, df[label_col], df[group_col]))
    split.iloc[test_idx] = "test"
    rest = df[split == "train"]
    k_val = max(2, round((1 - test_frac) / val_frac))
    sgkf = StratifiedGroupKFold(n_splits=k_val, shuffle=True, random_state=seed + 1)
    _, val_idx = next(sgkf.split(rest, rest[label_col], rest[group_col]))
    split.loc[rest.index[val_idx]] = "validation"
    return split


def ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def stratified_subsample(df: pd.DataFrame, n: int, label_col="label", seed=0) -> pd.DataFrame:
    """Draw n rows preserving label proportions (largest-remainder rounding)."""
    if n >= len(df):
        return df.copy()
    rng = np.random.default_rng(seed)
    props = df[label_col].value_counts(normalize=True)
    raw = props * n
    counts = np.floor(raw).astype(int)
    for lab in (raw - counts).sort_values(ascending=False).index[: n - counts.sum()]:
        counts[lab] += 1
    parts = []
    for lab, c in counts.items():
        pool = df.index[df[label_col] == lab].to_numpy()
        parts.append(rng.choice(pool, size=min(c, len(pool)), replace=False))
    return df.loc[np.concatenate(parts)]
