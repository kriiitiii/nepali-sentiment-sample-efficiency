"""Checks on the released split files and on release hygiene."""
from pathlib import Path

import pandas as pd
import pytest

from nepsent.data.split import ids_hash

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/processed/splits.csv"


@pytest.mark.skipif(not SPLITS.exists(), reason="splits not generated")
def test_splits_have_no_text_and_groups_do_not_cross():
    df = pd.read_csv(SPLITS)
    assert set(df.columns) == {"row_id", "text_sha1", "group", "split"}
    assert df.row_id.is_unique and df.text_sha1.is_unique
    assert (df.groupby("group").split.nunique() == 1).all()
    assert set(df.split) == {"train", "validation", "test"}


@pytest.mark.skipif(not SPLITS.exists(), reason="splits not generated")
def test_test_set_hash_unchanged():
    df = pd.read_csv(SPLITS)
    recorded = (ROOT / "data/processed/test_ids.sha256").read_text().strip()
    assert ids_hash(df.loc[df.split == "test", "row_id"]) == recorded


def test_predictions_contain_no_text_or_labels():
    for f in (ROOT / "outputs/predictions").glob("*.csv"):
        cols = pd.read_csv(f, nrows=1).columns
        assert list(cols) == ["row_id", "pred"], f


def test_gitignore_excludes_restricted_data():
    gi = (ROOT / ".gitignore").read_text()
    for entry in ("data/raw/*", "data/interim/*", ".venv/", ".env"):
        assert entry in gi
