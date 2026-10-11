# Data Governance

## Storage tiers

- `data/raw/`: the authors' archive and CSV. Not tracked.
- `data/interim/corpus.parquet`: cleaned text, labels, groups and splits. Not tracked.
- `data/processed/splits.csv`: row_id, text_sha1, group, split. Tracked. Contains no text and no labels.
- `data/processed/test_ids.sha256`: hash of the sorted test row ids, computed before any model was trained.
- `outputs/predictions/`: per-run test predictions (row_id, predicted label). Tracked. Gold labels are not included.

## Identifiers

`row_id` is the index column of the authors' release file. No tweet IDs, user handles or timestamps finer than those in the release are stored. Timestamps are not published.

## Release checklist

1. No text column in any tracked file (`tests/test_release.py`).
2. `scripts/check_pii.py` reports no findings.
3. `.gitignore` excludes `data/raw/` and `data/interim/`.
4. Every number in the README and report is produced by `scripts/analyze.py` or `scripts/prepare_data.py`.
