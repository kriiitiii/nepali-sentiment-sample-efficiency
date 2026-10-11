# Data

This repository does not redistribute NepCOV19Tweets text or labels. Obtain them from the dataset authors:

```
python scripts/download_data.py     # fetches and verifies the authors' Kaggle release
python scripts/prepare_data.py      # rebuilds data/interim/corpus.parquet and the split
```

## Tracked files

- `processed/splits.csv`: one row per cleaned tweet, with `row_id` (index column of the authors' file), `text_sha1` (SHA-1 of the NFC-normalised, whitespace-collapsed text), `group` (near-duplicate group) and `split` (train, validation or test). Rebuilding from the source should reproduce this file exactly. `tests/test_release.py` checks it.
- `processed/test_ids.sha256`: SHA-256 of the sorted test row ids, written before any model was trained.

## Untracked files

- `raw/`: the downloaded archive and CSV.
- `interim/corpus.parquet`: cleaned text with labels.

## Dataset documentation

- Source and terms: `docs/data_source_audit.md`
- Handling rules: `docs/data_governance.md`
- Cleaning counts: `outputs/metrics/data_audit.json`

No dataset card is published, because NepSent does not release a dataset. Users should cite Sitaula et al. (2021) for the data.
