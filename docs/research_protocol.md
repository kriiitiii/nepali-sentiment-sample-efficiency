# Research Protocol

Fixed on 2026-10-11, after the data audit and split generation and before any model results were inspected. Deviations are logged in `docs/protocol_deviations.md`.

## Research questions

- RQ1 (primary). As the number of labeled training examples grows from 100 to the full training set, at what size, if any, does fine-tuned XLM-R base reliably outperform a character n-gram TF-IDF logistic regression tuned on the same data?
- RQ2. Does a Nepali monolingual encoder (NepBERTa) change the answer to RQ1?
- RQ3. How much does a random split, which lets near-duplicate tweets cross the train/test boundary, inflate measured performance compared with a grouped split?

## Data

NepCOV19Tweets (Sitaula et al., 2021), obtained from the authors' release (see `data_source_audit.md`).

## Cleaning (scripts/prepare_data.py)

1. Keep rows whose label is -1, 0 or 1 and whose record parses as the expected four fields. Malformed rows, caused by broken quoting in the release, are dropped and counted.
2. Apply Unicode NFC normalisation and collapse whitespace.
3. Exact duplicate texts: drop all copies if their labels conflict; otherwise keep one copy.
4. Near duplicates: texts with character 3-5 gram TF-IDF cosine at or above 0.85 are linked, and connected components form groups. The threshold was set by manual inspection of clusters: at 0.85, the linked pairs were templated case-count updates and the same headline with or without a suffix.

## Split

Stratified grouped split by sentiment label and near-duplicate group, approximately 70/10/20 train/validation/test, seed 13. The test row ids are hashed into `data/processed/test_ids.sha256`. The test set is used only by the final evaluation of each run, never for model selection.

## Models

- Majority class.
- Character n-gram TF-IDF + logistic regression. A full grid (analyzer, n-gram range, min_df, C, class weight; 128 configurations) is searched once on the full training set against validation macro F1. At every learning-curve cell, C and class weight are re-selected on validation, with the selected n-gram settings fixed.
- XLM-R base (FacebookAI/xlm-roberta-base, pinned revision). The learning rate is selected from {1e-5, 2e-5, 3e-5} by validation macro F1 at full size, repeat 0, and then held fixed. Batch size 32, maximum length 128 (no text exceeds this), AdamW with weight decay 0.01, linear warmup over 10% of steps. Training lasts max(600 steps, 4 epochs), with 10 evaluations per run and early stopping (patience 3) on validation macro F1. The best checkpoint is restored.
- NepBERTa (pinned revision), with the same procedure and its own learning-rate selection.

## Learning curve

- Training sizes: 100, 250, 500, 1,000, 2,500, 5,000, 10,000 and the full training set.
- Repeats: 5 for sizes up to 2,500 and 3 above. Each (size, repeat) cell draws one label-stratified subsample with seed 1000 x repeat + size. All models in a cell see the same subsample, so comparisons are paired.

## Metrics

The primary metric is test macro F1. Secondary metrics are per-class precision, recall and F1, accuracy and confusion matrices.
