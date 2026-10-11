# Data Source Audit

Audit date: 2026-10-11. This is not legal advice.

## Source

NepCOV19Tweets, obtained from the dataset authors' Kaggle record `mathew11111/nepcov19tweets` (owner: Anish Basnet, co-author of Sitaula et al. 2021).

- Download: `https://www.kaggle.com/api/v1/datasets/download/mathew11111/nepcov19tweets` (public; no credentials used)
- Archive SHA-256: `eba73b8ca8bf359d1b39bdfa3e9d9a7d72f8e21602f2c762163d3234ba484851`
- Extracted file: `covid19_tweeter_dataset.csv`, 22,784,816 bytes, SHA-256 recorded in `outputs/metrics/data_audit.json`
- Columns: index, Label, Datetime, Tweet, Tokanize_tweet

## Terms

- The Kaggle license field reads "Other (specified in description)". The description requests citation of Sitaula et al. (2021) and contains no other terms.
- The paper is open access under CC BY. Its data availability statement says data are available from the corresponding author on request, while the introduction calls the dataset public. The later Kaggle release by a co-author is treated as the authors' public release.
- Research use with citation is permitted by the authors' stated terms. Redistribution is not explicitly granted. The text also originates from Twitter, whose terms restrict redistributing tweet content.

## Decisions

- Raw and cleaned text are kept locally in `data/raw/` and `data/interim/` and are never committed.
- Published artifacts contain only the authors' row index (`row_id`), a SHA-1 of the cleaned text, the near-duplicate group and the split. Labels are not published; users obtain them from the authors' release.
- Test predictions are published as row_id and predicted label only. Gold labels are joined from the user's own download of the source file when metrics are recomputed.
- No tweet IDs, usernames or user metadata are present in the release, and none are added.
- The report neither quotes nor paraphrases individual tweets.

## Reproduction

`scripts/download_data.py` fetches the archive, checks its hash and extracts it into `data/raw/nepcov19tweets/`.
