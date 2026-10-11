# How Much Data Do Nepali Sentiment Models Need? Character Baselines versus Multilingual and Language-Specific Transformers

Kriti Dahal and Gaurab Khadka. Independent research project, 2026.

This project compares a tuned character n-gram logistic regression with fine-tuned transformers (XLM-R base and NepBERTa) for three-class Nepali tweet sentiment. The comparison runs across training-set sizes from 100 examples to the full training set, with all models trained on the same subsamples (paired), under a split that keeps near-duplicate tweets together.

The study analyses an existing dataset, NepCOV19Tweets, created and annotated by Sitaula et al. (2021). It is not a new benchmark, and it introduces no new data or labels. The error analysis is quantitative only; no manual qualitative coding of errors was performed.

## Research questions

1. As labeled training data grows, at what size, if any, does fine-tuned XLM-R reliably outperform a tuned character n-gram model?
2. Does a Nepali monolingual encoder (NepBERTa) change that answer?
3. How much does near-duplicate leakage under random splits inflate measured performance?

## Results

<!-- results:start -->
Test macro F1 on the grouped held-out test set (6,684 tweets). Full training set, repeat 0, with 95% bootstrap intervals:

| Model | Macro F1 | 95% CI |
|---|---|---|
| Majority class | 0.206 | 0.203 to 0.210 |
| Character n-gram logistic regression | 0.657 | 0.644 to 0.669 |
| XLM-R | 0.676 | 0.663 to 0.688 |
| NepBERTa | 0.684 | 0.672 to 0.696 |

Learning curve (mean over repeats):

| Training examples | 100 | 500 | 1,000 | 2,500 | 10,000 | 23,385 |
|---|---|---|---|---|---|---|
| Character n-gram logistic regression | 0.448 | 0.505 | 0.538 | 0.580 | 0.635 | 0.657 |
| XLM-R | 0.521 | 0.592 | 0.612 | 0.633 | 0.654 | 0.679 |
| NepBERTa | 0.500 | 0.570 | 0.588 | 0.619 | 0.651 | 0.679 |

![Test macro F1 by training set size for the character n-gram model, XLM-R and NepBERTa](outputs/figures/learning_curve.png)

Lines join the median over repeats; faint points are individual runs; the dotted line is the majority class baseline.

Main findings:

- XLM-R outperforms the character model at every training size from 100 examples upward. Paired differences range from +0.019 to +0.087 macro F1, and all 95% intervals exclude zero after Holm correction. No size is found at which the character model is preferable.
- The advantage is largest at a few hundred to a few thousand examples and narrows to +0.023 at full size.
- The result holds when model selection uses a validation set scaled to the training budget.
- NepBERTa also beats the character model at every size, but it trails XLM-R between 100 and 2,500 examples and is indistinguishable from it from 5,000 examples upward.
- A random split, which lets near-duplicate tweets cross the train/test boundary, raises the character model's macro F1 from 0.657 to 0.673.
<!-- results:end -->

The full write-up is in [report/report.pdf](report/report.pdf) ([Markdown version](report/report.md)).

## Data

NepCOV19Tweets is not redistributed here. `scripts/download_data.py` fetches it from the dataset authors' Kaggle record and checks its SHA-256 hash. The repository publishes only row identifiers, text hashes, near-duplicate groups and split assignments (`data/processed/splits.csv`), plus per-run test predictions without gold labels. See [docs/data_source_audit.md](docs/data_source_audit.md) and [docs/data_governance.md](docs/data_governance.md).

## Reproducing the study

Requirements: Python 3.12 and, for the transformer runs, a CUDA GPU (the reported runs used one 8 GB laptop GPU).

```
python -m venv .venv
.venv/Scripts/activate            # or: source .venv/bin/activate
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[transformers]"
pip install pyarrow markdown

python scripts/download_data.py            # data/raw/nepcov19tweets/
python scripts/prepare_data.py             # cleaning, near-duplicate groups, frozen split
python scripts/run_experiments.py --model majority
python scripts/run_experiments.py --model char_lr
bash scripts/run_transformer_pipeline.sh xlmr
python scripts/convert_nepberta.py models_local/nepberta-pt   # separate env, see script header
bash scripts/run_transformer_pipeline.sh nepberta
python scripts/run_experiments.py --model char_lr --split-mode random --sizes full --max-repeats 1
python scripts/run_experiments.py --model char_lr --val-mode proportional --sizes 100 250 500 1000
python scripts/run_experiments.py --model xlmr --lr 3e-5 --val-mode proportional --sizes 100 250 500 1000
python scripts/analyze.py
python scripts/error_analysis.py --model-pred outputs/predictions/xlmr_grouped_nfull_r0_lr3e-05.csv \
    --baseline-pred outputs/predictions/char_lr_grouped_nfull_r0.csv
python scripts/build_report.py
pytest
```

Completed runs are appended to `outputs/metrics/runs.jsonl`, and the experiment script skips runs that are already recorded. GPU fine-tuning is not bitwise deterministic, so reruns can differ slightly from the reported values. `requirements.lock.txt` records the exact package versions used.

## Repository layout

- `config/experiment.yaml`: seeds, split fractions, learning-curve sizes, hyperparameter grids and pinned model revisions
- `src/nepsent/`: cleaning, splitting, baselines, transformer fine-tuning, metrics and statistical tests
- `scripts/`: the pipeline steps listed above
- `outputs/metrics/`: machine-readable results (runs, learning curves, paired tests, data audit)
- `outputs/tables/`, `outputs/figures/`: generated tables and figures
- `outputs/predictions/`: per-run test predictions (row id and predicted label)
- `docs/`: literature review, novelty audit, dataset and model audits, protocol, analysis plan, deviations, governance, ethics
- `report/`: report template, generated report and abstract

## Citation

If you use this code or analysis, please also cite the dataset:

Chiranjibi Sitaula, Anish Basnet, Ashish Mainali, and Tej Bahadur Shahi. 2021. Deep Learning-Based Methods for Sentiment Analysis on Nepali COVID-19-Related Tweets. Computational Intelligence and Neuroscience, 2021, Article 2158184.

## License

The code is released under the MIT License. The dataset is subject to its authors' terms. NepBERTa is licensed CC-BY-NC-SA-4.0, and no fine-tuned weights are distributed.
