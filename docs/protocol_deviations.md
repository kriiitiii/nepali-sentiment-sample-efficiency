# Protocol Deviations and Sensitivity Checks

## 2026-10-11: character n-gram range at grid boundary

The full grid selected n-gram range (1, 6), the largest range in the grid, with validation macro F1 0.669. Since a value on the grid boundary can indicate an undertuned baseline, longer ranges were checked on the validation set only, with the other selected settings held fixed (`outputs/metrics/char_lr_ngram_extension.json`):

| n-gram range | Validation macro F1 |
|---|---|
| (1, 6), selected | 0.669 |
| (1, 7) | 0.668 |
| (2, 7) | 0.671 |
| (1, 8) | 0.675 |

The differences are at most 0.006 and do not increase steadily with range. They are smaller than the sampling variability expected on a 3,342-item validation set. The pre-specified setting (1, 6) was kept. The test set was not consulted for this decision.

## 2026-10-11: interrupted runs

The host machine entered sleep during the first learning-curve run. That run's wall-clock time (`seconds` in runs.jsonl) is inflated, but its training and results are unaffected.

## 2026-10-11: post-hoc sensitivity analysis on validation-set size

This analysis was added after partial learning-curve results were seen, and it is reported as post hoc. In the pre-specified design, every learning-curve cell uses the full validation set (3,342 items) for model selection: early stopping and checkpoint choice for transformers, and C and class weight for the character model. At small training sizes, this gives both models more labeled selection data than a practitioner with that budget would have, and it may favour the transformer, whose checkpoint selection depends more on validation feedback. The sensitivity runs repeat sizes 100 to 1,000 for XLM-R and the character model, with selection on a stratified validation subsample of max(30, n/7) items. That size matches the 70/10 train/validation ratio. The test set and training subsamples are unchanged, so these runs are paired with the main runs. Run tag: `grouped_propval`.

## 2026-10-11: leakage run hyperparameters

To avoid repeating the eight-hour full grid, the random-split character model reuses the n-gram settings selected on the grouped split. C and class weight are re-selected on the random split's own validation set.

## 2026-10-11: NepBERTa checkpoint format

The NepBERTa Hub repository (revision 1b8293fd5ed883081619e9895540c58eb740e7d0) provides only a TensorFlow checkpoint (`tf_model.h5`), which Transformers 5 cannot load. The checkpoint was converted to PyTorch with the Transformers 4.44.2 TF-to-PyTorch loader (`scripts/convert_nepberta.py`). On three Nepali sentences, the masked-LM logits of the TensorFlow and PyTorch models differ by at most 2.0e-5, and their top predictions agree at every position (`outputs/metrics/nepberta_conversion.json`). The converted tokenizer reproduces the original tokenization. No weights were modified.
