Test macro F1 on the grouped held-out test set ({n_test} tweets). Full training set, repeat 0, with 95% bootstrap intervals:

| Model | Macro F1 | 95% CI |
|---|---|---|
| Majority class | {full_majority} | {full_majority_ci} |
| Char n-gram TF-IDF + LR | {full_char_lr} | {full_char_lr_ci} |
| XLM-R base | {full_xlmr} | {full_xlmr_ci} |
| NepBERTa | {full_nepberta} | {full_nepberta_ci} |

Learning curve (mean over repeats):

| Training examples | 100 | 500 | 1,000 | 2,500 | 10,000 | {n_full} |
|---|---|---|---|---|---|---|
| Char n-gram LR | {char_lr_100} | {char_lr_500} | {char_lr_1000} | {char_lr_2500} | {char_lr_10000} | {char_lr_full} |
| XLM-R base | {xlmr_100} | {xlmr_500} | {xlmr_1000} | {xlmr_2500} | {xlmr_10000} | {xlmr_full} |
| NepBERTa | {nepberta_100} | {nepberta_500} | {nepberta_1000} | {nepberta_2500} | {nepberta_10000} | {nepberta_full} |

![Learning curves](outputs/figures/learning_curve.png)

Main findings:

- XLM-R outperforms the character model at every training size from 100 examples upward. Paired differences range from {xlmr_diff_min} to {xlmr_diff_max} macro F1, and all 95% intervals exclude zero after Holm correction. No size is found at which the character model is preferable.
- The advantage is largest at a few hundred to a few thousand examples and narrows to {xlmr_diff_full} at full size.
- The result holds when model selection uses a validation set scaled to the training budget.
- {readme_nepberta}
- A random split, which lets near-duplicate tweets cross the train/test boundary, raises the character model's macro F1 from {leak_grouped} to {leak_random}.
