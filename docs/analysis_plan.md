# Analysis Plan

## Learning-curve summaries

For each model and size, report the mean, standard deviation, minimum and maximum of test macro F1 across repeats. Also report a 95% test-sampling interval for the mean. To compute it, test items are resampled with a bootstrap (500 resamples), and the same resampled indices are applied to every repeat.

## RQ1 and RQ2: paired difference by size

- Quantity: the mean over repeats of [macro F1(transformer) minus macro F1(char LR)], where both models in each repeat were trained on the same subsample.
- Uncertainty: a paired bootstrap over test items (2,000 resamples, the same indices for both models and all repeats), giving a 95% percentile interval.
- Hypothesis per size: H0 is that the mean paired difference is 0. The two-sided p-value comes from the centred bootstrap distribution. Holm-Bonferroni correction is applied across sizes, within each transformer.
- The number of repeats in which the transformer is better is also reported. With three to five repeats, between-subsample variance is described rather than modelled, and the interval reflects test-sampling uncertainty only.
- Crossover definition: the smallest training size at which the interval lies entirely above 0 and stays above 0 at every larger size. If no size satisfies this, no crossover is claimed.

## Full-size comparison

On repeat 0 at full size, report every model's macro F1 with a 2,000-resample bootstrap interval. Pairwise paired-bootstrap tests on macro F1 are Holm-corrected. The exact McNemar test is reported as a secondary test of H0: the two models make an equal number of discordant correct predictions. It concerns accuracy, not macro F1.

## RQ3: leakage

The character model is trained at full size on an ungrouped, label-stratified random split with the same proportions. Its test macro F1 is reported next to the grouped result. The two test sets differ, so this is a descriptive comparison and no paired test is applied.

## Error analysis

`scripts/error_analysis.py` computes macro F1 by text length, the overlap of errors between XLM-R and the character model (including how often both predict the same wrong label), and XLM-R's confusion patterns. The analysis is quantitative only. Manual qualitative coding of error categories was considered but not carried out.
