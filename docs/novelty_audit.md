# Novelty Audit

## Prior work on NepCOV19Tweets

- Sitaula et al. (2021) introduced the dataset. They evaluate CNNs on fastText, domain-specific and domain-agnostic features, using ten random 70/30 train/test splits and no separate validation set. The best reported accuracy is 68.7% (ensemble CNN).
- Shahi, Sitaula and Paudel (2022) combine TF-IDF and fastText features with nine classical classifiers.
- A multi-channel CNN study on hybrid features (arXiv 2203.10286).
- Sentiment Analysis of Nepali COVID-19 Tweets using BERT-LSTM (Journal of Engineering and Sciences, NepJOL), which reports average accuracy 76.04% and F1 76%.
- Unreviewed GitHub experiments report transformer F1 scores without stating the averaging method.

## What has not been done (as far as this search found)

1. A learning-curve comparison of a tuned character n-gram linear model and a fine-tuned multilingual transformer on this dataset, with repeated subsampling, paired uncertainty, and an explicit crossover criterion.
2. Evaluation under a split that keeps near-duplicate tweets together. The dataset contains many templated news updates and reposted headlines, and random splits place near copies in both train and test.
3. A paired comparison of a multilingual model (XLM-R) and a Nepali monolingual model (NepBERTa) on identical training subsamples.

## Central contribution

A sample-efficiency study: on three-class Nepali tweet sentiment, at what training size, if any, does fine-tuning XLM-R reliably outperform a character n-gram logistic regression tuned on the same data? This is reported with uncertainty under a leakage-controlled split.

Secondary analyses:

- the effect of near-duplicate leakage on measured performance
- whether Nepali-specific pretraining (NepBERTa) changes the comparison
- error patterns of the strongest models

## Not claimed

- No new dataset is created. All labels come from Sitaula et al. (2021).
- No consumer-domain, code-mixed, script or transliteration claims. The data are monolingual Devanagari news-style tweets.
- No state-of-the-art claim. Earlier papers use different splits and mostly report accuracy, so their numbers are not directly comparable.
