# Portfolio Summary

## CV entry

NepSent: Sample efficiency of character n-gram models and transformers for Nepali sentiment. Independent research project, 2026.

- Designed a pre-specified, paired learning-curve study on NepCOV19Tweets (33,411 tweets): 100 to 23,385 training examples, repeated subsampling, and paired bootstrap tests with Holm correction, under a near-duplicate-grouped split.
- Found that fine-tuned XLM-R beats a tuned character n-gram model at every training size (+0.019 to +0.087 macro F1). A Nepali-only encoder (NepBERTa) was less sample-efficient than XLM-R at most sizes below 5,000 examples, and random splits inflated scores through near-duplicate leakage.
- Built a reproducible pipeline (data audit, frozen hashed test set, 181 logged runs, generated tables, figures and report) with automated tests and no redistribution of restricted text.

## Research summary

The project asks when, as labeled data grows, a fine-tuned multilingual transformer reliably outperforms a strong character n-gram baseline for Nepali sentiment. An audit of public Nepali sentiment data found that most candidate datasets lacked clear terms or an authoritative source. The study therefore analyses NepCOV19Tweets, released by its authors with a citation request. Under a split that keeps near-duplicate tweets together, XLM-R was better at every size from 100 examples up. The gap was largest between a few hundred and a few thousand examples. The result held when model selection used a realistically small validation set. NepBERTa offered no sample-efficiency gain over XLM-R.

## Portfolio description (about 100 words)

NepSent is a reproducible study of how much labeled data a transformer needs to beat a simple character n-gram model for Nepali sentiment classification. Using the public NepCOV19Tweets corpus, it trains a tuned logistic regression, XLM-R and NepBERTa on identical subsamples, from 100 to over 23,000 examples. The evaluation uses a leakage-controlled split and paired statistical tests. XLM-R led at every size, most strongly in the few-hundred to few-thousand range, and Nepali-specific pretraining did not improve data efficiency. The repository includes the data audit, protocol, deviations log, all run logs and predictions, generated figures, and a technical report.

## Short description for faculty outreach

I ran a controlled learning-curve study comparing character n-gram models with fine-tuned transformers for Nepali sentiment. All models were trained on paired subsamples from 100 to 23,000 examples, and the test split kept near-duplicate tweets together. XLM-R outperformed the linear baseline at every size, and a Nepali-only model was not more sample-efficient. Random splits on this corpus inflate scores through near-duplicate leakage. I am interested in extending this to Romanized and code-mixed Nepali, where earlier work suggests simple models remain competitive.

## Technical skills demonstrated

- Experimental design: pre-registered protocol, frozen and hashed test set, paired learning curves, a documented deviations log
- Statistics: bootstrap confidence intervals, paired bootstrap tests, McNemar's test, Holm-Bonferroni correction
- NLP: character n-gram TF-IDF models, fine-tuning XLM-R and BERT models with Hugging Face Transformers, TensorFlow-to-PyTorch checkpoint conversion with a parity check
- Data work: provenance and license audit, near-duplicate detection, leakage-controlled grouped splitting, no redistribution of restricted text
- Engineering: reproducible pipeline, configuration files, pytest suite, generated tables, figures and PDF report
