# Related Datasets

Audit date: 2026-10-10 to 2026-10-11. Only information verified from the cited source is recorded. "Not verified" means the information was not found; it does not mean it is absent.

## Selected: NepCOV19Tweets

- Source: Kaggle dataset `mathew11111/nepcov19tweets`, uploaded on 2021-10-20 by Anish Basnet, a co-author of the dataset paper. Version 1. This is the authors' own release, not a third-party mirror.
- Paper: Chiranjibi Sitaula, Anish Basnet, Ashish Mainali, and Tej Bahadur Shahi. 2021. Deep Learning-Based Methods for Sentiment Analysis on Nepali COVID-19-Related Tweets. Computational Intelligence and Neuroscience, 2021. Open access under CC BY. https://pmc.ncbi.nlm.nih.gov/articles/PMC8561567/
- Download: https://www.kaggle.com/datasets/mathew11111/nepcov19tweets. The public download endpoint did not require a login when tested.
- Size: 33,474 records in the release file. The paper reports 33,435 raw and 33,247 clean tweets.
- Labels: negative (-1), neutral (0), positive (1)
- Language and script: Nepali in Devanagari. In the released text, Latin characters, digits, punctuation, URLs and mentions are absent, consistent with the authors' preprocessing.
- Romanized content: none. Code mixing: none in the released text.
- Domain: tweets geolocated to Nepal that match the Nepali COVID-19 hashtag, from 2020-02-11 to 2021-01-10. Many are news-style headlines.
- Annotation: four annotators (the co-authors) with majority voting. No agreement statistic is reported.
- License: the Kaggle license field reads "Other (specified in description)". The description asks users to cite the paper and states no other restriction.
- Redistribution: no explicit grant. The text originates from Twitter. NepSent therefore does not redistribute text or labels; it releases only row identifiers, text hashes and split assignments.
- Stable access: an author-maintained Kaggle record, unchanged since 2021.

## Investigated and not selected

| Dataset | Source | Size | Labels | Script | License / access | Reason not selected |
|---|---|---|---|---|---|---|
| Pahari and Shimada (2023), code-switched YouTube comments | ACL Anthology 2023.calcs-1.3 | 10,702 | pos/neu/neg | mostly Romanized, code-mixed | Paper says "released publicly"; no download location or license found in the paper, the Anthology page or the authors' repository record | Not obtainable without contacting the authors |
| NepEMO (Sitoula et al., 2025) | arXiv 2512.22823 | 4,462 | 5 emotions + pos/neu/neg | Devanagari, Romanized, English | The repository linked in the paper (github.com/Sameer67/Nepali-Reddit-NepEMO-) returned 404 on 2026-10-11 | Not accessible |
| NepaliXlit sentiment set (Patel et al., 2026) | ACL Anthology 2026.chipsal-1.12 | 1,518 annotated | not verified | Romanized | No release found | Not accessible |
| NepSA (Singh et al., 2020) | github.com/oya163/nepali-sentiment-analysis | 3,068 comments (README) | binary polarity per aspect, 6 aspect categories | Devanagari, some Latin | Repository under MIT License; text derived from YouTube comments | Aspect-level binary task, small, news and politics abuse domain; does not fit three-class learning curves |
| Sentiment_Analysis_Benchmark | huggingface.co/datasets/jinunyachhyon | 65,108 train / 16,277 test | binary | Devanagari | Card tag CC BY 4.0; no provenance statement | Inspection shows mostly machine-translated English movie-review sentences mixed with news headlines; not native user text and no documented provenance |
| IRIIS-RESEARCH/Sentiment-Analysis-Nepali | Hugging Face | 65,106 / 16,279 | binary | Devanagari | No license, no card text | Same size as the previous entry, no provenance |
| Shushant/NepaliSentiment, Titung/nepali-sentiment, Ajhesh7/nepali_sentiment_dataset, raygx/NepCov19Tweets and similar | Hugging Face | various | various | various | No license or no provenance; raygx copies are third-party mirrors of NepCOV19Tweets | Unclear permission, or the authoritative source exists |
| ENeMeme (Thapa et al.) | github.com/therealthapa/crossplatform | 4,211 memes | sentiment and hate | code-mixed | The repository contains only a README | No data present |
| Individual GitHub projects (Romanized sentiment, product reviews) | various | various | various | various | No license or no stated provenance | Unclear permission |

## Summary

NepCOV19Tweets is the only candidate that combines real user text, three-class human labels, an author-operated public download, explicit usage terms (citation) and enough examples for learning curves. It is monolingual and single-script, so questions about script variation and transliteration cannot be studied with it.
