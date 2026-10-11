# Literature Review

## Nepali sentiment analysis on NepCOV19Tweets

Sitaula et al. (2021) released NepCOV19Tweets, a three-class corpus of Nepali tweets about COVID-19 annotated by four annotators with majority voting. They proposed CNNs over fastText, domain-specific and domain-agnostic features, evaluated over ten random 70/30 splits, and reported a best accuracy of 68.7% with an ensemble CNN. Shahi, Sitaula and Paudel (2022) combined TF-IDF and fastText features with nine classical classifiers. Further studies applied multi-channel CNNs and BERT-LSTM models to the same corpus. All of these use random splits, most report accuracy, and none examine how performance depends on training size.

## Code-mixed and Romanized Nepali

Pahari and Shimada (2023) annotated 10,702 code-switched Nepali-English YouTube comments (kappa 0.55). TF-IDF with an SVM reached macro F1 0.68, matching mBERT (0.68) and close to MuRIL (0.70), with XLM-R lower (0.65). Patel et al. (2026) showed that transliterating Romanized Nepali to Devanagari with NepaliXlit improves mBERT and MuRIL sentiment accuracy. NepEMO (Sitoula et al., 2025) covers emotions and sentiment in multi-script Reddit posts. NepSA (Singh et al., 2020) covers aspect-based abusive sentiment in YouTube comments. None of the code-mixed or Romanized datasets was available from an authoritative public source with clear terms at the time of this study (see `related_datasets.md`).

## Pretrained models for Nepali

XLM-R (Conneau et al., 2020) is pretrained on CC-100, which includes Nepali. NepBERTa (Timilsina, Gautam and Bhattarai, 2022) is a BERT model pretrained on about 0.8B words of Nepali news. Its authors also introduced the Nep-gLUE benchmark.

## Linear baselines versus transformers

Character n-gram linear models are robust to spelling variation and inexpensive to train. In several low-resource and code-mixed settings they remain competitive with fine-tuned transformers at moderate data sizes, as the Pahari and Shimada results illustrate. A learning-curve comparison, with paired subsamples and uncertainty estimates, shows directly where such a model stops being competitive. That comparison is the design used in this project.

## References

- Alexis Conneau et al. 2020. Unsupervised Cross-lingual Representation Learning at Scale. ACL 2020.
- Niraj Pahari and Kazutaka Shimada. 2023. Language Preference for Expression of Sentiment for Nepali-English Bilingual Speakers on Social Media. CALCS 2023. https://aclanthology.org/2023.calcs-1.3
- Suraj Patel, Kashish Kumari Dhami, Norden Sherpa, and Supriya Khadka. 2026. From Romanized to Devanagari: Enhancing Nepali Sentiment Analysis with NepaliXlit. CHiPSAL 2026. https://aclanthology.org/2026.chipsal-1.12/
- Tej Bahadur Shahi, Chiranjibi Sitaula, and Nabin Paudel. 2022. A Hybrid Feature Extraction Method for Nepali COVID-19-Related Tweets Classification. Computational Intelligence and Neuroscience. doi:10.1155/2022/5681574
- Oyesh Mann Singh, Sandesh Timilsina, Bal Krishna Bal, and Anupam Joshi. 2020. Aspect Based Abusive Sentiment Detection in Nepali Social Media Texts. IEEE/ACM ASONAM 2020.
- Chiranjibi Sitaula, Anish Basnet, Ashish Mainali, and Tej Bahadur Shahi. 2021. Deep Learning-Based Methods for Sentiment Analysis on Nepali COVID-19-Related Tweets. Computational Intelligence and Neuroscience. doi:10.1155/2021/2158184
- Sameer Sitoula, Tej Bahadur Shahi, Laxmi Prasad Bhatt, Anisha Pokhrel, and Arjun Neupane. 2025. NepEMO: A Multi-Label Emotion and Sentiment Analysis on Nepali Reddit with Linguistic Insights and Temporal Trends. arXiv:2512.22823.
- Sulav Timilsina, Milan Gautam, and Binod Bhattarai. 2022. NepBERTa: Nepali Language Model Trained in a Large Corpus. AACL-IJCNLP 2022.
