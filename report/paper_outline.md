# Paper Outline

Target: a 4-page short paper or workshop paper on low-resource or South Asian language processing. The current technical report (`report.md`) is the long version.

1. Introduction: the practitioner's question of a cheap character model versus fine-tuning at a given labeling budget. State the three research questions and three contributions.
2. Related work: Nepali sentiment work on NepCOV19Tweets (random splits, mostly accuracy); code-mixed Nepali, where TF-IDF is competitive; NepBERTa and XLM-R.
3. Data and split: cleaning counts, near-duplicate grouping, and the frozen grouped split.
4. Setup: character model tuning, transformer protocol, learning-curve design, paired bootstrap and the crossover definition.
5. Results:
   - Figure 1: learning curves.
   - Figure 2: paired differences.
   - Table 1: full-size comparison.
   - Table 2: paired differences by size.
6. Analyses: validation-size sensitivity; NepBERTa versus XLM-R; near-duplicate leakage; quantitative error overlap.
7. Discussion and limitations: single corpus and script, label noise, post-hoc sensitivity, learning-rate boundary.
8. Ethics: no redistribution of tweets, no new data, no released weights.

Work needed before submission:

- Optionally, manual qualitative error coding by fluent Nepali readers (not part of the present study).
- Optionally, a higher learning-rate check (5e-5) for both transformers, because 3e-5 sat at the edge of the searched range.
- Optionally, a second corpus with Romanized or code-mixed text, if one becomes available with clear terms. This would test whether the conclusion holds outside formal Devanagari text.
