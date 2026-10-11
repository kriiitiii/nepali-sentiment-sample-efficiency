# Model Audit

Hugging Face metadata was checked on 2026-10-10. Revisions are pinned in `config/experiment.yaml`.

| Model | Hub ID | Revision | License | Nepali relevance | Role |
|---|---|---|---|---|---|
| XLM-R base | FacebookAI/xlm-roberta-base | e73636d4f797dec63c3081bb6ed5c7b0bb3f2089 | MIT | CC-100 pretraining includes Nepali | Primary transformer (RQ1) |
| NepBERTa | NepBERTa/NepBERTa | 1b8293fd5ed883081619e9895540c58eb740e7d0 | CC-BY-NC-SA-4.0 | BERT pretrained on about 0.8B words of Nepali news (Timilsina et al., 2022) | Monolingual comparison (RQ2) |

## Not used

- mBERT and MuRIL: neither adds a hypothesis beyond RQ1 and RQ2 on monolingual Devanagari data.
- Generative LLMs: outside the scope of a fine-tuned encoder versus linear baseline comparison.

## Tokenization notes

- Maximum token length over a 2,000-text sample: XLM-R 107 and NepBERTa 92. The median is about 30 for both. A maximum length of 128 therefore truncates nothing.
- The NepBERTa tokenizer applies BERT accent stripping. Unicode decomposition followed by removal of combining marks deletes Devanagari dependent vowel signs. For example, "अमेरिकामा" is tokenized as "अमर ##िकामा", which loses the vowel sign "े". The tokenizer is used exactly as published, presumably as it was used in pretraining. Some orthographic information is lost before the model sees the text, and this may affect NepBERTa's results.

## Licensing consequences

NepBERTa is licensed CC-BY-NC-SA-4.0. No fine-tuned weights are distributed by this project. Only predictions and metrics are released.

## Hardware and software

All runs used one NVIDIA GeForce RTX 4070 Laptop GPU (8 GB) with mixed precision (fp16), PyTorch 2.6.0+cu124, Transformers 5.19.0 and Python 3.12.10. Each run's metadata is stored in `outputs/metrics/runs.jsonl`. GPU training is not bitwise deterministic. Repeated runs under the same seed can differ slightly, and variance is reported across repeats.
