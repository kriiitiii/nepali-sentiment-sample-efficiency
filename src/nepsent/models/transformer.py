"""Fine-tuning of encoder models for three-class sentiment.

Requires the optional transformer dependencies (see pyproject.toml). GPU
training is not bitwise deterministic, so variance is handled by repeating
runs over seeds and training subsamples.
"""
from __future__ import annotations

import math
import platform
import shutil
import tempfile

from nepsent.data.clean import LABELS


def schedule(n_train: int, batch_size: int, min_steps: int, max_epochs: int, evals_per_run: int):
    """Training length rule used for every training size.

    Small training sets are trained for at least `min_steps` optimizer steps;
    large ones for `max_epochs` epochs. Evaluation happens `evals_per_run`
    times, and early stopping acts on validation macro F1.
    """
    steps_per_epoch = math.ceil(n_train / batch_size)
    max_steps = max(min_steps, max_epochs * steps_per_epoch)
    eval_steps = max(1, max_steps // evals_per_run)
    return max_steps, eval_steps


def finetune_predict(train_texts, train_labels, val_texts, val_labels, test_texts, model_name, revision,
                     seed=13, lr=2e-5, batch_size=32, max_length=128, min_steps=600, max_epochs=4,
                     evals_per_run=10, patience=3, weight_decay=0.01, warmup_ratio=0.1):
    """Fine-tune, select the best checkpoint on validation macro F1, predict test.

    Returns (val_predictions, test_predictions, metadata).
    """
    import numpy as np
    import torch
    import transformers
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

    from nepsent.eval.metrics import macro_f1

    set_seed(seed)
    l2i = {lab: i for i, lab in enumerate(LABELS)}
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=revision, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id=l2i)

    def enc(texts, labels=None):
        labels = labels if labels is not None else [LABELS[0]] * len(texts)
        ds = Dataset.from_dict({"text": list(texts), "label": [l2i[x] for x in labels]})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_length),
                      batched=True, remove_columns=["text"])

    def compute(p):
        pred = np.argmax(p.predictions, axis=-1)
        return {"macro_f1": macro_f1([LABELS[i] for i in p.label_ids], [LABELS[i] for i in pred])}

    max_steps, eval_steps = schedule(len(train_texts), batch_size, min_steps, max_epochs, evals_per_run)
    out_dir = tempfile.mkdtemp(prefix="nepsent_")
    args = TrainingArguments(
        output_dir=out_dir, learning_rate=lr, per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=128, max_steps=max_steps, weight_decay=weight_decay,
        warmup_steps=int(warmup_ratio * max_steps), eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps",
        save_steps=eval_steps, load_best_model_at_end=True, metric_for_best_model="macro_f1",
        save_total_limit=1, seed=seed, data_seed=seed, report_to=[], logging_strategy="no",
        fp16=torch.cuda.is_available(), disable_tqdm=True)
    trainer = Trainer(model=model, args=args, train_dataset=enc(train_texts, train_labels),
                      eval_dataset=enc(val_texts, val_labels), data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=compute, callbacks=[EarlyStoppingCallback(patience)])
    trainer.train()

    def predict(texts):
        out = trainer.predict(enc(texts))
        return [LABELS[i] for i in np.argmax(out.predictions, axis=-1)]

    val_pred, test_pred = predict(val_texts), predict(test_texts)
    meta = {"model": model_name, "revision": revision, "seed": seed, "lr": lr, "batch_size": batch_size,
            "max_length": max_length, "max_steps": max_steps, "eval_steps": eval_steps,
            "best_step": int(trainer.state.best_global_step or -1) if hasattr(trainer.state, "best_global_step") else None,
            "steps_run": int(trainer.state.global_step), "patience": patience,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    del trainer, model
    torch.cuda.empty_cache()
    shutil.rmtree(out_dir, ignore_errors=True)
    return val_pred, test_pred, meta
