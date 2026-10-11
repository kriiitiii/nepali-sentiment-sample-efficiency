"""Learning-curve experiments on the frozen split.

Every (size, repeat) cell draws one stratified training subsample. All
models in that cell are trained on exactly the same subsample, evaluated on
the same validation set (for model selection) and the same test set.

Usage:
    python scripts/run_experiments.py --model char_lr
    python scripts/run_experiments.py --model xlmr --lr 2e-5
    python scripts/run_experiments.py --model char_lr --split-mode random   # leakage analysis

Each completed run appends one JSON line to outputs/metrics/runs.jsonl and
writes test predictions (row ids and predicted labels only) to outputs/predictions/.
Completed runs are skipped, so the script can be resumed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from nepsent.data.split import stratified_subsample
from nepsent.eval.metrics import macro_f1, report
from nepsent.models.baselines import char_lr, majority, tune_char_lr

RUNS = Path("outputs/metrics/runs.jsonl")
PRED_DIR = Path("outputs/predictions")


def load(split_mode: str, seed: int):
    df = pd.read_parquet("data/interim/corpus.parquet")
    if split_mode == "random":
        # Ungrouped split of the same proportions, used only for the leakage analysis.
        from sklearn.model_selection import train_test_split
        rest, test = train_test_split(df, test_size=(df["split"] == "test").mean(),
                                      stratify=df["label"], random_state=seed)
        train, val = train_test_split(rest, test_size=(df["split"] == "validation").sum() / len(rest),
                                      stratify=rest["label"], random_state=seed)
        return train, val, test
    return (df[df["split"] == s] for s in ("train", "validation", "test"))


def done_keys():
    if not RUNS.exists():
        return set()
    return {(r["model"], r["split_mode"], r["size"], r["repeat"], r.get("lr"))
            for r in map(json.loads, RUNS.read_text(encoding="utf-8").splitlines())}


def cells(cfg, n_train):
    lc = cfg["learning_curve"]
    for size in lc["sizes"]:
        n = n_train if size == "full" else int(size)
        reps = lc["repeats_small"] if n <= 2500 else lc["repeats_large"]
        for r in range(reps):
            yield size, n, r


def grid_from(cfg_grid):
    g = dict(cfg_grid)
    if "ngram_range" in g:
        g["ngram_range"] = [tuple(x) for x in g["ngram_range"]]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["majority", "char_lr", "xlmr", "nepberta"])
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--split-mode", default="grouped", choices=["grouped", "random"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--sizes", nargs="*", default=None, help="restrict to these sizes")
    ap.add_argument("--max-repeats", type=int, default=None)
    ap.add_argument("--val-mode", default="full", choices=["full", "proportional"],
                    help="proportional: select models on a validation subsample of size max(30, n/7)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    train, val, test = load(args.split_mode, cfg["seed"])
    if args.sizes:
        cfg["learning_curve"]["sizes"] = [s if s == "full" else int(s) for s in args.sizes]
    done = done_keys()
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # Character model: full grid once on the full training set, then the
    # reduced grid (C, class weight) at each cell with the selected n-gram settings.
    fixed = {}
    if args.model == "char_lr":
        sel = Path(f"outputs/metrics/char_lr_full_grid_{args.split_mode}.json")
        if sel.exists():
            fixed = json.loads(sel.read_text())["best_params"]
        else:
            t0 = time.time()
            _, params, score = tune_char_lr(train["text"], train["label"], val["text"], val["label"],
                                            grid=grid_from(cfg["char_lr"]["full_grid"]), seed=cfg["seed"])
            fixed = {k: v for k, v in params.items() if k not in cfg["char_lr"]["curve_grid"]}
            sel.write_text(json.dumps({"best_params": params, "val_macro_f1": score,
                                       "seconds": time.time() - t0}, indent=2, default=list))
            fixed = json.loads(sel.read_text())["best_params"]
        fixed = {k: (tuple(v) if k == "ngram_range" else v) for k, v in fixed.items()
                 if k not in cfg["char_lr"]["curve_grid"]}

    for size, n, rep in cells(cfg, len(train)):
        if args.max_repeats is not None and rep >= args.max_repeats:
            continue
        mode = args.split_mode if args.val_mode == "full" else f"{args.split_mode}_propval"
        key = (args.model, mode, str(size), rep, args.lr)
        if key in done:
            continue
        sub_seed = 1000 * rep + n
        sub = stratified_subsample(train, n, seed=sub_seed)
        vsel = val if args.val_mode == "full" else stratified_subsample(val, max(30, round(n / 7)), seed=sub_seed)
        t0 = time.time()
        meta = {}
        if args.model == "majority":
            m = majority().fit(sub["text"], sub["label"])
            val_pred, test_pred = m.predict(val["text"]), m.predict(test["text"])
        elif args.model == "char_lr":
            grid = {**{k: [v] for k, v in fixed.items()}, **cfg["char_lr"]["curve_grid"]}
            m, params, _ = tune_char_lr(sub["text"], sub["label"], vsel["text"], vsel["label"],
                                        grid=grid, seed=cfg["seed"])
            val_pred, test_pred = m.predict(vsel["text"]), m.predict(test["text"])
            meta = {"params": {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}}
        else:
            from nepsent.models.transformer import finetune_predict
            mc, tc = cfg["models"][args.model], cfg["transformer"]
            val_pred, test_pred, meta = finetune_predict(
                sub["text"].tolist(), sub["label"].tolist(), vsel["text"].tolist(), vsel["label"].tolist(),
                test["text"].tolist(), mc["name"], mc["revision"], seed=sub_seed, lr=args.lr or 2e-5,
                batch_size=tc["batch_size"], max_length=tc["max_length"], min_steps=tc["min_steps"],
                max_epochs=tc["max_epochs"], evals_per_run=tc["evals_per_run"], patience=tc["patience"],
                weight_decay=tc["weight_decay"], warmup_ratio=tc["warmup_ratio"])
        rec = {"model": args.model, "split_mode": mode, "n_val_selection": len(vsel), "size": str(size), "n_train": n,
               "repeat": rep, "subsample_seed": sub_seed, "lr": args.lr,
               "val_macro_f1": macro_f1(vsel["label"], val_pred), "test": report(test["label"].tolist(), list(test_pred)),
               "seconds": round(time.time() - t0, 1), "meta": meta}
        tag = f"{args.model}_{mode}_n{size}_r{rep}" + (f"_lr{args.lr:g}" if args.lr else "")
        pd.DataFrame({"row_id": test["row_id"].values, "pred": list(test_pred)}).to_csv(PRED_DIR / f"{tag}.csv", index=False)
        with RUNS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        print(f"{tag}: val {rec['val_macro_f1']:.4f} test {rec['test']['macro_f1']:.4f} ({rec['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
