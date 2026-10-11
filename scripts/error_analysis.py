"""Quantitative error analysis.

Writes outputs/metrics/error_slices.json: macro F1 by text length, error overlap
between two models, and confusion patterns of the first model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nepsent.eval.metrics import macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-pred", required=True, help="prediction CSV of the model to analyse")
    ap.add_argument("--baseline-pred", required=True, help="prediction CSV of the comparison model")
    args = ap.parse_args()

    corpus = pd.read_parquet("data/interim/corpus.parquet")[["row_id", "label", "n_words", "group"]]
    a = pd.read_csv(args.model_pred).rename(columns={"pred": "pred_model"})
    b = pd.read_csv(args.baseline_pred)[["row_id", "pred"]].rename(columns={"pred": "pred_base"})
    df = a.merge(b, on="row_id").merge(corpus, on="row_id")
    gsize = corpus.groupby("group").size()
    df["group_size"] = df["group"].map(gsize)

    bins = [0, 8, 16, 27, 1000]
    names = ["1-8", "9-16", "17-27", "28+"]
    df["len_bin"] = pd.cut(df["n_words"], bins=bins, labels=names)
    slices = {"by_length_words": {}, "overlap": {}}
    for name, g in df.groupby("len_bin", observed=True):
        slices["by_length_words"][name] = {"n": int(len(g)), "model": macro_f1(g.label, g.pred_model),
                                           "baseline": macro_f1(g.label, g.pred_base)}
    em, eb = df.pred_model != df.label, df.pred_base != df.label
    slices["overlap"] = {"both_wrong": int((em & eb).sum()), "model_only_wrong": int((em & ~eb).sum()),
                         "baseline_only_wrong": int((~em & eb).sum()), "both_right": int((~em & ~eb).sum()),
                         "both_wrong_same_label": int((em & eb & (df.pred_model == df.pred_base)).sum())}
    errs = df[em]
    slices["model_error_confusions"] = {f"{t}->{p}": int(c) for (t, p), c in
                                        errs.groupby(["label", "pred_model"]).size().items()}
    Path("outputs/metrics/error_slices.json").write_text(json.dumps(slices, indent=2))

    print(json.dumps(slices, indent=2))


if __name__ == "__main__":
    main()
