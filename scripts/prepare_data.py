"""Build the cleaned corpus and the frozen train/validation/test split.

Input: the NepCOV19Tweets CSV released by the dataset authors
(see docs/data_source_audit.md), placed at data/raw/nepcov19tweets/.

Outputs:
    data/interim/corpus.parquet          cleaned text with labels (not tracked)
    data/processed/splits.csv            row_id, text_sha1, group, split (tracked; no text, no labels)
    data/processed/test_ids.sha256       hash of test row ids
    outputs/metrics/data_audit.json      audit counts
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from nepsent.data.clean import normalize
from nepsent.data.split import grouped_split, ids_hash

LABEL_MAP = {"-1": "negative", "0": "neutral", "1": "positive"}


def read_raw(path: Path):
    """Read the authors' CSV, keeping only well-formed rows.

    A small number of rows in the release contain broken quoting that merges
    several records into one field; these are counted and dropped.
    """
    rows, bad = [], 0
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for rec in reader:
            if len(rec) != len(header) or rec[1].strip() not in LABEL_MAP or not rec[0].strip().isdigit():
                bad += 1
                continue
            rows.append({"row_id": int(rec[0]), "label": LABEL_MAP[rec[1].strip()],
                         "datetime": rec[2], "text": rec[3]})
    return pd.DataFrame(rows), bad


def near_dup_groups(texts: list[str], threshold: float, chunk: int = 2000) -> np.ndarray:
    """Union-find clusters of texts whose character 3-5 gram TF-IDF cosine >= threshold."""
    x = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True).fit_transform(texts)
    parent = np.arange(len(texts))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, len(texts), chunk):
        sims = (x[start:start + chunk] @ x.T).tocoo()
        for i, j, v in zip(sims.row + start, sims.col, sims.data):
            if i < j and v >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(len(texts))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    raw_path = Path(cfg["data"]["raw_csv"])
    audit = {"raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()}

    df, bad = read_raw(raw_path)
    audit["rows_wellformed"] = len(df)
    audit["rows_malformed_dropped"] = bad
    df["text"] = df["text"].map(normalize)
    audit["empty_text_dropped"] = int((df["text"] == "").sum())
    df = df[df["text"] != ""]

    # Exact duplicates: keep one copy if labels agree, drop all copies if they conflict.
    n_labels = df.groupby("text")["label"].transform("nunique")
    audit["exact_dup_rows_conflicting_dropped"] = int((n_labels > 1).sum())
    df = df[n_labels == 1]
    dup = df.duplicated("text")
    audit["exact_dup_rows_collapsed"] = int(dup.sum())
    df = df[~dup].sort_values("row_id").reset_index(drop=True)

    thr = cfg["data"]["near_dup_threshold"]
    df["group"] = near_dup_groups(df["text"].tolist(), thr)
    sizes = df.groupby("group").size()
    audit["near_dup_threshold"] = thr
    audit["near_dup_groups_multi"] = int((sizes > 1).sum())
    audit["rows_in_multi_groups"] = int(sizes[sizes > 1].sum())
    audit["near_dup_groups_label_conflict"] = int((df.groupby("group")["label"].nunique() > 1).sum())

    s = cfg["split"]
    df["split"] = grouped_split(df, test_frac=s["test_frac"], val_frac=s["val_frac"], seed=cfg["seed"]).values
    df["text_sha1"] = df["text"].map(lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest())
    df["n_words"] = df["text"].str.split().str.len()

    audit["final_rows"] = len(df)
    audit["label_counts"] = df["label"].value_counts().to_dict()
    audit["split_counts"] = {k: g["label"].value_counts().to_dict() for k, g in df.groupby("split")}
    audit["words_per_text"] = df["n_words"].describe().round(2).to_dict()
    audit["date_range"] = [df["datetime"].min(), df["datetime"].max()]
    audit["contains_latin"] = int(df["text"].str.contains("[A-Za-z]").sum())
    audit["test_ids_sha256"] = ids_hash(df.loc[df["split"] == "test", "row_id"])

    Path("data/interim").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    df.to_parquet("data/interim/corpus.parquet", index=False)
    df[["row_id", "text_sha1", "group", "split"]].to_csv("data/processed/splits.csv", index=False)
    Path("data/processed/test_ids.sha256").write_text(audit["test_ids_sha256"] + "\n")
    Path("outputs/metrics/data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
