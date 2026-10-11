"""Aggregate saved runs into tables, figures and statistical comparisons.

Reads outputs/metrics/runs.jsonl and outputs/predictions/*.csv. Writes:
    outputs/metrics/lr_selection.json
    outputs/metrics/learning_curve.json, outputs/tables/learning_curve.{csv,md}
    outputs/metrics/paired_by_size.json, outputs/tables/paired_by_size.{csv,md}
    outputs/metrics/full_comparison.json, outputs/tables/full_comparison.md
    outputs/metrics/leakage.json
    outputs/figures/learning_curve.png, outputs/figures/difference_by_size.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nepsent.data.clean import LABELS  # noqa: E402
from nepsent.eval.metrics import bootstrap_ci, fast_macro_f1, holm, macro_f1, mcnemar_exact, paired_bootstrap  # noqa: E402

M, T, F = Path("outputs/metrics"), Path("outputs/tables"), Path("outputs/figures")
PRED = Path("outputs/predictions")
NAMES = {"majority": "Majority class", "char_lr": "Char n-gram LR", "xlmr": "XLM-R base", "nepberta": "NepBERTa"}
COLORS = {"char_lr": "#2a78d6", "xlmr": "#eb6834", "nepberta": "#1baf7a"}
MARKERS = {"char_lr": "o", "xlmr": "s", "nepberta": "^"}


_GOLD = None


def read_pred(path):
    """Prediction file joined with gold labels from the locally rebuilt corpus (labels are not tracked)."""
    global _GOLD
    if _GOLD is None:
        _GOLD = pd.read_parquet("data/interim/corpus.parquet")[["row_id", "label"]]
    p = pd.read_csv(path)
    out = p.merge(_GOLD, on="row_id", how="left")
    assert out["label"].notna().all() and len(out) == len(p)
    return out


def size_key(s):
    return 10**9 if s == "full" else int(s)


def tag(r):
    return f"{r['model']}_{r['split_mode']}_n{r['size']}_r{r['repeat']}" + (f"_lr{r['lr']:g}" if r["lr"] and r["lr"] == r["lr"] else "")


def codes(a):
    """Encode label strings as integer codes for fast bootstrap loops."""
    return np.array([LABELS.index(v) for v in a])


def mean_macro_f1_boot(y, preds, n_boot, seed=0):
    """Bootstrap the mean over repeats of per-repeat macro F1, resampling test items jointly."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = np.empty(n_boot)
    for b, idx in enumerate(rng.integers(0, n, size=(n_boot, n))):
        out[b] = np.mean([fast_macro_f1(y[idx], p[idx]) for p in preds])
    return out


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    cfg = yaml.safe_load(open("config/experiment.yaml", encoding="utf-8"))
    n_boot = cfg["bootstrap"]["n_boot"]
    for d in (M, T, F):
        d.mkdir(parents=True, exist_ok=True)
    runs = [json.loads(x) for x in (M / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    runs = pd.DataFrame(runs)
    runs["test_macro_f1"] = runs["test"].map(lambda t: t["macro_f1"])

    # Learning-rate selection: validation macro F1 at full size, repeat 0.
    lr_sel = {}
    for model in ("xlmr", "nepberta"):
        cand = runs[(runs.model == model) & (runs.split_mode == "grouped") & (runs["size"] == "full")
                    & (runs.repeat == 0) & runs.lr.notna()]
        cand = cand[cand.lr.isin(cfg["transformer"]["lr_candidates"])]
        if len(cand):
            best = cand.loc[cand.val_macro_f1.idxmax()]
            lr_sel[model] = {"selected_lr": float(best.lr),
                             "val_macro_f1_by_lr": {f"{lr:g}": float(v) for lr, v in zip(cand.lr, cand.val_macro_f1)}}
    (M / "lr_selection.json").write_text(json.dumps(lr_sel, indent=2))

    def keep(r):
        if r.model in lr_sel:
            return r.lr == lr_sel[r.model]["selected_lr"]
        return True
    g = runs[(runs.split_mode == "grouped") & runs.apply(keep, axis=1)].copy()
    models = [m for m in ("majority", "char_lr", "xlmr", "nepberta") if m in set(g.model)]

    # Learning-curve table.
    rows = []
    test_y = None
    for (model, size), sub in g.groupby(["model", "size"]):
        preds = []
        for _, r in sub.sort_values("repeat").iterrows():
            p = read_pred(PRED / f"{tag(r)}.csv")
            test_y = p["label"].to_numpy() if test_y is None else test_y
            preds.append(p["pred"].to_numpy())
        boot = mean_macro_f1_boot(codes(test_y), [codes(x) for x in preds], n_boot // 4)
        rows.append({"model": model, "size": size, "n_train": int(sub.n_train.iloc[0]), "repeats": len(sub),
                     "mean_macro_f1": sub.test_macro_f1.mean(), "median_macro_f1": sub.test_macro_f1.median(), "sd_across_repeats": sub.test_macro_f1.std(ddof=1)
                     if len(sub) > 1 else 0.0, "min": sub.test_macro_f1.min(), "max": sub.test_macro_f1.max(),
                     "test_boot_ci_low": float(np.quantile(boot, .025)),
                     "test_boot_ci_high": float(np.quantile(boot, .975))})
    lc = pd.DataFrame(rows).sort_values(["model", "n_train"])
    (M / "learning_curve.json").write_text(lc.to_json(orient="records", indent=2))
    lc.to_csv(T / "learning_curve.csv", index=False)
    show = lc[lc.model.isin([m for m in models if m != "majority"])].copy()
    show["model"] = show.model.map(NAMES)
    for c in ("mean_macro_f1", "median_macro_f1", "sd_across_repeats", "min", "max", "test_boot_ci_low", "test_boot_ci_high"):
        show[c] = show[c].map(lambda v: f"{v:.3f}")
    show = show.drop(columns="size").rename(columns={
        "model": "Model", "n_train": "Train size", "repeats": "Repeats", "mean_macro_f1": "Mean", "median_macro_f1": "Median",
        "sd_across_repeats": "SD", "min": "Min", "max": "Max", "test_boot_ci_low": "CI low", "test_boot_ci_high": "CI high"})
    (T / "learning_curve.md").write_text(md_table(show))

    # Paired differences per size: (transformer - char_lr) on identical subsamples.
    paired = {}
    gp = runs[(runs.split_mode == "grouped_propval") & runs.apply(keep, axis=1)].copy()
    for suffix, gg in (("", g), ("_propval", gp)):
        for model, base in [(m, b) for m, b in (("xlmr", "char_lr"), ("nepberta", "char_lr"), ("nepberta", "xlmr"))
                            if m in set(gg.model) and b in set(gg.model)]:
            bkey = "" if base == "char_lr" else f"_vs_{base}"
            rows, pvals = [], {}
            for size in sorted(set(gg[gg.model == model]["size"]), key=size_key):
                a = gg[(gg.model == model) & (gg["size"] == size)].set_index("repeat", drop=False)
                b = gg[(gg.model == base) & (gg["size"] == size)].set_index("repeat", drop=False)
                reps = sorted(set(a.index) & set(b.index))
                if not reps:
                    continue
                pa = [codes(read_pred(PRED / f"{tag(a.loc[r])}.csv")["pred"]) for r in reps]
                pb = [codes(read_pred(PRED / f"{tag(b.loc[r])}.csv")["pred"]) for r in reps]
                yc = codes(test_y)
                rng = np.random.default_rng(1)
                n = len(test_y)
                diffs = np.empty(n_boot)
                for k, idx in enumerate(rng.integers(0, n, size=(n_boot, n))):
                    diffs[k] = np.mean([fast_macro_f1(yc[idx], x[idx]) - fast_macro_f1(yc[idx], y[idx])
                                        for x, y in zip(pa, pb)])
                per_rep = [fast_macro_f1(yc, x) - fast_macro_f1(yc, y) for x, y in zip(pa, pb)]
                obs = float(np.mean(per_rep))
                centred = diffs - diffs.mean()
                p = float((np.sum(np.abs(centred) >= abs(obs)) + 1) / (n_boot + 1))
                pvals[size] = p
                rows.append({"size": size, "n_train": int(a.n_train.iloc[0]), "repeats": len(reps),
                             "mean_diff": obs, "ci_low": float(np.quantile(diffs, .025)),
                             "ci_high": float(np.quantile(diffs, .975)), "p_value": p,
                             "repeats_first_better": int(sum(d > 0 for d in per_rep)),
                             "per_repeat_diff": per_rep})
            adj = holm(pvals)
            for r in rows:
                r["p_holm"] = adj[r["size"]]
            # Crossover: smallest size whose interval is above zero and stays above zero at all larger sizes.
            cross = None
            for i, r in enumerate(rows):
                if all(x["ci_low"] > 0 for x in rows[i:]):
                    cross = r["n_train"]
                    break
            paired[model + bkey + suffix] = {"comparison": f"{model} minus {base}{suffix}", "rows": rows, "crossover_n_train": cross,
                             "definition": "smallest n with 95% CI of mean paired difference > 0 at n and all larger n"}
            tbl = pd.DataFrame([{k: v for k, v in r.items() if k != "per_repeat_diff"} for r in rows])
            tbl.insert(0, "comparison", f"{NAMES[model]} - {NAMES[base]}")
            for c in ("mean_diff", "ci_low", "ci_high"):
                tbl[c] = tbl[c].map(lambda v: f"{v:+.3f}")
            for c in ("p_value", "p_holm"):
                tbl[c] = tbl[c].map(lambda v: f"{v:.4f}")
            tbl.drop(columns="size").to_csv(T / f"paired_{model}{bkey}{suffix}.csv", index=False)
            (T / f"paired_{model}{bkey}{suffix}.md").write_text(md_table(tbl.drop(columns="size").rename(columns={
            "comparison": "Comparison", "n_train": "Train size", "repeats": "Repeats", "mean_diff": "Mean diff",
            "ci_low": "CI low", "ci_high": "CI high", "p_value": "p", "repeats_first_better": "Repeats first better",
            "p_holm": "p (Holm)"})))
    (M / "paired_by_size.json").write_text(json.dumps(paired, indent=2))

    # Full-training-set comparison (repeat 0 of each model, identical training data).
    full = {}
    preds0 = {}
    for model in models:
        r = g[(g.model == model) & (g["size"] == "full") & (g.repeat == 0)]
        if not len(r):
            continue
        r = r.iloc[0]
        preds0[model] = read_pred(PRED / f"{tag(r)}.csv")["pred"].to_numpy()
        lo, hi = bootstrap_ci(test_y, preds0[model], metric=fast_macro_f1, n_boot=n_boot)
        full[model] = {**r["test"], "macro_f1_ci": [lo, hi], "val_macro_f1": r.val_macro_f1, "meta": r.meta}
    tests = {}
    pairs = [(a, b) for i, a in enumerate(preds0) for b in list(preds0)[i + 1:] if "majority" not in (a, b)]
    for a, b in pairs:
        tests[f"{a}_vs_{b}"] = {"paired_bootstrap_macro_f1": paired_bootstrap(test_y, preds0[a], preds0[b], metric=fast_macro_f1, n_boot=n_boot),
                                "mcnemar_exact_accuracy": mcnemar_exact(test_y, preds0[a], preds0[b])}
    adj = holm({k: v["paired_bootstrap_macro_f1"]["p_value"] for k, v in tests.items()})
    for k in tests:
        tests[k]["p_holm"] = adj[k]
    (M / "full_comparison.json").write_text(json.dumps({"models": full, "tests": tests}, indent=2, default=str))
    lines = ["| Model | Macro F1 | 95% CI | Accuracy | F1 neg | F1 neu | F1 pos |", "|---|---|---|---|---|---|---|"]
    for m, v in full.items():
        pc = v["per_class"]
        lines.append(f"| {NAMES[m]} | {v['macro_f1']:.3f} | {v['macro_f1_ci'][0]:.3f} to {v['macro_f1_ci'][1]:.3f} | "
                     f"{v['accuracy']:.3f} | {pc['negative']['f1']:.3f} | {pc['neutral']['f1']:.3f} | {pc['positive']['f1']:.3f} |")
    (T / "full_comparison.md").write_text("\n".join(lines) + "\n")
    cm_lines = []
    for m, v in full.items():
        cm_lines.append(f"{NAMES[m]} (rows true, columns predicted: {', '.join(LABELS)})\n")
        cm_lines += ["    " + " ".join(f"{x:5d}" for x in row) for row in v["confusion_matrix"]["matrix"]]
        cm_lines.append("")
    (T / "confusion_matrices.txt").write_text("\n".join(cm_lines))

    # Leakage analysis: grouped vs random split for the character model at full size.
    rnd = runs[(runs.split_mode == "random") & (runs.model == "char_lr") & (runs["size"] == "full")]
    if len(rnd) and "char_lr" in full:
        rr = rnd.iloc[0]
        p = read_pred(PRED / f"{tag(rr)}.csv")
        lo, hi = bootstrap_ci(p["label"], p["pred"], metric=fast_macro_f1, n_boot=n_boot)
        (M / "leakage.json").write_text(json.dumps({
            "char_lr_grouped": {"macro_f1": full["char_lr"]["macro_f1"], "ci": full["char_lr"]["macro_f1_ci"]},
            "char_lr_random": {"macro_f1": rr.test_macro_f1, "ci": [lo, hi]},
            "note": "different test sets; intervals are not a paired comparison"}, indent=2))

    # Figures.
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for model in [m for m in ("char_lr", "xlmr", "nepberta") if m in models]:
        d = lc[lc.model == model].sort_values("n_train")
        pts = g[g.model == model]
        ax.scatter(pts.n_train * (1.06 if model == "xlmr" else 0.94), pts.test_macro_f1, s=10,
                   color=COLORS[model], alpha=0.45, linewidths=0)
        ax.plot(d.n_train, d.median_macro_f1, color=COLORS[model], marker=MARKERS[model], markersize=5, linewidth=2)
        ax.annotate(NAMES[model], (d.n_train.iloc[-1], d.median_macro_f1.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color="#0b0b0b", fontsize=9)
    maj = lc[lc.model == "majority"]
    if len(maj):
        ax.axhline(maj.mean_macro_f1.iloc[0], color="#52514e", linestyle=":", linewidth=1)
        ax.annotate("Majority class", (lc.n_train.min(), maj.mean_macro_f1.iloc[0]), xytext=(0, 4),
                    textcoords="offset points", fontsize=8, color="#52514e")
    ax.set_xscale("log")
    ax.set_xlabel("Training examples (log scale)")
    ax.set_ylabel("Test macro F1")
    ax.set_ylim(0.18, None)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_xlim(right=lc.n_train.max() * 2.6)
    fig.tight_layout()
    fig.savefig(F / "learning_curve.png", dpi=200)
    plt.close(fig)

    if paired:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for key, res in paired.items():
            if "_vs_" in key:
                continue
            model, prop = key.replace("_propval", ""), key.endswith("_propval")
            d = pd.DataFrame(res["rows"])
            ax.errorbar(d.n_train * (1.08 if prop else 1.0), d.mean_diff,
                        yerr=[d.mean_diff - d.ci_low, d.ci_high - d.mean_diff],
                        color=COLORS[model], marker=MARKERS[model], markersize=5, linewidth=2, capsize=3,
                        linestyle="--" if prop else "-", markerfacecolor="white" if prop else COLORS[model],
                        label=f"{NAMES[model]} minus Char n-gram LR" + (" (small validation set)" if prop else ""))
        ax.axhline(0, color="#52514e", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("Training examples (log scale)")
        ax.set_ylabel("Macro F1 difference")
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        fig.tight_layout()
        fig.savefig(F / "difference_by_size.png", dpi=200)
        plt.close(fig)
    print("analysis written")


if __name__ == "__main__":
    main()
