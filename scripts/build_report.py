"""Fill report/report_template.md with values from saved results and render outputs.

Every number in the report is read from files written by prepare_data.py,
run_experiments.py, analyze.py and error_analysis.py.

Writes report/report.md, report/report.html and, if Microsoft Edge or Chrome is
available, report/report.pdf. Also refreshes the results block in README.md.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import markdown

M, T = Path("outputs/metrics"), Path("outputs/tables")


def j(name):
    return json.loads((M / name).read_text(encoding="utf-8"))


def f3(x):
    return f"{x:.3f}"


def values():
    audit = j("data_audit.json")
    lc = j("learning_curve.json")
    paired = j("paired_by_size.json")
    full = j("full_comparison.json")
    leak = j("leakage.json")
    errs = j("error_slices.json")
    lrsel = j("lr_selection.json")
    grid = j("char_lr_full_grid_grouped.json")
    ext = j("char_lr_ngram_extension.json")
    v = {}
    v.update({k: f"{audit[k]:,}" for k in ("rows_wellformed", "rows_malformed_dropped", "exact_dup_rows_conflicting_dropped",
                                           "exact_dup_rows_collapsed", "final_rows", "near_dup_groups_multi",
                                           "rows_in_multi_groups", "near_dup_groups_label_conflict")})
    for lab, n in audit["label_counts"].items():
        v[f"n_{lab}"] = f"{n:,}"
        v[f"pct_{lab}"] = f"{100 * n / audit['final_rows']:.1f}"
    for s, counts in audit["split_counts"].items():
        v[f"n_{s}"] = f"{sum(counts.values()):,}"
    v["words_median"] = f"{audit['words_per_text']['50%']:.0f}"
    v["date_from"], v["date_to"] = audit["date_range"][0][:10], audit["date_range"][1][:10]
    v["test_hash"] = audit["test_ids_sha256"][:16]

    g = grid["best_params"]
    v["char_ngram"] = f"{g['ngram_range'][0]}-{g['ngram_range'][1]}"
    v["char_analyzer"], v["char_C"], v["char_cw"] = g["analyzer"], g["C"], g["class_weight"]
    v["char_val"] = f3(grid["val_macro_f1"])
    v["char_ext_max"] = f3(max(ext.values()))
    for m, sel in lrsel.items():
        v[f"{m}_lr"] = f"{sel['selected_lr']:g}"
        v[f"{m}_lr_vals"] = ", ".join(f"{k}: {float(x):.3f}" for k, x in sel["val_macro_f1_by_lr"].items())

    def cell(model, size):
        r = [x for x in lc if x["model"] == model and x["size"] == size]
        return r[0] if r else None
    for m in ("char_lr", "xlmr", "nepberta"):
        for size in ("100", "500", "1000", "2500", "10000", "full"):
            c = cell(m, size)
            if c:
                v[f"{m}_{size}"] = f3(c["mean_macro_f1"])
                v[f"{m}_{size}_med"] = f3(c["median_macro_f1"])
    v["n_full"] = f"{cell('char_lr', 'full')['n_train']:,}"
    collapsed = [x for x in lc if x["model"] in ("xlmr", "nepberta") and x["min"] < 0.3]
    names = {"xlmr": "XLM-R", "nepberta": "NepBERTa"}
    v["collapsed_cells"] = "; ".join(f"{names[x['model']]} at n = {x['n_train']} (minimum {x['min']:.3f})" for x in collapsed) or "none"

    for key, res in paired.items():
        rows = res["rows"]
        v[f"{key}_cross"] = "not reached" if res["crossover_n_train"] is None else f"{res['crossover_n_train']:,}"
        v[f"{key}_diff_min"] = f"{min(r['mean_diff'] for r in rows):+.3f}"
        v[f"{key}_diff_max"] = f"{max(r['mean_diff'] for r in rows):+.3f}"
        best = max(rows, key=lambda r: r["mean_diff"])
        v[f"{key}_diff_max_n"] = f"{best['n_train']:,}"
        last = rows[-1]
        v[f"{key}_diff_full"] = f"{last['mean_diff']:+.3f}"
        v[f"{key}_diff_full_ci"] = f"{last['ci_low']:+.3f} to {last['ci_high']:+.3f}"
        v[f"{key}_all_positive"] = "yes" if all(r["ci_low"] > 0 for r in rows) else "no"
        v[f"{key}_max_pholm"] = f"{max(r['p_holm'] for r in rows):.4f}"
        v[f"{key}_sizes"] = ", ".join(f"{r['n_train']:,}" for r in rows)

    fm = full["models"]
    for m, d in fm.items():
        v[f"full_{m}"] = f3(d["macro_f1"])
        v[f"full_{m}_ci"] = f"{d['macro_f1_ci'][0]:.3f} to {d['macro_f1_ci'][1]:.3f}"
        v[f"full_{m}_acc"] = f3(d["accuracy"])
        for lab in ("negative", "neutral", "positive"):
            v[f"full_{m}_{lab}"] = f3(d["per_class"][lab]["f1"])
            v[f"full_{m}_{lab}_rec"] = f3(d["per_class"][lab]["recall"])
    for name, t in full["tests"].items():
        pb, mc = t["paired_bootstrap_macro_f1"], t["mcnemar_exact_accuracy"]
        v[f"test_{name}_diff"] = f"{pb['diff']:+.3f}"
        v[f"test_{name}_ci"] = f"{pb['ci_low']:+.3f} to {pb['ci_high']:+.3f}"
        v[f"test_{name}_p"] = f"{t['p_holm']:.4f}"
        v[f"test_{name}_mc"] = f"{mc['a_only_correct']} vs {mc['b_only_correct']}, exact p = {mc['p_value']:.2g}"

    v["leak_grouped"] = f3(leak["char_lr_grouped"]["macro_f1"])
    v["leak_random"] = f3(leak["char_lr_random"]["macro_f1"])
    v["leak_grouped_ci"] = "{:.3f} to {:.3f}".format(*leak["char_lr_grouped"]["ci"])
    v["leak_random_ci"] = "{:.3f} to {:.3f}".format(*leak["char_lr_random"]["ci"])
    v["leak_gap"] = f"{leak['char_lr_random']['macro_f1'] - leak['char_lr_grouped']['macro_f1']:+.3f}"

    o = errs["overlap"]
    v.update({f"err_{k}": f"{x:,}" for k, x in o.items()})
    v["err_same_pct"] = f"{100 * o['both_wrong_same_label'] / o['both_wrong']:.0f}"
    c = errs["model_error_confusions"]
    tot = sum(c.values())
    v["err_total"] = f"{tot:,}"
    v["err_polarity"] = f"{c.get('negative->positive', 0) + c.get('positive->negative', 0):,}"
    v["err_polarity_pct"] = f"{100 * (c.get('negative->positive', 0) + c.get('positive->negative', 0)) / tot:.0f}"
    v["err_neutral_out"] = f"{c.get('neutral->positive', 0) + c.get('neutral->negative', 0):,}"
    bl = errs["by_length_words"]
    v["len_short_model"], v["len_short_base"] = f3(bl["1-8"]["model"]), f3(bl["1-8"]["baseline"])
    v["len_long_model"], v["len_long_base"] = f3(bl["28+"]["model"]), f3(bl["28+"]["baseline"])

    for tbl in ("learning_curve", "paired_xlmr", "paired_xlmr_propval", "paired_nepberta", "paired_nepberta_vs_xlmr", "full_comparison"):
        p = T / f"{tbl}.md"
        text = p.read_text(encoding="utf-8").strip() if p.exists() else "(not available)"
        if tbl.startswith("paired") and p.exists():
            # The section heading names the comparison, so the repeated first column is dropped here.
            text = "\n".join("|" + "|".join(line.split("|")[2:]) for line in text.splitlines())
        v[f"table_{tbl}"] = text
    return v


def render(md_text: str) -> str:
    body = markdown.markdown(md_text, extensions=["tables", "md_in_html"])
    css = """
    body { font-family: Georgia, 'Times New Roman', serif; max-width: 46em; margin: 2em auto; padding: 0 16px;
           line-height: 1.45; color: #111; background: #fff; font-size: 11pt; }
    h1 { font-size: 1.55em; line-height: 1.25; } h2 { font-size: 1.2em; margin-top: 1.6em; } h3 { font-size: 1.05em; }
    table { border-collapse: collapse; margin: 0.8em 0; font-size: 0.86em; }
    th, td { border-bottom: 1px solid #ccc; padding: 3px 7px; text-align: left; }
    th { border-bottom: 1.5px solid #333; }
    img { max-width: 100%; display: block; margin: 0 auto; } code { font-size: 0.9em; }
    figure { margin: 1.2em 0; break-inside: avoid; page-break-inside: avoid; }
    figcaption { font-size: 0.9em; color: #333; margin-top: 0.4em; }
    h1, h2, h3 { break-after: avoid; page-break-after: avoid; }
    tr { break-inside: avoid; page-break-inside: avoid; }
    .keep { break-inside: avoid; page-break-inside: avoid; }
    p, li { orphans: 3; widows: 3; }
    @page { margin: 18mm 16mm; }
    """
    return f"<!doctype html><html><head><meta charset='utf-8'><title>How Much Data Do Nepali Sentiment Models Need?</title><style>{css}</style></head><body>{body}</body></html>"


def find_browser():
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
        if Path(p).exists():
            return p
    return shutil.which("chromium") or shutil.which("google-chrome") or shutil.which("msedge")


def main():
    v = values()
    # Interpretive sentences that depend on results are kept in report/narrative.json;
    # they may themselves reference value keys, which are filled here.
    narrative = json.loads(Path("report/narrative.json").read_text(encoding="utf-8"))
    v.update({k: t.format_map(v) for k, t in narrative.items()})
    tmpl = Path("report/report_template.md").read_text(encoding="utf-8")
    missing = sorted(set(re.findall(r"\{([a-zA-Z0-9_]+)\}", tmpl)) - set(v))
    if missing:
        raise SystemExit(f"template keys without values: {missing}")
    md_text = tmpl.format_map(v)
    Path("report/report.md").write_text(md_text, encoding="utf-8")
    html = render(md_text.replace("](../outputs/", "](../outputs/"))
    html_path = Path("report/report.html").resolve()
    html_path.write_text(html, encoding="utf-8")
    browser = find_browser()
    if browser:
        pdf = Path("report/report.pdf").resolve()
        pdf.unlink(missing_ok=True)
        import tempfile
        profile = tempfile.mkdtemp(prefix="nepsent_pdf_")
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={profile}", f"--print-to-pdf={pdf}", html_path.as_uri()],
                       check=False, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import time
        for _ in range(60):  # the Edge launcher can return before the PDF is written
            if pdf.exists() and pdf.stat().st_size > 0:
                time.sleep(2)
                break
            time.sleep(1)
        shutil.rmtree(profile, ignore_errors=True)
        print("pdf:", pdf.exists())

    readme = Path("README.md").read_text(encoding="utf-8")
    block = Path("report/readme_results_template.md").read_text(encoding="utf-8").format_map(v)
    readme = re.sub(r"<!-- results:start -->.*<!-- results:end -->",
                    "<!-- results:start -->\n" + block.strip() + "\n<!-- results:end -->", readme, flags=re.S)
    Path("README.md").write_text(readme, encoding="utf-8")
    print("report written")


if __name__ == "__main__":
    main()
