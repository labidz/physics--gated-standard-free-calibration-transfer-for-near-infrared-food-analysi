"""E3 robustness (added after the main analysis; conservative sensitivity check, no new model runs).

The pooled replicate-level Wilcoxon in make_tables.py is dominated by the corn family (240/351 replicates),
which was also the TTPA development pilot (protocol deviation 6). Here we report
  (a) replicate-level Holm-Wilcoxon excluding corn,
  (b) family-level comparison (one mean log-ratio per family; sign test + bootstrap CI over families),
  (c) the same split by shift type (instrument/measurement vs population/process).
Regression families use log(RMSEP ratio); the grapes classification family uses log(error-rate ratio).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest
ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "results/tables/e3_all_runs.csv")
df["err"] = np.where(df["kind"] == "cls", 1 - df["value"], df["value"])
SHIFT = {"corn": "instrument", "ch2021": "instrument", "tablets": "instrument", "simfood": "instrument",
         "wheat drift (2 months)": "instrument", "mango fruit temperature low-mid -> high": "instrument",
         "cassava new year": "population", "grapes new harvest dates": "population", "mango season 1-3 -> 4": "population",
         "plum leave-one-cultivar-out SSC": "population", "ch2018 leave-one-product-out protein": "population", "ligno": "process"}
W = df.pivot_table(index=["family", "scenario", "rep", "kind"], columns="method", values="err").reset_index()
OURS, BASES = "SpecTwin-TTPA", ["NC", "DMC", "CORAL", "diPLS"]
rng = np.random.default_rng(0)
lines = ["# E3 robustness: SpecTwin-TTPA vs unsupervised baselines", "",
         "Negative log-ratio = TTPA better. Replicate-level and family-level columns use the 11 regression families",
         "(grapes, classification, is listed per family only). Family-level: one mean log-ratio per family, equal weight.", ""]
lines += ["| Comparison | replicate-level, all (n, W/L, median, p) | replicate-level, excl. corn (n, W/L, median, p) | family-level W/L | family mean log-ratio [95% bootstrap CI] | sign-test p |",
          "|---|---|---|---|---|---|"]
fam_tab = {}
for b in BASES:
    sub = W.dropna(subset=[OURS, b])
    lr = np.log(sub[OURS].clip(lower=1e-12) / sub[b].clip(lower=1e-12))
    def rep_stats(mask):
        x = lr[mask & (sub["kind"] == "reg")]
        x = x[np.abs(x) > 1e-12]
        return len(x), int((x < 0).sum()), int((x > 0).sum()), float(np.median(x)), float(wilcoxon(x).pvalue)
    a = rep_stats(np.ones(len(sub), bool)); c = rep_stats(sub["family"] != "corn")
    fam = lr.groupby(sub["family"]).mean(); fam_tab[b] = fam
    fv = fam[[f for f in fam.index if f != "grapes new harvest dates"]].values; boots = [rng.choice(fv, len(fv)).mean() for _ in range(10000)]
    nz = fv[np.abs(fv) > 1e-12]; st = binomtest(int((nz < 0).sum()), len(nz)).pvalue
    lines.append(f"| TTPA vs {b} | {a[0]}, {a[1]}/{a[2]}, {a[3]:+.3f}, {a[4]:.2g} | {c[0]}, {c[1]}/{c[2]}, {c[3]:+.3f}, {c[4]:.2g} | "
                 f"{int((fv < 0).sum())}/{int((fv > 0).sum())} | {fv.mean():+.3f} [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}] | {st:.2g} |")
lines += ["", "Per-family mean log-ratio (TTPA vs baseline) and shift type:", "",
          "| Family | shift type | n reps | vs NC | vs DMC | vs CORAL | vs di-PLS |", "|---|---|---|---|---|---|---|"]
for f in sorted(fam_tab["NC"].index, key=lambda f: (SHIFT[f], f)):
    n = int((W["family"] == f).sum())
    cells = [f"{fam_tab[b].get(f, np.nan):+.3f}" if f in fam_tab[b].index else "–" for b in BASES]
    lines.append(f"| {f} | {SHIFT[f]} | {n} | " + " | ".join(cells) + " |")
lines += ["", "By shift type (family-level W/L for TTPA):", ""]
for t in ["instrument", "population", "process"]:
    fs = [f for f in fam_tab["NC"].index if SHIFT[f] == t]
    cells = []
    for b in BASES:
        v = fam_tab[b][[f for f in fs if f in fam_tab[b].index]]
        cells.append(f"vs {b}: {int((v < 0).sum())}/{int((v > 0).sum())}")
    lines.append(f"- {t} ({len(fs)} families): " + "; ".join(cells))
out = ROOT / "results/tables/e3_robustness.md"; out.write_text("\n".join(lines) + "\n"); print("\n".join(lines))
