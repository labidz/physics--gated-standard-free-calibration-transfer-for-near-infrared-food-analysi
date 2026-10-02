"""Collect all results, run the pre-registered statistics and write the results tables (Markdown + CSV + JSON)."""
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spectwin.tasks import E1_TASKS

R = ROOT / "results"
T = R / "tables"; T.mkdir(parents=True, exist_ok=True)
NAMES = {"pls": "PLS", "ridge": "Ridge", "svm": "SVR/SVC", "gpr": "GPR", "lwpls": "kNN-LWPLS", "lgbm": "LightGBM", "knn": "kNN",
         "cnn": "CNN1D", "transformer": "Transformer", "twin_pls": "SpecTwin-PLS", "twin_lin": "SpecTwin-Lin", "twin_mlp": "SpecTwin-MLP"}
REG_METHODS = ["pls", "ridge", "svm", "gpr", "lwpls", "lgbm", "knn", "cnn", "transformer", "twin_pls", "twin_lin", "twin_mlp"]
CLS_METHODS = ["pls", "ridge", "svm", "lgbm", "knn", "cnn", "transformer", "twin_pls", "twin_lin", "twin_mlp"]
NAMES["twin_plsplus"] = "SpecTwin-PLS+ (post hoc)"
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354}
summary = {}


def fname(task, method):
    return R / "e1" / (task.name.replace(":", "__").replace("[", "_").replace("]", "").replace("=", "-").replace(",", "_") + f"__{method}.json")


def preds_of(res):
    """Pooled (y, pred) over folds; seed-averaged for multi-seed methods."""
    if "folds" in res:
        fo = res["folds"]
        return np.concatenate([f["y"] for f in fo]), np.concatenate([f["pred"] for f in fo])
    seeds = res["seeds"]
    ys = np.concatenate([f["y"] for f in seeds[0]["folds"]])
    if res["kind"] == "cls":
        P = np.stack([np.concatenate([f["pred"] for f in s["folds"]]) for s in seeds])
        from collections import Counter
        p = np.array([Counter(col).most_common(1)[0][0] for col in P.T])
        return ys, p
    P = np.stack([np.concatenate([np.asarray(f["pred"], float) for f in s["folds"]]) for s in seeds])
    return ys, P.mean(0)


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj


def randomization_test(e1, e2, n=10000, seed=0):
    d = e1 ** 2 - e2 ** 2
    rng = np.random.default_rng(seed)
    obs = d.mean()
    sims = (rng.choice([-1, 1], (n, len(d))) * d).mean(1)
    return float((np.abs(sims) >= abs(obs)).mean())


def e1_tables():
    rows, cls_rows, raw = [], [], {}
    for task in E1_TASKS:
        methods = REG_METHODS if task.kind == "reg" else CLS_METHODS
        r = {}
        for m in methods:
            fp = fname(task, m)
            if fp.exists():
                res = json.loads(fp.read_text())
                r[m] = res["metrics"]["RMSEP" if task.kind == "reg" else "accuracy"]
                raw[(task.name, m)] = res
        (rows if task.kind == "reg" else cls_rows).append(dict(task=task.name, **r))
    df = pd.DataFrame(rows).set_index("task"); dc = pd.DataFrame(cls_rows).set_index("task")
    df.to_csv(T / "e1_regression_rmsep.csv"); dc.to_csv(T / "e1_classification_accuracy.csv")
    complete = df.dropna(axis=0, how="any")
    ranks = complete.rank(axis=1, method="average")
    avg_rank = ranks.mean().sort_values()
    out = dict(n_tasks_complete=int(len(complete)), avg_rank={NAMES[k]: float(v) for k, v in avg_rank.items()})
    if len(complete) >= 3 and complete.shape[1] >= 3:
        fr = stats.friedmanchisquare(*[complete[c].values for c in complete.columns])
        k, N = complete.shape[1], len(complete)
        out["friedman_chi2"], out["friedman_p"] = float(fr.statistic), float(fr.pvalue)
        out["nemenyi_CD"] = float(Q05.get(k, 3.3) * np.sqrt(k * (k + 1) / (6 * N)))
    # geometric-mean RMSEP ratio to PLS
    rel = np.exp(np.log(complete.div(complete["pls"], axis=0)).mean())
    out["geomean_rmsep_ratio_vs_PLS"] = {NAMES[k]: float(v) for k, v in rel.items()}
    # Wilcoxon (Holm) SpecTwin-PLS vs each baseline, on log RMSEP ratios
    tests = {}
    for ref in ["twin_pls", "twin_mlp"]:
        if ref not in complete:
            continue
        ps, names = [], []
        for c in complete.columns:
            if c.startswith("twin"):
                continue
            lr = np.log(complete[ref] / complete[c])
            w = stats.wilcoxon(lr)
            ps.append(w.pvalue); names.append(c)
        adj = holm(ps)
        tests[NAMES[ref]] = {NAMES[n]: dict(median_log_ratio=float(np.median(np.log(complete[ref] / complete[n]))),
                                            wins=int((complete[ref] < complete[n]).sum()), losses=int((complete[ref] > complete[n]).sum()),
                                            p_holm=float(a)) for n, a in zip(names, adj)}
    out["wilcoxon_holm"] = tests
    # Bayesian signed-rank vs PLS (ROPE = 1% relative RMSEP)
    try:
        import baycomp
        for ref in ["twin_pls", "twin_mlp"]:
            relx = (complete[ref] - complete["pls"]) / complete["pls"]
            p = baycomp.SignedRankTest.probs(-relx.values, np.zeros(len(relx)), rope=0.01)
            out.setdefault("bayes_signed_rank_vs_PLS", {})[NAMES[ref]] = dict(P_better=float(p[0]), P_rope=float(p[1]), P_worse=float(p[2]))
    except Exception as e:
        out["bayes_error"] = repr(e)
    # per-task randomization tests: SpecTwin-PLS vs PLS and vs best baseline
    per_task = []
    for task in E1_TASKS:
        if task.kind != "reg" or (task.name, "twin_pls") not in raw or (task.name, "pls") not in raw:
            continue
        yt, pt = preds_of(raw[(task.name, "twin_pls")]); yp, pp = preds_of(raw[(task.name, "pls")])
        if len(yt) != len(yp):
            continue
        p = randomization_test(pt - yt, pp - yp)
        per_task.append(dict(task=task.name, rmsep_twin=float(np.sqrt(np.mean((pt - yt) ** 2))), rmsep_pls=float(np.sqrt(np.mean((pp - yp) ** 2))), p=p))
    pt_df = pd.DataFrame(per_task)
    if len(pt_df):
        out["randomization_vs_PLS"] = dict(sig_better=int(((pt_df.p < 0.05) & (pt_df.rmsep_twin < pt_df.rmsep_pls)).sum()),
                                           sig_worse=int(((pt_df.p < 0.05) & (pt_df.rmsep_twin > pt_df.rmsep_pls)).sum()), n=int(len(pt_df)))
        pt_df.to_csv(T / "e1_randomization_vs_pls.csv", index=False)
    summary["E1"] = out
    # markdown
    show = df.copy(); show.columns = [NAMES[c] for c in show.columns]
    md = ["| Task | " + " | ".join(show.columns) + " |", "|---|" + "---|" * len(show.columns)]
    for t, row in show.iterrows():
        best = row.min()
        md.append(f"| {t} | " + " | ".join(("**%.4g**" % v if v == best else "%.4g" % v) if pd.notna(v) else "–" for v in row.values) + " |")
    md.append("| **Average rank** | " + " | ".join(f"{out['avg_rank'].get(c, float('nan')):.2f}" for c in show.columns) + " |")
    (T / "e1_regression.md").write_text("\n".join(md))
    show = dc.copy(); show.columns = [NAMES[c] for c in show.columns]
    md = ["| Task | " + " | ".join(show.columns) + " |", "|---|" + "---|" * len(show.columns)]
    for t, row in show.iterrows():
        md.append(f"| {t} | " + " | ".join("%.3f" % v if pd.notna(v) else "–" for v in row.values) + " |")
    (T / "e1_classification.md").write_text("\n".join(md))
    return df, dc


def e2_tables():
    rows = []
    for fp in sorted((R / "e2").glob("*.json")):
        for r in json.loads(fp.read_text()):
            rows.append(dict(task=fp.stem.replace("__", ":"), **r))
    if not rows:
        return
    df = pd.DataFrame(rows)
    piv = df.groupby(["task", "n", "method"]).RMSEP.mean().unstack("method")
    piv.to_csv(T / "e2_label_efficiency.csv")
    rel = piv.div(piv["pls"], axis=0)
    by_n = np.exp(np.log(rel).groupby(level="n").mean())
    summary["E2"] = dict(geomean_ratio_vs_PLS_by_n={int(n): {NAMES.get(k, k): float(v) for k, v in row.items()} for n, row in by_n.iterrows()})
    wins = {}
    for n, g in piv.groupby(level="n"):
        for m in [c for c in piv.columns if c != "pls"]:
            w = stats.wilcoxon(np.log(g[m] / g["pls"])) if len(g) >= 5 else None
            wins.setdefault(int(n), {})[NAMES.get(m, m)] = dict(wins=int((g[m] < g["pls"]).sum()), of=int(len(g)), p=float(w.pvalue) if w else None)
    summary["E2"]["wins_vs_PLS"] = wins
    md = ["| Task | n | " + " | ".join(NAMES.get(c, c) for c in piv.columns) + " |", "|---|---|" + "---|" * len(piv.columns)]
    for (t, n), row in piv.iterrows():
        best = row.min()
        md.append(f"| {t} | {n} | " + " | ".join(("**%.4g**" % v if v == best else "%.4g" % v) for v in row.values) + " |")
    (T / "e2_label_efficiency.md").write_text("\n".join(md))


def e3_tables():
    rows = []
    for fp in sorted((R / "e3").glob("*.json")):
        for r in json.loads(fp.read_text()):
            fam = fp.stem
            for m, v in r["metrics"].items():
                if v is None:
                    continue
                rows.append(dict(family=fam, scenario=r["scenario"], rep=r["rep"], kind=r["kind"], method=m,
                                 value=v["RMSEP"] if r["kind"] == "reg" else v["accuracy"], bias=v.get("bias")))
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(T / "e3_all_runs.csv", index=False)
    fam = df.groupby(["family", "kind", "method"]).value.mean().unstack("method")
    fam.to_csv(T / "e3_by_family.csv")
    # gap closed vs oracle where available (regression)
    out = {}
    reg = df[df.kind == "reg"].pivot_table(index=["family", "scenario", "rep"], columns="method", values="value")
    sf = ["DMC", "CORAL", "diPLS", "NC", "TTPA-fleet", "TTPA-chem"]
    tests = {}
    for ref in ["SpecTwin-TTPA"]:
        for other in [c for c in sf if c in reg.columns]:
            sub = reg[[ref, other]].dropna()
            if len(sub) < 6:
                continue
            w = stats.wilcoxon(np.log(sub[ref] / sub[other]))
            tests[other] = dict(n=int(len(sub)), wins=int((sub[ref] < sub[other]).sum()), losses=int((sub[ref] > sub[other]).sum()),
                                median_log_ratio=float(np.median(np.log(sub[ref] / sub[other]))), p=float(w.pvalue))
    adj = holm([v["p"] for v in tests.values()])
    for (k, v), a in zip(tests.items(), adj):
        v["p_holm"] = float(a)
    out["SpecTwin-TTPA_vs"] = tests
    if "Oracle (target labels)" in reg.columns:
        g = reg.dropna(subset=["Oracle (target labels)", "NC"])
        gap = {}
        for m in [c for c in reg.columns if c not in ("Oracle (target labels)",)]:
            gg = g.dropna(subset=[m])
            denom = gg["NC"] - gg["Oracle (target labels)"]
            ok = denom > 1e-9
            gap[m] = float(np.median(((gg["NC"] - gg[m]) / denom)[ok])) if ok.any() else None
        out["median_gap_closed_vs_oracle"] = gap
    summary["E3"] = out
    md = ["| Scenario family | " + " | ".join(fam.columns) + " |", "|---|" + "---|" * len(fam.columns)]
    for (f, k), row in fam.iterrows():
        vals = row.values.astype(float)
        best = np.nanmin(vals) if k == "reg" else np.nanmax(vals)
        md.append(f"| {f} ({'RMSEP' if k == 'reg' else 'accuracy'}) | " + " | ".join(("**%.4g**" % v if v == best else "%.4g" % v) if np.isfinite(v) else "–" for v in vals) + " |")
    (T / "e3_transfer.md").write_text("\n".join(md))


def posthoc_table():
    rows = []
    for task in E1_TASKS:
        r = {}
        for m in ["pls", "ridge", "twin_pls", "twin_plsplus"]:
            fp = fname(task, m)
            if fp.exists():
                res = json.loads(fp.read_text()); r[m] = res["metrics"]["RMSEP" if task.kind == "reg" else "accuracy"]
                if m == "twin_plsplus":
                    ch = [f["hp"]["input"] for f in res["folds"]]; r["twin chosen (folds)"] = f"{ch.count('twin')}/{len(ch)}"
        rows.append(dict(task=task.name, kind=task.kind, **r))
    df = pd.DataFrame(rows).set_index("task")
    df.to_csv(T / "e1_posthoc_plsplus.csv")
    reg = df[(df.kind == "reg")].dropna(subset=["pls", "twin_plsplus"])
    if len(reg):
        lr = np.log(reg["twin_plsplus"] / reg["pls"])
        w = stats.wilcoxon(lr) if (lr != 0).sum() >= 5 else None
        summary["E1_posthoc"] = dict(n=int(len(reg)), geomean_ratio_vs_PLS=float(np.exp(lr.mean())), wins=int((lr < 0).sum()), losses=int((lr > 0).sum()),
                                     ties=int((lr == 0).sum()), wilcoxon_p=float(w.pvalue) if w else None,
                                     twin_chosen_any=int(sum(1 for v in reg["twin chosen (folds)"] if not str(v).startswith("0/"))))


def e4_tables():
    fp = R / "e4" / "e4_results.json"
    if not fp.exists():
        return
    d = json.loads(fp.read_text())
    na = {}
    fpn = R / "e4" / "e4_nonamortised.json"
    if fpn.exists():
        na = {r["system"].split(" ")[0]: r for r in json.loads(fpn.read_text())}
    md = ["| System (ground truth) | SpecTwin (amortised) | SpecTwin decoder (non-amortised) | PCA | NMF | MCR-ALS | Spearman(kappa(J), error) |", "|---|---|---|---|---|---|---|"]
    for r in d["a_recovery"]:
        n = na.get(r["system"], {})
        f = lambda v: f"{v[0]:.2f} / {v[1]:.2f}"
        md.append(f"| SimFood-RT {r['system']} (K={r['K']}) | {f(r['SpecTwin'])} | " + (f"{n['affine_r2']:.2f} / {n['perm_corr']:.2f}" if n else "–") +
                  f" | {f(r['PCA'])} | {f(r['NMF'])} | {f(r['MCR-ALS'])} | rho={r['kappa_spearman'][0]:.2f} (p={r['kappa_spearman'][1]:.2g}) |")
    for r in d["d_real"]:
        key = r["dataset"].split(" ")[0]
        n = na.get(key, {})
        b = r["baselines"]
        f = lambda v: f"{v[0]:.2f} / {v[1]:.2f}"
        md.append(f"| {r['dataset']} | {r['SpecTwin_affine_r2']:.2f} / {r['SpecTwin_perm_corr']:.2f} | " + (f"{n['affine_r2']:.2f} / {n['perm_corr']:.2f}" if n else "–") +
                  f" | {f(b['PCA'])} | {f(b['NMF'])} | {f(b['MCR-ALS'])} | – |")
    md.append("")
    md.append("Cells: affine R^2 / permutation-matched correlation between recovered and true composition fractions.")
    md.append("")
    md.append("| Dictionary learned from | affine R^2 (emulsion test) | matched corr. |")
    md.append("|---|---|---|")
    for r in d["b_pooling"]:
        md.append(f"| {r['pool']} | {r['affine_r2']:.3f} | {r['perm_corr']:.3f} |")
    ins = d.get("instrument")
    if ins:
        # twin convention: positive shift moves features to longer wavelength = minus the simulator's pixel offset
        rt = [-v for v in ins["rel_true"]]
        md.append(""); md.append(f"Instrument recovery (SimFood-RT dispersive units, shift relative to unit 0, nm): true {np.round(rt, 2).tolist()} vs SpecTwin {np.round(ins['rel_est'], 2).tolist()}")
    if "e_bands" in d:
        e = d["e_bands"]
        md.append(f"Band assignment (pooled twin, {e['n_peaks']} atom peaks): median distance to nearest literature band {e['median_dist_nm']:.1f} nm "
                  f"(random positions {e['random_median_dist_nm']:.1f} nm); within 15 nm: {100 * e['frac_within_15nm']:.0f}% (random {100 * e['random_frac_within_15nm']:.0f}%).")
    (T / "e4_e5_identifiability.md").write_text("\n".join(md))


def e6_table():
    fp = R / "e6e10" / "e6.json"
    if not fp.exists():
        return
    df = pd.DataFrame(json.loads(fp.read_text()))
    piv = df.pivot_table(index=["task", "method"], columns="mode", values="RMSEP")[["full", "x2", "x4", "x8", "trim20"]]
    piv.to_csv(T / "e6_resolution.csv")
    md = ["| Task | Method | full | 1/2 points | 1/4 | 1/8 | range -20% |", "|---|---|---|---|---|---|---|"]
    for (t, m), row in piv.iterrows():
        md.append(f"| {t} | {m} | " + " | ".join(f"{v:.3g}" for v in row.values) + " |")
    (T / "e6_resolution.md").write_text("\n".join(md))
    rel = piv.div(piv["full"], axis=0)
    summary["E6_median_rmsep_ratio_to_full"] = {m: {c: float(rel.xs(m, level="method")[c].median()) for c in ["x2", "x4", "x8", "trim20"]}
                                               for m in piv.index.get_level_values("method").unique()}


def e7_table():
    fp = R / "e7e8" / "e7.json"
    if not fp.exists():
        return
    rows = []
    for t in json.loads(fp.read_text()):
        for m, v in t["methods"].items():
            for a, mt in v.items():
                rows.append(dict(task=t["task"], method=m, alpha=float(a), **{k: mt.get(k) for k in ["PICP", "MPIW_norm", "interval_score_norm", "worst_group_PICP", "frac_infinite"]}))
    df = pd.DataFrame(rows); df.to_csv(T / "e7_uq_all.csv", index=False)
    g = df.groupby(["method", "alpha"]).agg(PICP=("PICP", "mean"), PICP_min=("PICP", "min"), MPIW=("MPIW_norm", "mean"), IS=("interval_score_norm", "mean"),
                                            worst_group=("worst_group_PICP", "mean"), n=("task", "count"))
    md = ["| Method | nominal | mean PICP | min PICP over tasks | mean width (/range) | mean interval score | mean worst-group PICP |", "|---|---|---|---|---|---|---|"]
    for (m, a), r in g.iterrows():
        md.append(f"| {m} | {1 - a:.2f} | {r.PICP:.3f} | {r.PICP_min:.3f} | {r.MPIW:.3f} | {r.IS:.3f} | " + (f"{r.worst_group:.3f}" if pd.notna(r.worst_group) else "–") + " |")
    (T / "e7_uq.md").write_text("\n".join(md))
    summary["E7"] = {f"{m}@{1 - a:.2f}": dict(PICP=float(r.PICP), width=float(r.MPIW), IS=float(r.IS)) for (m, a), r in g.iterrows()}


def e8_table():
    fp = R / "e7e8" / "e8.json"
    if not fp.exists():
        return
    rows = []
    for sc in json.loads(fp.read_text()):
        fam = sc["scenario"].split(" unseen")[0].split(" authenticity")[0].split(" instrument")[0]
        for m, v in sc["metrics"].items():
            rows.append(dict(family=fam, scenario=sc["scenario"], method=m, **v))
    df = pd.DataFrame(rows); df.to_csv(T / "e8_ood_all.csv", index=False)
    g = df.groupby(["family", "method"]).agg(AUROC=("AUROC", "mean"), FPR95=("FPR_at_95TPR", "mean"), FDR=("BH_FDR", "mean"), power=("BH_power", "mean"), n=("scenario", "count"))
    md = ["| Scenario family | Scorer | AUROC | FPR@95%TPR | BH realised FDR (target 0.10) | BH power |", "|---|---|---|---|---|---|"]
    for (f, m), r in g.iterrows():
        md.append(f"| {f} (n={int(r.n)}) | {m} | {r.AUROC:.3f} | {r.FPR95:.3f} | {r.FDR:.3f} | {r.power:.3f} |")
    (T / "e8_ood.md").write_text("\n".join(md))
    summary["E8"] = {f"{f} | {m}": dict(AUROC=float(r.AUROC), FDR=float(r.FDR), power=float(r.power)) for (f, m), r in g.iterrows()}


def e11_table():
    files = sorted((R / "e11").glob("*.json"))
    if not files:
        return
    res = {fp.stem: json.loads(fp.read_text()) for fp in files}
    ref = res.get("abl_full")
    md = ["| Variant | geo-mean RMSEP ratio vs full (reg. tasks) | mean accuracy (cls) | mean recon. error | E2 n=20 RMSEP ratio |", "|---|---|---|---|---|"]
    out = {}
    for v, r in res.items():
        reg = [k for k in r["e1"] if not any(c in k for c in ["grapes", "mayonnaise"])]
        cls = [k for k in r["e1"] if any(c in k for c in ["grapes", "mayonnaise"])]
        gm = float(np.exp(np.mean([np.log(r["e1"][k] / ref["e1"][k]) for k in reg]))) if ref else float("nan")
        acc = float(np.mean([r["e1"][k] for k in cls]))
        rec = float(np.mean(list(r["recon"].values())))
        e2 = float(np.exp(np.mean([np.log(r["e2"][k] / ref["e2"][k]) for k in r["e2"]]))) if ref else float("nan")
        out[v] = dict(rmsep_ratio=gm, acc=acc, recon=rec, e2_ratio=e2)
        md.append(f"| {v} | {gm:.3f} | {acc:.3f} | {rec:.4f} | {e2:.3f} |")
    (T / "e11_ablations.md").write_text("\n".join(md))
    summary["E11_table"] = out


def simple_json(name, key):
    fp = R / name
    if fp.exists():
        summary[key] = json.loads(fp.read_text())


if __name__ == "__main__":
    e1_tables(); e2_tables(); e3_tables(); posthoc_table(); e4_tables(); e6_table(); e7_table(); e8_table(); e11_table()
    simple_json("e4/e4_results.json", "E4_E5")
    simple_json("e6e10/e6.json", "E6"); simple_json("e6e10/e10.json", "E10")
    simple_json("e7e8/e7.json", "E7"); simple_json("e7e8/e8.json", "E8")
    e11 = {}
    for fp in sorted((R / "e11").glob("*.json")):
        e11[fp.stem] = json.loads(fp.read_text())
    if e11:
        summary["E11"] = e11
    (T / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: (v if k in ("E1", "E2", "E3") else "...") for k, v in summary.items()}, indent=1, default=float)[:6000])
