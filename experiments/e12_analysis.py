"""E12 analysis: family-level and mixed-model comparisons of standard-free transfer methods.

  python experiments/e12_analysis.py dev|conf|all

Primary endpoint (PROTOCOL Amendment A): replicate-level log RMSEP ratio of TTPA v4 vs each comparator,
linear mixed model with random intercepts for family and for scenario within family; Holm over comparators.
Secondary: family-level wins (one geometric-mean ratio per family), sign test; safety: number of families
in which TTPA v4 is more than 5% worse than no correction.
"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"

PRIMARY = os.environ.get("E12_PRIMARY", "TTPA4")      # the frozen confirmatory configuration (Amendment A)
COMPARATORS = ["NC", "DMC", "CORAL", "diPLS", "uDOP", "TCR", "OT-DA", "DANN", "CT-VPdtw", "TTPA3"]
SHOW = ["NC", "DMC", "CORAL", "diPLS", "uDOP", "TCR", "OT-DA", "DANN", "CT-VPdtw", "TTPA3", PRIMARY]
STD = ["DS (std)", "PDS (std)", "SBC (std)", "Oracle (target labels)"]
ABL = ["TTPA4", "TTPA4-nogof", "TTPA4-abstain", "TTPA4-pargate-off", "TTPA4-nobroad", "TTPA4-broadtest", "TTPA4-nogate",
       "TTPA4-perp", "TTPA4-safe", "TTPA4-safe-abstain"]


def load(stage):
    rows = []
    for f in sorted((ROOT / "results" / "e12" / stage).glob("*.json")):
        fam = f.stem
        if fam.startswith("Published protocol"):
            continue
        for r in json.loads(f.read_text()):
            for m, v in r["metrics"].items():
                if v is None:
                    continue
                err = v["RMSEP"] if r["kind"] == "reg" else 1 - v["accuracy"]
                rows.append(dict(stage=stage, family=fam, scenario=r["scenario"], rep=r["rep"], kind=r["kind"], context=r["context"],
                                 method=m, err=err, theta_gate=r["info"][PRIMARY]["theta_gate"], dperp_gate=r["info"][PRIMARY]["dperp_gate"],
                                 dpar_used=r["info"][PRIMARY].get("dpar_used", False)))
    return pd.DataFrame(rows)


def mixed(lr, fam, scen):
    """Random-intercept model log-ratio ~ 1 + (1|family) + (1|family:scenario). Returns est, lo, hi, p."""
    import statsmodels.formula.api as smf
    d = pd.DataFrame(dict(lr=lr, family=fam, scenario=scen))
    try:
        if d.family.nunique() < 2:
            raise ValueError
        m = smf.mixedlm("lr ~ 1", d, groups="family", vc_formula={"scenario": "0 + C(scenario)"}).fit(reml=True, method="lbfgs")
        est, se, p = m.params["Intercept"], m.bse["Intercept"], m.pvalues["Intercept"]
    except Exception:
        from scipy.stats import ttest_1samp
        g = d.groupby("family").lr.mean()
        est, se = g.mean(), g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.nan
        p = ttest_1samp(g, 0).pvalue if len(g) > 1 else np.nan
    return est, est - 1.96 * se, est + 1.96 * se, p


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def analyse(df, label, ours=PRIMARY):
    lines = [f"## {label}", ""]
    W = df.pivot_table(index=["family", "scenario", "rep", "kind"], columns="method", values="err").reset_index()
    fams = sorted(W.family.unique(), key=lambda f: (df[df.family == f].context.iloc[0], f))
    # --- per-family table (regression: mean RMSEP over replicates; classification: accuracy)
    cols = [m for m in SHOW + STD if m in W.columns]
    lines += ["| Family | context | runs | " + " | ".join(cols) + " |", "|---|---|---|" + "---|" * len(cols)]
    for f in fams:
        w = W[W.family == f]; kind = w.kind.iloc[0]
        vals = {m: (w[m].mean() if kind == "reg" else 1 - w[m].mean()) for m in cols if m in w and w[m].notna().any()}
        sf = [m for m in SHOW if m in vals]
        best = min(sf, key=lambda m: vals[m]) if kind == "reg" else max(sf, key=lambda m: vals[m])
        cell = lambda m: ("–" if m not in vals else (f"**{vals[m]:.4g}**" if m == best else f"{vals[m]:.4g}"))
        lines.append(f"| {f}{' (acc.)' if kind == 'cls' else ''} | {df[df.family == f].context.iloc[0]} | {len(w)} | " + " | ".join(cell(m) for m in cols) + " |")
    lines += ["", "Bold: best method that uses no transfer standards and no target labels.", ""]
    # --- TTPA v4 vs comparators
    res = []
    for b in COMPARATORS + [a for a in ABL if a != ours]:
        if b not in W.columns:
            continue
        sub = W.dropna(subset=[ours, b])
        sub = sub[(sub[ours] > 0) & (sub[b] > 0)]
        if len(sub) == 0:
            continue
        lr = np.log(sub[ours].values / sub[b].values)
        est, lo, hi, p = mixed(lr, sub.family.values, (sub.family + "|" + sub.scenario).values)
        fam = pd.Series(lr).groupby(sub.family.values).mean()
        nz = fam[np.abs(fam) > 1e-9]
        sp = binomtest(int((nz < 0).sum()), len(nz)).pvalue if len(nz) else np.nan
        worse5 = int((fam > np.log(1.05)).sum())
        res.append(dict(comp=b, n=len(sub), fams=len(fam), est=est, lo=lo, hi=hi, p=p, fw=int((fam < -1e-9).sum()), fl=int((fam > 1e-9).sum()), sp=sp, worse5=worse5))
    R = pd.DataFrame(res)
    main = R.comp.isin(COMPARATORS)
    R.loc[main, "p_holm"] = holm(R.loc[main, "p"].values)
    lines += [f"| {ours} vs | runs | families | change in error, mixed model [95% CI] | p (Holm) | family W/L | sign-test p | families >5% worse |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        pc = lambda v: f"{100 * (np.exp(v) - 1):+.1f}%"
        ph = f"{r.p_holm:.2g}" if not np.isnan(r.get("p_holm", np.nan)) else f"{r.p:.2g} (ablation)"
        lines.append(f"| {r.comp} | {r.n} | {r.fams} | {pc(r.est)} [{pc(r.lo)}, {pc(r.hi)}] | {ph} | {r.fw}/{r.fl} | {r.sp:.2g} | {r.worse5} |")
    lines += ["", "Change in error: geometric-mean ratio of TTPA v4 error to comparator error minus one (negative = TTPA v4 better).", ""]
    # --- gate behaviour
    G = df[df.method == ours].groupby("family")[["theta_gate", "dperp_gate", "dpar_used"]].mean()
    lines += ["| Family | operator applied | out-of-span mean shift corrected | in-span mean shift corrected |", "|---|---|---|---|"]
    for f in fams:
        lines.append(f"| {f} | {100 * G.loc[f, 'theta_gate']:.0f}% | {100 * G.loc[f, 'dperp_gate']:.0f}% | {100 * G.loc[f, 'dpar_used']:.0f}% |")
    # --- best-method ranks per family (standard-free methods only)
    rk = []
    for f in fams:
        w = W[W.family == f]
        sf = [m for m in SHOW if m in w and w[m].notna().all()]
        mean_lr = {m: np.mean(np.log(w[m].clip(lower=1e-12) / w["NC"].clip(lower=1e-12))) for m in sf}
        order = sorted(mean_lr, key=mean_lr.get)
        rk.append({m: order.index(m) + 1 for m in sf})
    RK = pd.DataFrame(rk, index=fams)
    lines += ["", "Average rank across families (standard-free methods; 1 = best): " +
              ", ".join(f"{m} {RK[m].mean():.2f}" for m in RK.mean().sort_values().index), ""]
    return "\n".join(lines), R, RK


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "dev"
    stages = ["dev", "conf"] if stage == "all" else [stage]
    out = []
    for s in stages:
        df = load(s)
        if len(df) == 0:
            continue
        txt, R, RK = analyse(df, {"dev": "Development families", "conf": "Confirmatory families (pre-registered)"}[s])
        out.append(txt)
        R.to_csv(OUT / f"e12_{s}_comparisons.csv", index=False)
    if stage == "all":
        df = pd.concat([load(s) for s in ["dev", "conf"]])
        txt, R, RK = analyse(df, "All families (development + confirmatory)")
        out.append(txt); R.to_csv(OUT / "e12_all_comparisons.csv", index=False)
    (OUT / f"e12_{stage}.md").write_text("\n\n".join(out) + "\n")
    print("\n\n".join(out))
