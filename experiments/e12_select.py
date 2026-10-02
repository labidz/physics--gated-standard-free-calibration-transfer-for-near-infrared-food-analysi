"""Apply the Amendment A selection rule to the development results (mechanical; no confirmatory data)."""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from e12_analysis import load

CANDS = ["TTPA4", "TTPA4-abstain", "TTPA4-nogof", "TTPA4-pargate-off", "TTPA4-safe", "TTPA4-safe-abstain"]
ORDER = {"TTPA4": 0, "TTPA4-abstain": 1, "TTPA4-nogof": 2, "TTPA4-pargate-off": 3, "TTPA4-safe": 4, "TTPA4-safe-abstain": 5}
REFS = ["TTPA4-nobroad", "TTPA4-broadtest", "TTPA4-nogate", "TTPA4-perp", "TTPA3", "DMC", "diPLS", "uDOP", "TCR", "CT-VPdtw", "OT-DA", "DANN", "CORAL"]

if __name__ == "__main__":
    df = load("dev")
    W = df.pivot_table(index=["family", "scenario", "rep"], columns="method", values="err").reset_index()
    rows, per = [], {}
    for m in CANDS + REFS:
        if m not in W:
            continue
        sub = W.dropna(subset=[m, "NC"]); sub = sub[(sub[m] > 0) & (sub.NC > 0)]
        fam = np.log(sub[m] / sub["NC"]).groupby(sub.family).mean()
        per[m] = 100 * (np.exp(fam) - 1)
        rows.append(dict(method=m, families=len(fam), mean_change_vs_NC=100 * (np.exp(fam.mean()) - 1),
                         families_worse_5pct=int((fam > np.log(1.05)).sum()), worst=100 * (np.exp(fam.max()) - 1)))
    R = pd.DataFrame(rows)
    C = R[R.method.isin(CANDS)]
    pool = C[C.families_worse_5pct == 0]
    if not len(pool):
        pool = C[C.families_worse_5pct == C.families_worse_5pct.min()]
    best = pool.mean_change_vs_NC.min()
    close = pool[np.log(1 + pool.mean_change_vs_NC / 100) <= np.log(1 + best / 100) + np.log(1.005)]
    chosen = sorted(close.method, key=ORDER.get)[0]
    txt = ["| Variant | families | mean change vs NC | families >5% worse than NC | worst family |", "|---|---|---|---|---|"]
    for _, r in R.iterrows():
        txt.append(f"| {r.method}{' (candidate)' if r.method in CANDS else ''} | {r.families} | {r.mean_change_vs_NC:+.1f}% | {r.families_worse_5pct} | {r.worst:+.1f}% |")
    txt += ["", f"Chosen by the Amendment A rule: **{chosen}**", "", "Per-family change vs NC (%):", "",
            pd.DataFrame(per).round(1).to_markdown()]
    out = "\n".join(txt)
    (ROOT / "results" / "tables" / "e12_dev_selection.md").write_text(out + "\n")
    print(out)
