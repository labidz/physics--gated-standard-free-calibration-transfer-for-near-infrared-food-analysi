"""Post-hoc sensitivity analysis S1 (not pre-registered): one corrupted target spectrum in IDRC-2016 (A2.CAL sample
187; label-free Q-residual screen against the master PCA: Q = 163 x the master 99th percentile, all other IDRC
target spectra < 4 x). Re-evaluates the confirmatory analysis without the 7 runs in which it was target data."""
import json, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("E12_PRIMARY", "TTPA4-safe-abstain")
import e12_analysis as E

df = E.load("conf")
fam = "IDRC-2016 wheat fleet A (same samples)"
bad = []
for rep in range(10):
    perm = np.random.default_rng(100 + rep).permutation(248)
    if 187 in perm[124:]:
        bad.append(rep)
mask = (df.family == fam) & (df.scenario == "IDRC A1->A2") & (df.rep.isin(bad))
txt, R, RK = E.analyse(df[~mask], f"Confirmatory families, excluding the {len(bad)} IDRC A1->A2 runs with the corrupted spectrum in the target data")
(ROOT / "results" / "tables" / "e12_conf_sensitivity_S1.md").write_text(txt + "\n")
print(txt)
