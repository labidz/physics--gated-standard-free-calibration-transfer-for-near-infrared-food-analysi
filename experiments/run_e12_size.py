"""E12-size: empirical false-adaptation rate of the TTPA v4 significance gates.

For each development dataset, 20 random splits of one instrument's spectra into a pseudo-source and a
pseudo-target (same instrument, same population: the null hypothesis of Proposition 3). The gates should
fire in at most ~5% of splits. No reference values are used.
"""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spectwin.bench import load
from spectwin.transfer2 import TTPA4


def sets():
    out = []
    d = load("corn"); inst = d.domains["instrument"]
    for u in ["m5", "mp5", "mp6"]:
        out.append((f"corn {u}", d.X[inst == u], d.axis))
    d = load("ch2021"); out.append(("ch2021 unit 1 calibration", d.X[d.domains["role"] == "cal"], d.axis))
    d = load("tablets"); out.append(("tablets instrument 1", d.X[d.domains["instrument"] == "1"], d.axis))
    for name in ["wheat", "cassava", "grapes", "mango", "plum", "ch2018"]:
        d = load(name); X = d.X[d.split == "train"] if (d.split == "train").any() else d.X
        if len(X) > 400:
            X = X[np.random.default_rng(0).choice(len(X), 400, replace=False)]
        out.append((name, X, d.axis))
    d = load("ligno"); out.append(("ligno off-line", d.X[d.domains["role"] == "offline"], d.axis))
    for u in [0, 3, 9]:
        z = np.load(ROOT / "data" / "sim" / f"simfood_vnir_unit{u:02d}.npz")
        out.append((f"SimFood unit {u}", z["X"][(z["system"] == "emulsion") & (z["split"] == "train")], z["axis"].astype(float)))
    return out


if __name__ == "__main__":
    res = []
    for name, X, axis in sets():
        X = np.asarray(X, float); n = len(X); na = int(min(max(10, n // 4), 100)); t0 = time.time()
        rec = {k: [] for k in ["theta_fleet", "broad_fleet", "theta_chem", "dperp", "dpar"]}
        for b in range(20):
            perm = np.random.default_rng(1000 + b).permutation(n)
            tt = TTPA4("fleet", seed=b, broaden="test").fit_source(X[perm[na:]], axis, k_min=15).fit_target(X[perm[:na]])
            tt.configure(broaden="test", gate_par=True, gof=False)
            rec["theta_fleet"].append(tt.stats["theta_gate"]); rec["broad_fleet"].append(tt.stats["broad_gate"])
            rec["dperp"].append(tt.stats["dperp_gate"]); rec["dpar"].append(tt.stats["dpar_used"])
            tt.context = "chem"; tt.configure(broaden=False, gate_par=True, gof=True, use_par=False)
            rec["theta_chem"].append(tt.stats["theta_gate"])
        res.append(dict(dataset=name, n=n, n_target=na, **{k + "_rate": float(np.mean(v)) for k, v in rec.items()}))
        print(res[-1], f"({time.time() - t0:.0f}s)", flush=True)
    out = ROOT / "results" / "e12" / "size.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("mean false-adaptation rates:", {k: round(float(np.mean([r[k] for r in res])), 3) for k in res[0] if k.endswith("_rate")})
