"""E2: label efficiency (n_train in {10, 20, 50, 100, 200}, 5 random draws, fixed test sets)."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
torch.set_num_threads(1)
from spectwin.bench import load
from spectwin import baselines as B
from spectwin.deep import m_deep
from spectwin.chemo import reg_metrics
from spectwin.twin_run import latents_for, phys_features, mlp_head, corrected_native

OUT = ROOT / "results" / "e2"; OUT.mkdir(parents=True, exist_ok=True)
SIZES = [10, 20, 50, 100, 200]
METHODS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["pls", "ridge", "cnn", "twin_pls", "twin_mlp"]


def setup(name, target):
    d = load(name)
    y = d.targets[target].astype(float)
    ok = np.isfinite(y)
    if name == "plum":
        fold = d.domains["fold"]
        test = np.where(ok & (fold == "1"))[0]; pool = np.where(ok & (fold != "1"))[0]
    elif name == "corn":
        m5 = np.where(d.domains["instrument"] == "m5")[0]
        perm = np.random.default_rng(7).permutation(m5)
        test, pool = np.sort(perm[:20]), np.sort(perm[20:])
    else:
        test = np.where(ok & (d.split == "test"))[0]; pool = np.where(ok & (d.split == "train"))[0]
    return d, y, pool, test


TASKS = [("mango", "DM"), ("plum", "SSC"), ("ch2018", "protein"), ("sorghum", "CP"), ("pears", "Brix"), ("wheat", "protein"), ("corn", "moisture")]

for name, target in TASKS:
    fp = OUT / f"{name}__{target}.json"
    res = json.loads(fp.read_text()) if fp.exists() else []
    done = {(r["method"], r["n"], r["draw"]) for r in res}
    d, y, pool, test = setup(name, target)
    lat = None
    if any(m.startswith("twin") for m in METHODS):
        lat, valid = latents_for(name, pool, np.concatenate([pool, test]), "real", cache_key="e2pool", seed=0)
        pos = {r: i for i, r in enumerate(lat["rows"])}
        C = corrected_native(d, lat, valid, lat["rows"]); H = np.column_stack([lat["h"], phys_features(lat)])
    t0 = time.time()
    for n in SIZES:
        if n >= len(pool):
            continue
        for draw in range(5):
            rng = np.random.default_rng(1000 * n + draw)
            ug = np.unique(d.groups[pool]); rng.shuffle(ug)
            chosen = []
            for gname in ug:                      # draw whole groups (no replicate leakage)
                chosen += list(pool[d.groups[pool] == gname])
                if len(chosen) >= n:
                    break
            tr = np.array(chosen[:n])
            for m in METHODS:
                if (m, n, draw) in done:
                    continue
                g = d.groups[tr]
                if m == "pls":
                    p, _ = B.m_pls(d.X[tr].astype(float), y[tr], g, d.X[test].astype(float), "reg", seed=draw)
                elif m == "ridge":
                    p, _ = B.m_ridge(d.X[tr].astype(float), y[tr], g, d.X[test].astype(float), "reg", seed=draw)
                elif m == "cnn":
                    p, _ = m_deep("cnn", d.X[tr].astype(float), y[tr], g, d.X[test].astype(float), "reg", seed=draw)
                elif m == "twin_pls":
                    itr = [pos[r] for r in tr]; ite = [pos[r] for r in test]
                    p, _ = B.m_pls(C[itr], y[tr], g, C[ite], "reg", seed=draw)
                elif m == "twin_mlp":
                    itr = [pos[r] for r in tr]; ite = [pos[r] for r in test]
                    p, _ = mlp_head(H[itr], y[tr], g, H[ite], "reg", seed=draw)
                mt = reg_metrics(y[test], p)
                res.append(dict(method=m, n=n, draw=draw, RMSEP=mt["RMSEP"], R2=mt["R2"], bias=mt["bias"]))
                fp.write_text(json.dumps(res))
        summ = {m: np.mean([r["RMSEP"] for r in res if r["method"] == m and r["n"] == n]) for m in METHODS}
        print(f"{name}:{target} n={n:4d} " + "  ".join(f"{m}={v:.4g}" for m, v in summ.items()) + f" ({time.time()-t0:.0f}s)", flush=True)
