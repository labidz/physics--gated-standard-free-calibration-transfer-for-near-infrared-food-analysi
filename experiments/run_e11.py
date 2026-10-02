"""E11: ablations and corpus-size scaling of SpecTwin pretraining.

For each pretrained variant (results/models/twin_vnir_<variant>.pt) evaluate:
  - E1 subset (in-corpus official-split tasks, SpecTwin-PLS head): RMSEP / accuracy
  - physics fit: mean reconstruction error on held-out test spectra
  - E2 small-n (n = 20, 5 draws) SpecTwin-PLS RMSEP on mango, sorghum, pears
"""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
torch.set_num_threads(1)
from spectwin.bench import load
from spectwin.baselines import m_pls
from spectwin.chemo import reg_metrics, cls_metrics
from spectwin.grid import to_global
from spectwin.twin import encode_np
from spectwin.twin_run import load_twin, ds_ids, scale_of, corrected_native

OUT = ROOT / "results" / "e11"; OUT.mkdir(parents=True, exist_ok=True)
TASKS = [("tecator", "fat", "reg"), ("cookie", "fat", "reg"), ("wheat", "protein", "reg"), ("cassava", "TBC", "reg"),
         ("sorghum", "CP", "reg"), ("forages", "DM", "reg"), ("pears", "Brix", "reg"), ("ch2018", "protein", "reg"),
         ("ch2021", "y", "reg"), ("mango", "DM", "reg"), ("grapes", "variety", "cls"), ("mayonnaise", "oil", "cls")]


def run(variant):
    fp = OUT / f"{variant}.json"
    if fp.exists():
        return json.loads(fp.read_text())
    m = load_twin("vnir", variant)
    t0 = time.time(); res = dict(variant=variant, e1={}, recon={}, e2={})
    cache = {}
    for name, target, kind in TASKS:
        d = load(name)
        y = d.targets[target]; y = y.astype(float) if kind == "reg" else y.astype(str)
        ok = np.isfinite(y) if kind == "reg" else np.ones(len(y), bool)
        tr = np.where(ok & (d.split == "train"))[0]; te = np.where(ok & (d.split == "test"))[0]
        if name not in cache:
            Xg, valid = to_global(d.X, d.axis, d.axis_unit, "vnir")
            sc = scale_of(Xg[np.where(d.split == "train")[0]], valid)
            cache[name] = (encode_np(m, Xg, valid, "vnir", d.signal, ds_ids()[name], sc), valid)
        lat, valid = cache[name]
        C = corrected_native(d, lat, valid, np.arange(len(d.X))) if m.cfg.use_physics else lat["h"]
        p, _ = m_pls(C[tr], y[tr], d.groups[tr], C[te], kind)
        mt = reg_metrics(y[te], p) if kind == "reg" else cls_metrics(y[te], p)
        res["e1"][f"{name}:{target}"] = mt["RMSEP"] if kind == "reg" else mt["accuracy"]
        res["recon"][name] = float(np.mean(lat["recon_err"][te]))
    for name, target in [("mango", "DM"), ("sorghum", "CP"), ("pears", "Brix")]:
        d = load(name); y = d.targets[target].astype(float); lat, valid = cache[name]
        pool = np.where(d.split == "train")[0]; te = np.where(d.split == "test")[0]
        C = corrected_native(d, lat, valid, np.arange(len(d.X))) if m.cfg.use_physics else lat["h"]; r = []
        for draw in range(5):
            tr = np.random.default_rng(1000 * 20 + draw).choice(pool, 20, replace=False)
            p, _ = m_pls(C[tr], y[tr], d.groups[tr], C[te], "reg", seed=draw)
            r.append(reg_metrics(y[te], p)["RMSEP"])
        res["e2"][f"{name}:{target}"] = float(np.mean(r))
    res["seconds"] = time.time() - t0
    fp.write_text(json.dumps(res, indent=1))
    print(variant, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["e1"].items()}), flush=True)
    return res


if __name__ == "__main__":
    for v in sys.argv[1].split(","):
        run(v)
