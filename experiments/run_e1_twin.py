"""E1: SpecTwin variants on all in-domain tasks. Resumable.

twin_pls : tuned PLS (same preprocessing search) on twin-canonicalised spectra
twin_lin : ridge on the physical latents (abundances, scatter, baseline, instrument terms)
twin_mlp : small MLP head on frozen encoder embedding + physical latents (3 seeds; 1 if n_train >= 2000)
usage: run_e1_twin.py [variant] [datasets]
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
torch.set_num_threads(1)
from spectwin.tasks import E1_TASKS, task_data, outer_folds
from spectwin import baselines as B
from spectwin.chemo import reg_metrics, cls_metrics
from spectwin.twin_run import latents_for, phys_features, ridge_feat, mlp_head

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "real"
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None
OUT = Path(__file__).resolve().parents[1] / "results" / "e1"
SUFFIX = "" if VARIANT == "real" else f"_{VARIANT}"


def fname(task, method):
    return OUT / (task.name.replace(":", "__").replace("[", "_").replace("]", "").replace("=", "-").replace(",", "_") + f"__{method}{SUFFIX}.json")


def metrics_of(task, fo):
    n_rep = 2 if len(fo) == 10 else 1
    per = []
    for r in range(n_rep):
        ff = [f for f in fo if (f["fold"] // 5 if n_rep == 2 else 0) == r]
        y = np.concatenate([f["y"] for f in ff]); p = np.concatenate([f["pred"] for f in ff])
        per.append(reg_metrics(y.astype(float), p.astype(float)) if task.kind == "reg" else cls_metrics(y, p))
    return {k: float(np.mean([m[k] for m in per])) for k in per[0]}


for task in E1_TASKS:
    if ONLY and task.dataset not in ONLY:
        continue
    methods = ["twin_pls", "twin_lin", "twin_mlp"]
    if all(fname(task, m).exists() for m in methods):
        continue
    ds, idx = task_data(task); folds = outer_folds(task, ds, idx)
    y = ds.targets[task.target]; y = y.astype(float) if task.kind == "reg" else y.astype(str); g = ds.groups
    t0 = time.time(); per_method = {m: [] for m in methods}; seeds_mlp = None
    try:
        for f, (tr, te) in enumerate(folds):
            # latents are target-independent: cache per (dataset, fold-definition)
            import hashlib
            sub = "_".join(f"{k}-{v}" for k, v in (task.subset or {}).items())
            key = f"{sub}fold{f}of{len(folds)}_" + hashlib.md5(np.sort(tr).tobytes()).hexdigest()[:10]
            rows = np.concatenate([tr, te])
            lat, valid = latents_for(task.dataset, tr, rows, VARIANT, adapt_steps=100, cache_key=key, seed=f)
            pos = {r: i for i, r in enumerate(lat["rows"])}
            itr = np.array([pos[r] for r in tr]); ite = np.array([pos[r] for r in te])
            from spectwin.twin_run import corrected_native
            C = corrected_native(ds, lat, valid, lat["rows"])
            F = phys_features(lat)
            H = np.column_stack([lat["h"], F])
            p1, hp1 = B.m_pls(C[itr], y[tr], g[tr], C[ite], task.kind, seed=f)
            p2, hp2 = ridge_feat(F[itr], y[tr], g[tr], F[ite], task.kind, seed=f)
            per_method["twin_pls"].append(dict(fold=f, test_idx=te.tolist(), y=y[te].tolist(), pred=np.asarray(p1).tolist(), hp=hp1))
            per_method["twin_lin"].append(dict(fold=f, test_idx=te.tolist(), y=y[te].tolist(), pred=np.asarray(p2).tolist(), hp=hp2))
            n_seeds = 3 if len(tr) < 2000 else 1
            preds = []
            for s in range(n_seeds):
                p3, _ = mlp_head(H[itr], y[tr], g[tr], H[ite], task.kind, seed=1000 * s + f)
                preds.append(np.asarray(p3))
            per_method["twin_mlp"].append(dict(fold=f, test_idx=te.tolist(), y=y[te].tolist(), preds=[p.tolist() for p in preds]))
        for m in ["twin_pls", "twin_lin"]:
            fo = per_method[m]
            res = dict(task=task.name, method=m + SUFFIX, kind=task.kind,
                       metrics=metrics_of(task, [dict(x, y=np.array(x["y"]), pred=np.array(x["pred"])) for x in fo]), folds=fo, seconds=time.time() - t0)
            fname(task, m).write_text(json.dumps(res))
        # MLP: per-seed metrics, aggregated like the deep baselines
        fo = per_method["twin_mlp"]; ns = len(fo[0]["preds"]); seeds_out = []
        for s in range(ns):
            ff = [dict(fold=x["fold"], y=np.array(x["y"]), pred=np.array(x["preds"][s])) for x in fo]
            seeds_out.append(dict(seed=s, metrics=metrics_of(task, ff),
                                  folds=[dict(fold=x["fold"], test_idx=x["test_idx"], y=x["y"], pred=x["preds"][s]) for x in fo]))
        keys = seeds_out[0]["metrics"].keys()
        agg = {k: float(np.mean([so["metrics"][k] for so in seeds_out])) for k in keys}
        agg.update({k + "_sd": float(np.std([so["metrics"][k] for so in seeds_out])) for k in keys})
        fname(task, "twin_mlp").write_text(json.dumps(dict(task=task.name, method="twin_mlp" + SUFFIX, kind=task.kind, metrics=agg, seeds=seeds_out, seconds=time.time() - t0)))
        mk = "RMSEP" if task.kind == "reg" else "accuracy"
        r = {m: json.loads(fname(task, m).read_text())["metrics"][mk] for m in methods}
        print(f"{task.name:32s} " + "  ".join(f"{m}={v:.4g}" for m, v in r.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"FAILED {task.name}: {e!r}", flush=True)
