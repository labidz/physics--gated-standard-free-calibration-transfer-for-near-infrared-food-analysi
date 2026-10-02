"""E1: deep baselines (CNN1D, patch Transformer), 3 seeds. Resumable."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectwin.tasks import E1_TASKS, task_data, outer_folds
from spectwin.deep import m_deep, set_threads
from spectwin.chemo import reg_metrics, cls_metrics
set_threads(1)
OUT = Path(__file__).resolve().parents[1] / "results" / "e1"
ARCHS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["cnn", "transformer"]
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None
SEEDS = [0, 1, 2]

def fname(task, method):
    return OUT / (task.name.replace(":", "__").replace("[", "_").replace("]", "").replace("=", "-").replace(",", "_") + f"__{method}.json")

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
    ds, idx = task_data(task); folds = outer_folds(task, ds, idx)
    X = ds.X.astype(float); y = ds.targets[task.target]; g = ds.groups
    y = y.astype(float) if task.kind == "reg" else y.astype(str)
    for arch in ARCHS:
        fp = fname(task, arch)
        if fp.exists():
            continue
        t0 = time.time(); seeds_out = []
        try:
            n_tr = len(folds[0][0])
            for s in (SEEDS if (n_tr < 2000 and len(folds) == 1) else SEEDS[:1]):
                fo = []
                for f, (tr, te) in enumerate(folds):
                    pred, info = m_deep(arch, X[tr], y[tr], g[tr], X[te], task.kind, seed=1000 * s + f)
                    fo.append(dict(fold=f, test_idx=te.tolist(), y=y[te].tolist(), pred=np.asarray(pred).tolist(), info=info))
                m = metrics_of(task, [dict(f, y=np.array(f["y"]), pred=np.array(f["pred"])) for f in fo])
                seeds_out.append(dict(seed=s, metrics=m, folds=fo))
            keys = seeds_out[0]["metrics"].keys()
            agg = {k: float(np.mean([so["metrics"][k] for so in seeds_out])) for k in keys}
            agg.update({k + "_sd": float(np.std([so["metrics"][k] for so in seeds_out])) for k in keys})
            res = dict(task=task.name, method=arch, kind=task.kind, metrics=agg, seeds=seeds_out, seconds=time.time() - t0)
            fp.write_text(json.dumps(res))
            mk = "RMSEP" if task.kind == "reg" else "accuracy"
            print(f"{task.name:32s} {arch:11s} {mk}={agg[mk]:.4g}±{agg[mk + '_sd']:.2g} ({res['seconds']:.0f}s)", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"FAILED {task.name} {arch}: {e!r}", flush=True)
