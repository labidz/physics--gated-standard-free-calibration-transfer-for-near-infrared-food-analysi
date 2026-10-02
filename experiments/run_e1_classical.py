"""E1: classical / ML baselines on all in-domain tasks. Resumable; one JSON per (task, method)."""
import json, sys, time, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectwin.tasks import E1_TASKS, task_data, outer_folds
from spectwin import baselines as B
from spectwin.chemo import reg_metrics, cls_metrics

OUT = Path(__file__).resolve().parents[1] / "results" / "e1"
OUT.mkdir(parents=True, exist_ok=True)
METHODS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["pls", "ridge", "svm", "gpr", "lwpls", "lgbm", "knn"]
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None

def fname(task, method):
    return OUT / (task.name.replace(":", "__").replace("[", "_").replace("]", "").replace("=", "-").replace(",", "_") + f"__{method}.json")

def summarize(task, folds_out):
    n_rep = 2 if len(folds_out) == 10 else 1
    per = []
    for r in range(n_rep):
        fo = [f for f in folds_out if (f["fold"] // 5 if n_rep == 2 else 0) == r]
        y = np.concatenate([f["y"] for f in fo]); p = np.concatenate([f["pred"] for f in fo])
        per.append(reg_metrics(y.astype(float), p.astype(float)) if task.kind == "reg" else cls_metrics(y, p))
    return {k: float(np.mean([m[k] for m in per])) for k in per[0]}

for task in E1_TASKS:
    if ONLY and task.dataset not in ONLY:
        continue
    ds, idx = task_data(task)
    folds = outer_folds(task, ds, idx)
    X = ds.X.astype(float); y = ds.targets[task.target]; g = ds.groups
    y = y.astype(float) if task.kind == "reg" else y.astype(str)
    for method in METHODS:
        if method in ("gpr", "lwpls") and task.kind == "cls":
            continue
        fp = fname(task, method)
        if fp.exists():
            continue
        pls_hp = None
        if method not in ("pls", "ridge"):
            pf = fname(task, "pls")
            if not pf.exists():
                print("need pls first for", task.name, flush=True); continue
            pls_hp = json.loads(pf.read_text())["folds"]
        t0 = time.time(); fo = []
        try:
            for f, (tr, te) in enumerate(folds):
                t1 = time.time()
                if method == "pls":
                    pred, hp = B.m_pls(X[tr], y[tr], g[tr], X[te], task.kind, seed=f)
                elif method == "ridge":
                    pred, hp = B.m_ridge(X[tr], y[tr], g[tr], X[te], task.kind, seed=f)
                else:
                    prep = pls_hp[f]["hp"]["prep"]
                    fn = {"svm": B.m_svm, "gpr": B.m_gpr, "lgbm": B.m_lgbm, "knn": B.m_knn}.get(method)
                    if method == "lwpls":
                        pred, hp = B.m_lwpls(X[tr], y[tr], g[tr], X[te], task.kind, seed=f, prep=prep, A_global=int(pls_hp[f]["hp"]["k"]))
                    else:
                        pred, hp = fn(X[tr], y[tr], g[tr], X[te], task.kind, seed=f, prep=prep)
                fo.append(dict(fold=f, test_idx=te.tolist(), y=y[te].tolist(), pred=np.asarray(pred).tolist(), hp=hp, sec=time.time() - t1))
            res = dict(task=task.name, method=method, kind=task.kind, metrics=summarize(task, [dict(f, y=np.array(f["y"]), pred=np.array(f["pred"])) for f in fo]),
                       folds=fo, seconds=time.time() - t0)
            fp.write_text(json.dumps(res))
            mk = "RMSEP" if task.kind == "reg" else "accuracy"
            print(f"{task.name:32s} {method:6s} {mk}={res['metrics'][mk]:.4g}  ({res['seconds']:.0f}s)", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"FAILED {task.name} {method}: {e!r}", flush=True)
