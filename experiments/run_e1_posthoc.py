"""POST-HOC (exploratory, not pre-registered): SpecTwin-PLS+ = PLS whose preprocessing candidate set is
extended with the twin nuisance correction (7 standard options on raw spectra + the same 7 on
twin-corrected spectra), chosen by the same inner CV. Uses the cached twin latents from run_e1_twin.py.
"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
torch.set_num_threads(1)
from spectwin.tasks import E1_TASKS, task_data, outer_folds
from spectwin.baselines import pls_tune
from spectwin.chemo import PLS, reg_metrics, cls_metrics
from spectwin.prep import Prep
from spectwin.twin_run import latents_for, corrected_native

OUT = ROOT / "results" / "e1"


def fname(task, method):
    return OUT / (task.name.replace(":", "__").replace("[", "_").replace("]", "").replace("=", "-").replace(",", "_") + f"__{method}.json")


def fit_predict(Xtr, ytr, Xte, hp, kind):
    P = Prep(hp["prep"]).fit(Xtr)
    if kind == "cls":
        classes = np.unique(ytr); Y = (ytr[:, None] == classes[None, :]).astype(float)
        m = PLS(hp["k"]).fit(P.transform(Xtr), Y)
        return classes[m.predict(P.transform(Xte), hp["k"]).argmax(1)]
    m = PLS(hp["k"]).fit(P.transform(Xtr), ytr)
    return m.predict(P.transform(Xte), hp["k"])


def cv_score(X, y, g, hp, kind, seed):
    from spectwin.chemo import group_kfold
    err = 0.0
    for tr, va in group_kfold(g, 5, seed):
        p = fit_predict(X[tr], y[tr], X[va], hp, kind)
        err += (p != y[va]).sum() if kind == "cls" else ((p - y[va]) ** 2).sum()
    return err


for task in E1_TASKS:
    fp = fname(task, "twin_plsplus")
    if fp.exists():
        continue
    ds, idx = task_data(task); folds = outer_folds(task, ds, idx)
    y = ds.targets[task.target]; y = y.astype(float) if task.kind == "reg" else y.astype(str); g = ds.groups
    fo = []; t0 = time.time()
    for f, (tr, te) in enumerate(folds):
        sub = "_".join(f"{k}-{v}" for k, v in (task.subset or {}).items())
        key = f"{sub}fold{f}of{len(folds)}_" + hashlib.md5(np.sort(tr).tobytes()).hexdigest()[:10]
        rows = np.concatenate([tr, te])
        lat, valid = latents_for(task.dataset, tr, rows, "real", adapt_steps=100, cache_key=key, seed=f)
        pos = {r: i for i, r in enumerate(lat["rows"])}
        itr = np.array([pos[r] for r in tr]); ite = np.array([pos[r] for r in te])
        C = corrected_native(ds, lat, valid, lat["rows"])
        Xr = ds.X[rows].astype(float)
        cands = []
        for name, X in [("raw", Xr), ("twin", C)]:
            hp = pls_tune(X[itr], y[tr], g[tr], task.kind, seed=f)
            cands.append((cv_score(X[itr], y[tr], g[tr], hp, task.kind, seed=f + 7), name, hp, X))
        best = min(cands, key=lambda c: c[0])
        p = fit_predict(best[3][itr], y[tr], best[3][ite], best[2], task.kind)
        fo.append(dict(fold=f, test_idx=te.tolist(), y=y[te].tolist(), pred=np.asarray(p).tolist(), hp=dict(best[2], input=best[1])))
    n_rep = 2 if len(fo) == 10 else 1
    per = []
    for r in range(n_rep):
        ff = [x for x in fo if (x["fold"] // 5 if n_rep == 2 else 0) == r]
        yy = np.concatenate([x["y"] for x in ff]); pp = np.concatenate([x["pred"] for x in ff])
        per.append(reg_metrics(yy.astype(float), pp.astype(float)) if task.kind == "reg" else cls_metrics(yy, pp))
    met = {k: float(np.mean([m[k] for m in per])) for k in per[0]}
    fp.write_text(json.dumps(dict(task=task.name, method="twin_plsplus", kind=task.kind, metrics=met, folds=fo, posthoc=True)))
    mk = "RMSEP" if task.kind == "reg" else "accuracy"
    print(f"{task.name:32s} SpecTwin-PLS+ {mk}={met[mk]:.4g}  inputs chosen={[x['hp']['input'] for x in fo]} ({time.time()-t0:.0f}s)", flush=True)
