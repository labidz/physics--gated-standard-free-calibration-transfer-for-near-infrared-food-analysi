"""E7: uncertainty quantification (split conformal variants) and E8: OOD / fraud screening."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
torch.set_num_threads(1)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from spectwin.bench import load
from spectwin.baselines import pls_tune
from spectwin.chemo import PLS, group_kfold
from spectwin.prep import Prep
from spectwin.grid import to_global
from spectwin.twin import adapt, encode_np
from spectwin.twin_run import load_twin, ds_ids, scale_of, phys_features, PRETRAIN, corrected_native

OUT = ROOT / "results" / "e7e8"; OUT.mkdir(parents=True, exist_ok=True)
ALPHAS = [0.2, 0.1, 0.05]


def conformal_q(scores, alpha, weights=None, w_test=1.0):
    s = np.asarray(scores)
    if weights is None:
        n = len(s); k = int(np.ceil((n + 1) * (1 - alpha)))
        return np.inf if k > n else np.sort(s)[k - 1]
    o = np.argsort(s); s, w = s[o], np.asarray(weights)[o]
    p = np.concatenate([w, [w_test]]); p = p / p.sum()
    cum = np.cumsum(p[:-1])
    idx = np.searchsorted(cum, 1 - alpha)
    return np.inf if idx >= len(s) else s[idx]


def interval_metrics(y, lo, hi, alpha, groups=None):
    if len(y) == 0:
        return dict(PICP=None, MPIW_norm=None, interval_score_norm=None, worst_group_PICP=None)
    cover = (y >= lo) & (y <= hi)
    width = hi - lo
    rng = np.ptp(y) if np.ptp(y) > 0 else 1.0
    iscore = width + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    out = dict(PICP=float(cover.mean()), MPIW_norm=float(np.mean(width) / rng), interval_score_norm=float(np.mean(iscore) / rng))
    if groups is not None:
        gc = [cover[groups == g].mean() for g in np.unique(groups) if (groups == g).sum() >= 10]
        out["worst_group_PICP"] = float(min(gc)) if gc else None
    return out


def twin_latents(name, train_rows, rows):
    d = load(name); mod = d.modality
    base = load_twin(mod, "real"); did = ds_ids()[name]
    Xg, valid = to_global(d.X, d.axis, d.axis_unit, mod)
    sc = scale_of(Xg[train_rows], valid)
    m = base if name in PRETRAIN.get(mod, []) else adapt(base, Xg[train_rows], valid, mod, d.signal, did, steps=100)
    lat = encode_np(m, Xg[rows], valid, mod, d.signal, did, sc)
    return lat, valid


def e7():
    tasks = [("mango", "DM", "cultivar"), ("ch2018", "protein", "product"), ("wheat", "protein", None), ("cassava", "TBC", None),
             ("sorghum", "CP", None), ("forages", "DM", "type"), ("pears", "Brix", None), ("tecator", "fat", None),
             ("cookie", "fat", None), ("ch2021", "y", None)]
    fp = OUT / "e7.json"
    res = json.loads(fp.read_text()) if fp.exists() else []
    done = {r["task"] for r in res}
    for name, target, gkey in tasks:
        tname = f"{name}:{target}"
        if tname in done:
            continue
        t0 = time.time()
        d = load(name); y = d.targets[target].astype(float)
        ok = np.isfinite(y)
        tr_all = np.where(ok & (d.split == "train"))[0]; te = np.where(ok & (d.split == "test"))[0]
        folds = group_kfold(d.groups[tr_all], 4, seed=0)
        ptr, cal = tr_all[folds[0][0]], tr_all[folds[0][1]]
        X = d.X.astype(float)
        grp = d.domains[gkey][te] if gkey else None
        out = dict(task=tname, n_cal=int(len(cal)), n_test=int(len(te)), methods={})
        # PLS point model on proper-training set
        hp = pls_tune(X[ptr], y[ptr], d.groups[ptr], "reg")
        P = Prep(hp["prep"]).fit(X[ptr]); m = PLS(hp["k"]).fit(P.transform(X[ptr]), y[ptr])
        pc, pt = m.predict(P.transform(X[cal]), hp["k"]), m.predict(P.transform(X[te]), hp["k"])
        # leverage-based PLS interval (ASTM E1655 style)
        Ttr = m.scores(P.transform(X[ptr]))[:, :hp["k"]]; Tte = m.scores(P.transform(X[te]))[:, :hp["k"]]
        G = np.linalg.pinv(Ttr.T @ Ttr); h = np.einsum("ij,jk,ik->i", Tte, G, Tte) + 1 / len(ptr)
        s = np.sqrt(np.sum((m.predict(P.transform(X[ptr]), hp["k"]) - y[ptr]) ** 2) / max(len(ptr) - hp["k"] - 1, 1))
        from scipy.stats import t as tdist
        for a in ALPHAS:
            half = tdist.ppf(1 - a / 2, max(len(ptr) - hp["k"] - 1, 1)) * s * np.sqrt(1 + h)
            out["methods"].setdefault("PLS leverage interval", {})[str(a)] = interval_metrics(y[te], pt - half, pt + half, a, grp)
            q = conformal_q(np.abs(pc - y[cal]), a)
            out["methods"].setdefault("PLS split conformal", {})[str(a)] = interval_metrics(y[te], pt - q, pt + q, a, grp)
        # LightGBM conformalised quantile regression
        import lightgbm as lgb
        A = P.transform(X[ptr]); Ac = P.transform(X[cal]); At = P.transform(X[te])
        for a in ALPHAS:
            qs = []
            for qq in [a / 2, 1 - a / 2]:
                bst = lgb.train(dict(objective="quantile", alpha=qq, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                                     feature_fraction=0.5, verbose=-1, num_threads=1), lgb.Dataset(A, y[ptr]), 400)
                qs.append((bst.predict(Ac), bst.predict(At)))
            sc_ = np.maximum(qs[0][0] - y[cal], y[cal] - qs[1][0])
            q = conformal_q(sc_, a)
            out["methods"].setdefault("LightGBM CQR", {})[str(a)] = interval_metrics(y[te], qs[0][1] - q, qs[1][1] + q, a, grp)
        # SpecTwin: point model on canonical spectra + physics-residual-scaled conformal (+ weighted under shift)
        rows = np.concatenate([ptr, cal, te])
        lat, valid = twin_latents(name, ptr, rows)
        pos = {r: i for i, r in enumerate(rows)}
        ip, ic, it = [pos[r] for r in ptr], [pos[r] for r in cal], [pos[r] for r in te]
        C = corrected_native(d, lat, valid, rows)
        hp2 = pls_tune(C[ip], y[ptr], d.groups[ptr], "reg")
        P2 = Prep(hp2["prep"]).fit(C[ip]); m2 = PLS(hp2["k"]).fit(P2.transform(C[ip]), y[ptr])
        qc, qt = m2.predict(P2.transform(C[ic]), hp2["k"]), m2.predict(P2.transform(C[it]), hp2["k"])
        r = np.sqrt(lat["recon_err"]); beta = np.median(r[ip])
        sig_c, sig_t = beta + r[ic], beta + r[it]
        F = phys_features(lat)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(np.vstack([F[ic], F[it]]), np.r_[np.zeros(len(ic)), np.ones(len(it))])
        pr = clf.predict_proba(F)[:, 1]
        wr = pr / np.clip(1 - pr, 1e-6, None)
        for a in ALPHAS:
            q = conformal_q(np.abs(qc - y[cal]), a)
            out["methods"].setdefault("SpecTwin split conformal", {})[str(a)] = interval_metrics(y[te], qt - q, qt + q, a, grp)
            q = conformal_q(np.abs(qc - y[cal]) / sig_c, a)
            out["methods"].setdefault("SpecTwin residual-scaled conformal", {})[str(a)] = interval_metrics(y[te], qt - q * sig_t, qt + q * sig_t, a, grp)
            lo, hi = np.empty(len(it)), np.empty(len(it))
            sc_ = np.abs(qc - y[cal]) / sig_c
            for j in range(len(it)):
                q = conformal_q(sc_, a, weights=wr[ic], w_test=wr[it[j]])
                lo[j], hi[j] = qt[j] - q * sig_t[j], qt[j] + q * sig_t[j]
            fin = np.isfinite(lo) & np.isfinite(hi)
            mtr = interval_metrics(y[te][fin], lo[fin], hi[fin], a, grp[fin] if grp is not None else None)
            mtr["frac_infinite"] = float(1 - fin.mean())
            out["methods"].setdefault("SpecTwin weighted residual-scaled conformal", {})[str(a)] = mtr
        out["seconds"] = time.time() - t0
        res.append(out); fp.write_text(json.dumps(res, indent=1))
        print("E7", tname, {k: (round(v["0.1"]["PICP"], 3) if v["0.1"]["PICP"] is not None else None) for k, v in out["methods"].items()}, flush=True)


def pca_scores(Xtr, Xte, var=0.99):
    m = Xtr.mean(0)
    U, s, Vt = np.linalg.svd(Xtr - m, full_matrices=False)
    k = int(np.searchsorted(np.cumsum(s ** 2) / (s ** 2).sum(), var) + 1); k = max(2, min(k, len(s) - 1))
    V = Vt[:k].T; lam = (s[:k] ** 2) / (len(Xtr) - 1)
    def sc(X):
        T = (X - m) @ V
        T2 = ((T ** 2) / lam).sum(1)
        Q = (((X - m) - T @ V.T) ** 2).sum(1)
        return T2, Q, T
    return sc, k


def ddsimca_score(T2tr, Qtr, T2, Q):
    """DD-SIMCA full distance with moment-estimated degrees of freedom (Pomerantsev & Rodionova)."""
    h0, q0 = T2tr.mean(), Qtr.mean()
    Nh = max(1, round(2 * h0 ** 2 / max(T2tr.var(), 1e-12))); Nq = max(1, round(2 * q0 ** 2 / max(Qtr.var(), 1e-12)))
    return Nh * T2 / h0 + Nq * Q / q0


def ood_eval(scores_id_cal, scores_test, is_ood, alpha=0.1):
    auc = roc_auc_score(is_ood, scores_test); ap = average_precision_score(is_ood, scores_test)
    thr = np.percentile(scores_test[~is_ood], 95)
    tpr95 = np.percentile(scores_test[is_ood], 5)
    fpr_at95 = float(np.mean(scores_test[~is_ood] >= tpr95))
    n = len(scores_id_cal)
    p = (1 + (scores_id_cal[None, :] >= scores_test[:, None]).sum(1)) / (n + 1)        # conformal p-values
    o = np.argsort(p); m = len(p)
    passed = np.where(p[o] <= alpha * np.arange(1, m + 1) / m)[0]
    flag = np.zeros(m, bool)
    if len(passed):
        flag[o[: passed.max() + 1]] = True
    fdr = float((flag & ~is_ood).sum() / max(flag.sum(), 1)); power = float((flag & is_ood).sum() / max(is_ood.sum(), 1))
    return dict(AUROC=float(auc), AUPRC=float(ap), FPR_at_95TPR=fpr_at95, BH_FDR=fdr, BH_power=power, n_flagged=int(flag.sum()))


def e8_one(name, X_tr, X_id, X_ood, axis, axis_unit, modality, signal, ds_name=None):
    """Return metrics per scorer. X_tr: in-distribution training spectra; X_id/X_ood: test spectra."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X_tr)); ncal = max(10, int(0.2 * len(X_tr)))
    cal, fit = idx[:ncal], idx[ncal:]
    X_test = np.vstack([X_id, X_ood]); is_ood = np.r_[np.zeros(len(X_id), bool), np.ones(len(X_ood), bool)]
    out = {}
    sc, k = pca_scores(X_tr[fit], X_tr[fit])
    T2f, Qf, _ = sc(X_tr[fit]); T2c, Qc, Tc = sc(X_tr[cal]); T2t, Qt, Tt = sc(X_test)
    norm = lambda T2, Q: T2 / np.percentile(T2f, 95) + Q / np.percentile(Qf, 95)
    out["Hotelling T2 + Q (PCA)"] = ood_eval(norm(T2c, Qc), norm(T2t, Qt), is_ood)
    out["Q residual (PCA)"] = ood_eval(Qc, Qt, is_ood)
    out["DD-SIMCA"] = ood_eval(ddsimca_score(T2f, Qf, T2c, Qc), ddsimca_score(T2f, Qf, T2t, Qt), is_ood)
    from sklearn.neighbors import NearestNeighbors
    _, _, Tf = sc(X_tr[fit]); sdv = Tf.std(0) + 1e-12
    nn_ = NearestNeighbors(n_neighbors=5).fit(Tf / sdv)
    out["kNN distance (PCA)"] = ood_eval(nn_.kneighbors(Tc / sdv)[0].mean(1), nn_.kneighbors(Tt / sdv)[0].mean(1), is_ood)
    # SpecTwin physics residual: pooled twin adapted (unlabelled) to the in-distribution training spectra
    base = load_twin(modality, "real"); did = ds_ids().get(ds_name or name, 90)
    Xg_tr, valid = to_global(X_tr, axis, axis_unit, modality)
    Xg_te, _ = to_global(X_test, axis, axis_unit, modality)
    scl = scale_of(Xg_tr[fit], valid)
    m = adapt(base, Xg_tr[fit], valid, modality, signal, did, steps=150)
    rc = encode_np(m, Xg_tr[cal], valid, modality, signal, did, scl)["recon_err"]
    rt = encode_np(m, Xg_te, valid, modality, signal, did, scl)["recon_err"]
    out["SpecTwin physics residual"] = ood_eval(rc, rt, is_ood)
    return out


def e8():
    fp = OUT / "e8.json"
    res = json.loads(fp.read_text()) if fp.exists() else []
    done = {r["scenario"] for r in res}
    jobs = []
    d = load("oregano"); cls = d.targets["class"]
    tr = np.where(d.split == "train")[0]; te = np.where(d.split == "test")[0]
    jobs.append(("oregano authenticity (non-target + suspicious herbs)", "oregano", d, d.X[tr], d.X[te][cls[te] == "target"], d.X[te][cls[te] != "target"]))
    d = load("mayonnaise"); oil = d.targets["oil"]
    for o in np.unique(oil):
        tr = np.where((d.split == "train") & (oil != o))[0]; ti = np.where((d.split == "test") & (oil != o))[0]; to = np.where(oil == o)[0]
        jobs.append((f"mayonnaise unseen oil {o}", "mayonnaise", d, d.X[tr], d.X[ti], d.X[to]))
    d = load("ch2018"); prod = d.targets["product"]
    for p in np.unique(prod):
        tr = np.where((d.split == "train") & (prod != p))[0]; ti = np.where((d.split == "test") & (prod != p))[0]; to = np.where((d.split == "test") & (prod == p))[0]
        if len(to) >= 10:
            jobs.append((f"ch2018 unseen product {p}", "ch2018", d, d.X[tr], d.X[ti], d.X[to]))
    for scen, nm, d, Xtr, Xid, Xood in jobs:
        if scen in done:
            continue
        t0 = time.time()
        out = e8_one(scen, Xtr.astype(float), Xid.astype(float), Xood.astype(float), d.axis, d.axis_unit, d.modality, d.signal, ds_name=nm)
        res.append(dict(scenario=scen, n_train=int(len(Xtr)), n_id=int(len(Xid)), n_ood=int(len(Xood)), metrics=out, seconds=time.time() - t0))
        fp.write_text(json.dumps(res, indent=1))
        print("E8", scen, {k: round(v["AUROC"], 3) for k, v in out.items()}, flush=True)
    # simulated instrument faults on SimFood-RT (dispersive NIR unit)
    scen = "SimFood-RT instrument faults"
    if scen not in done:
        from spectwin import sim
        z = np.load(ROOT / "data/sim/simfood_vnir_unit03.npz"); units = json.loads((ROOT / "data/sim/units.json").read_text())
        truth = np.load(ROOT / "data/sim/simfood_truth.npz")
        tr = np.where(z["split"] == "train")[0]; te = np.where(z["split"] == "test")[0]
        u = dict(units[3]); faults = []
        for kind in ["shift+3nm", "stray 2%", "linewidth x2", "foreign absorber"]:
            uu = dict(u); A = truth["Afine"][te].astype(float)
            if kind == "shift+3nm": uu["shift"] = u["shift"] + 3.0
            if kind == "stray 2%": uu["stray"] = 0.02
            if kind == "linewidth x2": uu["fwhm_scale"] = u["fwhm_scale"] * 2
            if kind == "foreign absorber":
                A = A + 0.15 * np.exp(-0.5 * ((sim.WL - 1650) / 15) ** 2)[None, :]
            Xf, _ = sim.measure(A, uu, np.random.default_rng(5))
            faults.append(Xf)
        out = e8_one(scen, z["X"][tr].astype(float), z["X"][te].astype(float), np.vstack(faults), z["axis"], "nm", "vnir", "absorbance", ds_name="sim_3")
        res.append(dict(scenario=scen, n_train=int(len(tr)), n_id=int(len(te)), n_ood=int(sum(len(f) for f in faults)), metrics=out))
        fp.write_text(json.dumps(res, indent=1))
        print("E8", scen, {k: round(v["AUROC"], 3) for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    which = sys.argv[1].split(",") if len(sys.argv) > 1 else ["e7", "e8"]
    if "e7" in which: e7()
    if "e8" in which: e8()
