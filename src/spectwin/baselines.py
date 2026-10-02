"""Baseline methods with fixed inner-CV tuning budgets (PROTOCOL.md)."""
from __future__ import annotations

import warnings

import numpy as np

from .chemo import PLS, RidgePath, group_kfold, lwpls_predict
from .prep import Prep

warnings.filterwarnings("ignore")

PREPS = Prep.NAMES
ALPHAS = np.logspace(-6, 3, 10)


def _onehot(y):
    classes = np.unique(y)
    Y = (y[:, None] == classes[None, :]).astype(float)
    return classes, Y


def _subset(n, cap, seed):
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))


# ----------------------------------------------------------------------------- PLS / PLS-DA
def pls_tune(Xtr, ytr, gtr, kind, seed=0, preps=PREPS, Amax=50):
    folds = group_kfold(gtr, 5, seed)
    if kind == "cls":
        classes, Y = _onehot(ytr)
        yi = np.searchsorted(classes, ytr)
    else:
        Y = ytr.astype(float)
    best = (np.inf, None, None)
    p = Xtr.shape[1]
    A_eff = int(min(Amax, p, min(len(tr) for tr, _ in folds) - 1))
    for pp in preps:
        score = np.zeros(A_eff)
        for tr, va in folds:
            P = Prep(pp).fit(Xtr[tr])
            m = PLS(A_eff).fit(P.transform(Xtr[tr]), Y[tr])
            allp = m.predict_all(P.transform(Xtr[va]))      # A x nva x m
            A = allp.shape[0]
            if kind == "cls":
                err = (allp.argmax(2) != yi[va][None, :]).sum(1).astype(float)
            else:
                err = ((allp[:, :, 0] - Y[va][None, :]) ** 2).sum(1)
            if A < A_eff:
                err = np.concatenate([err, np.full(A_eff - A, np.inf)])
            score += err
        k = int(np.argmin(score))
        if score[k] < best[0] - 1e-12:
            best = (score[k], pp, k + 1)
    return dict(prep=best[1], k=best[2])


def m_pls(Xtr, ytr, gtr, Xte, kind, seed=0, hp=None):
    hp = hp or pls_tune(Xtr, ytr, gtr, kind, seed)
    P = Prep(hp["prep"]).fit(Xtr)
    if kind == "cls":
        classes, Y = _onehot(ytr)
        m = PLS(hp["k"]).fit(P.transform(Xtr), Y)
        return classes[m.predict(P.transform(Xte), hp["k"]).argmax(1)], hp
    m = PLS(hp["k"]).fit(P.transform(Xtr), ytr.astype(float))
    return m.predict(P.transform(Xte), hp["k"]), hp


# ----------------------------------------------------------------------------- Ridge
def m_ridge(Xtr, ytr, gtr, Xte, kind, seed=0):
    folds = group_kfold(gtr, 5, seed)
    if kind == "cls":
        classes, Y = _onehot(ytr)
        yi = np.searchsorted(classes, ytr)
    else:
        Y = ytr.astype(float)
    best = (np.inf, None, None)
    for pp in PREPS:
        score = np.zeros(len(ALPHAS))
        for tr, va in folds:
            P = Prep(pp).fit(Xtr[tr])
            r = RidgePath().fit(P.transform(Xtr[tr]), Y[tr])
            Xv = P.transform(Xtr[va])
            for j, a in enumerate(ALPHAS):
                pr = r.predict(Xv, a)
                score[j] += (pr.argmax(1) != yi[va]).sum() if kind == "cls" else ((pr - Y[va]) ** 2).sum()
        j = int(np.argmin(score))
        if score[j] < best[0] - 1e-12:
            best = (score[j], pp, ALPHAS[j])
    hp = dict(prep=best[1], alpha=float(best[2]))
    P = Prep(hp["prep"]).fit(Xtr)
    r = RidgePath().fit(P.transform(Xtr), Y)
    pr = r.predict(P.transform(Xte), hp["alpha"])
    return (classes[pr.argmax(1)] if kind == "cls" else pr), hp


# ----------------------------------------------------------------------------- PCA helper
class _PCA:
    def __init__(self, var=0.999, cap=50):
        self.var, self.cap = var, cap

    def fit(self, X):
        self.m = X.mean(0)
        U, s, Vt = np.linalg.svd(X - self.m, full_matrices=False)
        ev = s ** 2 / max((s ** 2).sum(), 1e-30)
        k = int(np.searchsorted(np.cumsum(ev), self.var) + 1)
        self.k = max(2, min(k, self.cap, len(s)))
        self.V = Vt[: self.k].T
        sc = (X - self.m) @ self.V
        self.sd = sc.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        return self

    def transform(self, X):
        return ((X - self.m) @ self.V) / self.sd


def _feat(Xtr, Xte, prep, var=0.999, cap=50):
    P = Prep(prep).fit(Xtr)
    A, B = P.transform(Xtr), P.transform(Xte)
    pca = _PCA(var, cap).fit(A)
    return pca.transform(A), pca.transform(B), pca.k


# ----------------------------------------------------------------------------- SVR / SVC
def m_svm(Xtr, ytr, gtr, Xte, kind, seed=0, prep="snv"):
    from sklearn.svm import SVC, SVR
    Ftr, Fte, k = _feat(Xtr, Xte, prep)
    sub = _subset(len(Ftr), 2000, seed)
    folds = group_kfold(gtr[sub], 5, seed)
    if kind == "reg":
        mu, sd = ytr.astype(float).mean(), ytr.astype(float).std() + 1e-12
        yz = (ytr.astype(float) - mu) / sd
    best = (np.inf, None)
    for C in [1, 10, 100, 1000]:
        for g in [0.01, 0.1, 1.0]:
            err = 0.0
            for tr, va in folds:
                a, b = sub[tr], sub[va]
                if kind == "reg":
                    m = SVR(C=C, gamma=g / k, epsilon=0.05).fit(Ftr[a], yz[a])
                    err += ((m.predict(Ftr[b]) - yz[b]) ** 2).sum()
                else:
                    m = SVC(C=C, gamma=g / k).fit(Ftr[a], ytr[a])
                    err += (m.predict(Ftr[b]) != ytr[b]).sum()
            if err < best[0]:
                best = (err, (C, g))
    C, g = best[1]
    if kind == "reg":
        m = SVR(C=C, gamma=g / k, epsilon=0.05).fit(Ftr, yz)
        return m.predict(Fte) * sd + mu, dict(prep=prep, C=C, gamma=g, n_pc=k)
    m = SVC(C=C, gamma=g / k).fit(Ftr, ytr)
    return m.predict(Fte), dict(prep=prep, C=C, gamma=g, n_pc=k)


# ----------------------------------------------------------------------------- GPR
def m_gpr(Xtr, ytr, gtr, Xte, kind, seed=0, prep="snv"):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    Ftr, Fte, k = _feat(Xtr, Xte, prep, cap=30)
    sub = _subset(len(Ftr), 2000, seed)
    kern = ConstantKernel(1.0) * RBF(length_scale=np.sqrt(k)) + WhiteKernel(0.1)
    gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=0, random_state=seed)
    gp.fit(Ftr[sub], ytr[sub].astype(float))
    return gp.predict(Fte), dict(prep=prep, n_pc=k, n_fit=int(len(sub)))


# ----------------------------------------------------------------------------- kNN-LWPLS
def m_lwpls(Xtr, ytr, gtr, Xte, kind, seed=0, prep="snv", A_global=15):
    P = Prep(prep).fit(Xtr)
    A, B = P.transform(Xtr), P.transform(Xte)
    y = ytr.astype(float)
    ks_all = [30, 60, 120, 240]
    locals_ = [3, 6, 10]
    sub = _subset(len(A), 1500, seed)
    folds = group_kfold(gtr[sub], 5, seed)
    ks = [k for k in ks_all if k <= int(0.8 * len(sub))] or [max(5, int(0.5 * len(sub)))]
    score = {(k, a): 0.0 for k in ks for a in locals_}
    for tr, va in folds:
        a_idx, b_idx = sub[tr], sub[va]
        pr = lwpls_predict(A[a_idx], y[a_idx], A[b_idx], A_global, ks, locals_)
        for key, v in pr.items():
            score[key] += ((v - y[b_idx]) ** 2).sum()
    k, a = min(score, key=score.get)
    k = min(k, len(A))
    pr = lwpls_predict(A, y, B, A_global, [k], [a])[(k, a)]
    return pr, dict(prep=prep, k=k, a_local=a, A_global=A_global)


# ----------------------------------------------------------------------------- LightGBM
def m_lgbm(Xtr, ytr, gtr, Xte, kind, seed=0, prep="snv"):
    import lightgbm as lgb
    P = Prep(prep).fit(Xtr)
    A, B = P.transform(Xtr), P.transform(Xte)
    folds = group_kfold(gtr, 10, seed)
    tr, va = folds[0]
    if kind == "cls":
        classes = np.unique(ytr)
        yi = np.searchsorted(classes, ytr)
        params = dict(objective="multiclass", num_class=len(classes), metric="multi_logloss")
        ytrain = yi
    else:
        params = dict(objective="regression", metric="l2")
        ytrain = ytr.astype(float)
    base = dict(learning_rate=0.03, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
                verbose=-1, seed=seed, num_threads=1)
    best = (np.inf, None, None)
    for nl in [15, 63]:
        for mcs in [5, 20]:
            prm = dict(base, **params, num_leaves=nl, min_child_samples=mcs)
            dtr = lgb.Dataset(A[tr], ytrain[tr])
            dva = lgb.Dataset(A[va], ytrain[va])
            bst = lgb.train(prm, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
            sc = bst.best_score["valid_0"][params["metric"]]
            if sc < best[0]:
                best = (sc, prm, bst.best_iteration)
    prm, it = best[1], max(50, int(best[2] * 1.1))
    bst = lgb.train(prm, lgb.Dataset(A, ytrain), it)
    pr = bst.predict(B)
    hp = dict(prep=prep, num_leaves=prm["num_leaves"], min_child_samples=prm["min_child_samples"], n_iter=it)
    return (classes[pr.argmax(1)] if kind == "cls" else pr), hp


# ----------------------------------------------------------------------------- kNN
def m_knn(Xtr, ytr, gtr, Xte, kind, seed=0, prep="snv"):
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    Ftr, Fte, k = _feat(Xtr, Xte, prep)
    folds = group_kfold(gtr, 5, seed)
    Est = KNeighborsClassifier if kind == "cls" else KNeighborsRegressor
    best = (np.inf, None)
    for kk in [1, 3, 5, 10, 20]:
        err = 0.0
        for tr, va in folds:
            if kk > len(tr):
                err = np.inf
                break
            m = Est(n_neighbors=kk, weights="distance").fit(Ftr[tr], ytr[tr])
            p = m.predict(Ftr[va])
            err += (p != ytr[va]).sum() if kind == "cls" else ((p - ytr[va].astype(float)) ** 2).sum()
        if err < best[0]:
            best = (err, kk)
    m = Est(n_neighbors=best[1], weights="distance").fit(Ftr, ytr)
    return m.predict(Fte), dict(prep=prep, k=best[1], n_pc=k)
