"""Core chemometric algorithms: SIMPLS (all component counts at once), ridge path, kNN-LWPLS, metrics, CV folds."""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------- PLS
def simpls(X, Y, A):
    """SIMPLS (de Jong, 1993). X, Y must be column-centred. Returns weights R (p x A) and Y-loadings Q (m x A).

    The coefficient matrix for k components is R[:, :k] @ Q[:, :k].T.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, p = X.shape
    m = Y.shape[1]
    A = int(max(1, min(A, n - 1, p)))
    S = X.T @ Y
    R = np.zeros((p, A))
    Q = np.zeros((m, A))
    V = np.zeros((p, A))
    for a in range(A):
        if m == 1:
            r = S[:, 0].copy()
        else:
            w, U = np.linalg.eigh(S.T @ S)
            r = S @ U[:, -1]
        t = X @ r
        tn = np.linalg.norm(t)
        if tn < 1e-14:
            R, Q, V = R[:, :a], Q[:, :a], V[:, :a]
            break
        t /= tn
        r /= tn
        pv = X.T @ t
        q = Y.T @ t
        v = pv.copy()
        if a > 0:
            v -= V[:, :a] @ (V[:, :a].T @ pv)
        v /= max(np.linalg.norm(v), 1e-14)
        S = S - np.outer(v, v @ S)
        R[:, a] = r
        Q[:, a] = q
        V[:, a] = v
    return R, Q


class PLS:
    """Linear PLS regression / discriminant with cumulative predictions for 1..A components."""

    def __init__(self, n_components=10):
        self.A = n_components

    def fit(self, X, Y):
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        self.one_d = Y.ndim == 1
        Y = Y[:, None] if self.one_d else Y
        self.xm, self.ym = X.mean(0), Y.mean(0)
        self.R, self.Q = simpls(X - self.xm, Y - self.ym, self.A)
        self.A = self.R.shape[1]
        return self

    def predict_all(self, X):
        """Array (A, n, m): predictions using 1..A components."""
        T = (np.asarray(X, float) - self.xm) @ self.R               # n x A
        contrib = T[:, :, None] * self.Q.T[None, :, :]             # n x A x m
        return np.cumsum(contrib, axis=1).transpose(1, 0, 2) + self.ym

    def predict(self, X, k=None):
        k = self.A if k is None else k
        B = self.R[:, :k] @ self.Q[:, :k].T
        out = (np.asarray(X, float) - self.xm) @ B + self.ym
        return out[:, 0] if self.one_d else out

    def scores(self, X):
        return (np.asarray(X, float) - self.xm) @ self.R

    def coef(self, k):
        return self.R[:, :k] @ self.Q[:, :k].T


# ----------------------------------------------------------------------------- ridge
class RidgePath:
    """Ridge regression for a path of relative penalties (alpha * mean squared singular value)."""

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.one_d = y.ndim == 1
        y = y[:, None] if self.one_d else y
        self.xm, self.ym = X.mean(0), y.mean(0)
        U, s, Vt = np.linalg.svd(X - self.xm, full_matrices=False)
        self.U, self.s, self.Vt = U, s, Vt
        self.Uty = U.T @ (y - self.ym)
        self.scale = (s ** 2).mean()
        return self

    def coef(self, alpha):
        d = self.s / (self.s ** 2 + alpha * self.scale)
        return self.Vt.T @ (d[:, None] * self.Uty)

    def predict(self, X, alpha):
        out = (np.asarray(X, float) - self.xm) @ self.coef(alpha) + self.ym
        return out[:, 0] if self.one_d else out


# ----------------------------------------------------------------------------- kNN-LWPLS
def lwpls_predict(Xtr, ytr, Xte, A_global, ks, locals_, rng=None):
    """kNN-LWPLSR (Lesnoff et al. 2020): neighbours in global-PLS score space, local PLS per query.

    Returns dict {(k, a): predictions}.
    """
    g = PLS(A_global).fit(Xtr, ytr)
    Ttr = g.scores(Xtr)
    Tte = g.scores(Xte)
    sd = Ttr.std(0)
    sd[sd < 1e-12] = 1.0
    Ttr, Tte = Ttr / sd, Tte / sd
    out = {(k, a): np.empty(len(Xte)) for k in ks for a in locals_}
    kmax = max(ks)
    for i in range(len(Xte)):
        d = ((Ttr - Tte[i]) ** 2).sum(1)
        order = np.argpartition(d, min(kmax, len(d) - 1))[:kmax] if kmax < len(d) else np.arange(len(d))
        order = order[np.argsort(d[order])]
        for k in ks:
            nb = order[:k]
            amax = min(max(locals_), len(nb) - 1)
            loc = PLS(amax).fit(Xtr[nb], ytr[nb])
            allp = loc.predict_all(Xte[i:i + 1])[:, 0, 0]
            for a in locals_:
                out[(k, a)][i] = allp[min(a, loc.A) - 1]
    return out


# ----------------------------------------------------------------------------- folds
def group_kfold(groups, k=5, seed=0):
    """Shuffled grouped K-fold: returns list of (train_idx, val_idx)."""
    groups = np.asarray(groups)
    ug = np.unique(groups)
    if len(ug) < min(k, len(groups)):          # too few groups (tiny calibration sets): fall back to samples
        groups = np.arange(len(groups)).astype(str)
        ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(ug)
    k = int(min(k, len(ug)))
    fold_of = {g: i % k for i, g in enumerate(ug)}
    f = np.array([fold_of[g] for g in groups])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


# ----------------------------------------------------------------------------- metrics
def reg_metrics(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    e = yhat - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    bias = float(np.mean(e))
    sep = float(np.std(e, ddof=1)) if len(e) > 1 else float("nan")
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    q75, q25 = np.percentile(y, [75, 25])
    slope = float(np.polyfit(yhat, y, 1)[0]) if np.std(yhat) > 0 else float("nan")
    return dict(RMSEP=rmse, R2=r2, bias=bias, SEP=sep, RPD=float(np.std(y, ddof=1) / rmse) if rmse > 0 else float("nan"),
                RPIQ=float((q75 - q25) / rmse) if rmse > 0 else float("nan"), slope=slope, n=int(len(y)))


def cls_metrics(y, yhat):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
    y = np.asarray(y).astype(str)
    yhat = np.asarray(yhat).astype(str)
    return dict(accuracy=float(accuracy_score(y, yhat)), balanced_accuracy=float(balanced_accuracy_score(y, yhat)),
                macro_f1=float(f1_score(y, yhat, average="macro")), MCC=float(matthews_corrcoef(y, yhat)), n=int(len(y)))
