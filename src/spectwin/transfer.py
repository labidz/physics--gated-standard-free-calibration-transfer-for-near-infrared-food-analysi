"""Calibration transfer: SpecTwin test-time physical adaptation (TTPA) and reference methods.

TTPA (subspace prior). Target spectra are explained as instrument-transformed source chemistry:

    x_t  ~=  S_theta (mu_S + P_S c_t) + L beta_t + d

S_theta : axis shift/warp + Gaussian line-shape broadening (shared by all target spectra)
P_S     : source chemical subspace (top-k principal components of source spectra)
L beta  : per-sample low-order photometric baseline;  d : shared smooth photometric offset curve
theta is estimated from UNLABELLED target spectra only (no reference values, no transfer standards)
by minimising the out-of-subspace residual. Chemistry shifts stay inside span(P_S) and do not bias
theta; instrument effects (derivative-like shift, 2nd-derivative-like broadening) leave the
subspace (transversality condition of Theorem 2). Prediction: source model on mu_S + P_S c_t.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.signal import savgol_filter

from .baselines import m_pls, pls_tune
from .chemo import PLS
from .prep import Prep


# ----------------------------------------------------------------------------- instrument operator (native, uniform grid)
def _uniformize(axis):
    d = np.diff(axis)
    return np.allclose(d, d.mean(), rtol=1e-3, atol=1e-6)


def instrument_op(X, axis, shift, warp, broad):
    """Apply axis shift (axis units) + linear warp (relative) and Gaussian broadening (points)."""
    X = np.atleast_2d(np.asarray(X, float))
    c = 0.5 * (axis[0] + axis[-1])
    half = 0.5 * (axis[-1] - axis[0])
    src = axis - shift - warp * (axis - c) / half
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = np.interp(src, axis, X[i])
    if broad > 1e-3:
        out = gaussian_filter1d(out, broad, axis=1, mode="nearest")
    return out


def _poly_basis(p, deg=2):
    t = np.linspace(-1, 1, p)
    return np.stack([t ** j for j in range(deg + 1)], 1)


class TTPA:
    """Test-time physical adaptation (standard-free, label-free calibration transfer).

    Step 1 - instrument operator theta (axis shift, linear warp, optional relative broadening) from
      the SHAPE of the target covariance: the dominant (signal) principal components of the centred
      target spectra must lie in S_theta(span P_S) + low-order baseline. Centring makes theta
      invariant to additive instrument offsets and to shifts of the chemical mean (no confounding).
    Step 2 - additive photometric response d:
      mode='fleet': instrument changed, population unchanged -> d = mean(target) - mean(S_theta X_S)
      mode='chem' : same instrument, population/process changed -> d = smooth (cubic) part only.
    Prediction - forward simulation: the source calibration is re-fitted (same preprocessing and
      latent variables) on S_theta(X_S) + d, i.e. on source spectra as the target instrument would
      record them, and applied to raw target spectra. No information is lost by projection.
    """

    def __init__(self, mode="fleet", broaden=False, var=0.99999, kmax=40, deg=2):
        self.mode, self.broaden, self.var, self.kmax, self.deg = mode, broaden, var, kmax, deg

    def fit_source(self, Xs, axis, k_min=2):
        self.axis = np.asarray(axis, float)
        self.Xs = np.asarray(Xs, float)
        self.mu = self.Xs.mean(0)
        U, s, Vt = np.linalg.svd(self.Xs - self.mu, full_matrices=False)
        ev = s ** 2 / (s ** 2).sum()
        k = int(np.searchsorted(np.cumsum(ev), self.var) + 1)
        self.k = int(max(k_min, 2, min(max(k, k_min), self.kmax, len(s) - 1)))
        self.P = Vt[: self.k].T
        self.L = _poly_basis(len(axis), self.deg)
        return self

    def _theta(self, th):
        return (th[0], th[1], th[2] if self.broaden else 0.0)

    def _tx_source(self, V, th):
        shift, warp, beta = self._theta(th)
        return instrument_op(V, self.axis, shift, warp, max(beta, 0.0))

    def _tx_target(self, X, th):
        beta = self._theta(th)[2]
        X = np.asarray(X, float)
        return gaussian_filter1d(X, -beta, axis=1, mode="nearest") if beta < -1e-3 else X

    def objective(self, th, Ua):
        SP = self._tx_source(self.P.T, th).T
        Q, _ = np.linalg.qr(np.column_stack([SP, self.L]))
        T = self._tx_target(Ua, th)
        R = T - (T @ Q) @ Q.T
        return float((R ** 2).sum() / max((T ** 2).sum(), 1e-30))

    def fit_target(self, Xa):
        Xa = np.asarray(Xa, float)
        Xc = Xa - Xa.mean(0)
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        kt = int(min(self.k, len(s)))
        Ua = (s[:kt, None] * Vt[:kt])            # target signal subspace, variance-weighted (denoised)
        step = float(np.mean(np.diff(self.axis)))
        if self.broaden:
            starts = [np.array([sh * step, 0.0, b]) for sh in (-2, 0, 2) for b in (-1.0, 0.0, 1.0)]
        else:
            starts = [np.array([sh * step, 0.0, 0.0]) for sh in (-3, -1, 0, 1, 3)]
        res = [minimize(self.objective, x0, args=(Ua,), method="Nelder-Mead",
                        options=dict(xatol=1e-4, fatol=1e-10, maxiter=400)) for x0 in starts]
        best = min(res, key=lambda r: r.fun)
        self.theta, self.obj, self.obj0 = best.x, best.fun, self.objective(np.zeros(3), Ua)
        diff = self._tx_target(Xa, self.theta).mean(0) - self._tx_source(self.mu[None, :], self.theta)[0]
        if self.mode == "fleet":
            self.d = diff
        else:
            t = np.linspace(-1, 1, len(diff))
            self.d = np.polyval(np.polyfit(t, diff, 3), t)
        return self

    def source_as_target(self, Xs=None):
        Xs = self.Xs if Xs is None else Xs
        return self._tx_source(Xs, self.theta) + self.d

    def target_input(self, X):
        return self._tx_target(X, self.theta)

    def residual_norm(self, X):
        SP = self._tx_source(self.P.T, self.theta).T
        m = self._tx_source(self.mu[None, :], self.theta)[0] + self.d
        Q, _ = np.linalg.qr(SP)
        Xp = self._tx_target(X, self.theta) - m
        R = Xp - (Xp @ Q) @ Q.T
        return np.sqrt((R ** 2).mean(1))


# ----------------------------------------------------------------------------- reference methods
def ds_transform(Xs_std, Xt_std):
    """Direct standardisation F (target -> source) from paired standards (truncated pseudo-inverse)."""
    ms, mt = Xs_std.mean(0), Xt_std.mean(0)
    F = np.linalg.pinv(Xt_std - mt, rcond=1e-3) @ (Xs_std - ms)
    return lambda X: (X - mt) @ F + ms


def pds_transform(Xs_std, Xt_std, window=None):
    """Piecewise direct standardisation (Wang et al. 1991); window chosen by leave-one-standard-out error."""
    n, p = Xs_std.shape
    ms, mt = Xs_std.mean(0), Xt_std.mean(0)
    A, Bm = Xt_std - mt, Xs_std - ms

    def build(w):
        Fs = []
        for j in range(p):
            lo, hi = max(0, j - w), min(p, j + w + 1)
            b = np.linalg.pinv(A[:, lo:hi], rcond=1e-2) @ Bm[:, j]
            Fs.append((lo, hi, b))
        return Fs

    def apply(Fs, X):
        Xc = X - mt
        out = np.empty((len(X), p))
        for j, (lo, hi, b) in enumerate(Fs):
            out[:, j] = Xc[:, lo:hi] @ b
        return out + ms

    if window is None:
        best = (np.inf, 2)
        for w in [1, 2, 3, 5, 7]:
            err = 0.0
            for i in range(min(n, 12)):   # LOO on (up to) 12 standards
                keep = np.arange(n) != i
                ms_, mt_ = Xs_std[keep].mean(0), Xt_std[keep].mean(0)
                A_, B_ = Xt_std[keep] - mt_, Xs_std[keep] - ms_
                xi = Xt_std[i] - mt_
                rec = np.empty(p)
                for j in range(p):
                    lo, hi = max(0, j - w), min(p, j + w + 1)
                    rec[j] = xi[lo:hi] @ (np.linalg.pinv(A_[:, lo:hi], rcond=1e-2) @ B_[:, j])
                err += ((rec + ms_ - Xs_std[i]) ** 2).sum()
            if err < best[0]:
                best = (err, w)
        window = best[1]
    Fs = build(window)
    f = lambda X: apply(Fs, X)
    f.window = window
    return f


def coral_transform(Xs, Xa, k=20, eps=1e-3):
    """CORAL (Sun et al. 2016) in a shared PCA space: whiten target covariance, recolour with source."""
    Z = np.vstack([Xs, Xa])
    m = Z.mean(0)
    U, s, Vt = np.linalg.svd(Z - m, full_matrices=False)
    k = int(min(k, len(s) - 1))
    V = Vt[:k].T
    Ts, Ta = (Xs - m) @ V, (Xa - m) @ V
    Cs = np.cov(Ts, rowvar=False) + eps * np.eye(k) * np.trace(np.cov(Ts, rowvar=False)) / k
    Ca = np.cov(Ta, rowvar=False) + eps * np.eye(k) * np.trace(np.cov(Ta, rowvar=False)) / k

    def msqrt(C, inv=False):
        w, Q = np.linalg.eigh(C)
        w = np.clip(w, 1e-12, None)
        return (Q * (w ** (-0.5 if inv else 0.5))) @ Q.T

    Mt = msqrt(Ca, True) @ msqrt(Cs)
    ms_, ma_ = Ts.mean(0), Ta.mean(0)

    def f(X):
        T = (X - m) @ V
        T2 = (T - ma_) @ Mt + ms_
        resid = (X - m) - T @ V.T
        return m + T2 @ V.T + resid
    return f


def dipls_predict(Xs, ys, Xa, Xt, A):
    from diPLSlib.models import DIPLS
    mdl = DIPLS(A=int(A), l=0, heuristic=True, rescale="Target")
    mdl.fit(Xs, ys, Xs, Xa)
    return np.asarray(mdl.predict(Xt)).ravel()
