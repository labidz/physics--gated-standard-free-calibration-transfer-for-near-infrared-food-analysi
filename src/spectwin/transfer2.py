"""Standard-free calibration transfer, round 2 (E12).

* Prior-art baselines that need only unlabelled target spectra:
    - uDOP  (Fonseca Diaz, Roger & Saeys, Anal. Chim. Acta 2022) via the authors' pycaltransfer code,
    - TCA/TCR (Pan et al. 2011; Malli et al. 2017) via pycaltransfer.linear_tca + OLS,
    - CT-VPdtw (Zou et al., Anal. Methods 2019) via the authors' released R code (rsrc/ctvpdtw_run.R),
    - OT-DA  (exact optimal-transport mapping of target to source in a pooled PCA space; POT),
    - DANN   (Ganin et al. 2016; 1-D CNN with gradient reversal), the deep-learning reference.
* TTPA v4: test-time physical adaptation with two built-in significance gates and a
  physics-consistent split of the mean shift (see docs/THEORY_TTPA.md):
    x_t = S_theta(mu_S + P_S c) + d_par + d_perp + noise,  d_par in span[S_theta P_S, L], d_perp orthogonal.
  - theta (axis shift, linear warp, Gaussian broadening) is estimated from the SHAPE of the target
    covariance (as in v3) and used only if it explains significantly more out-of-subspace target
    structure than it does for same-instrument pseudo-targets drawn from the source (exact Monte-Carlo
    test, 19 draws, alpha = 0.05);
  - d_perp (mean shift that no source chemistry can produce) is corrected only if significant by the
    same Monte-Carlo test;
  - d_par (mean shift inside the chemical/baseline subspace) is NOT identifiable from unlabelled data
    (Proposition 2): it is corrected only when the deployment context declares an unchanged population
    ('fleet': new instrument or measurement condition, same product stream).
  When nothing is significant and the context is 'chem', TTPA v4 returns the source model unchanged.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from .chemo import PLS
from .prep import Prep

RSRC = Path(__file__).resolve().parents[2] / "rsrc"


# ----------------------------------------------------------------------------- fast instrument operator
def _interp_plan(axis, shift, warp):
    c = 0.5 * (axis[0] + axis[-1])
    half = 0.5 * (axis[-1] - axis[0])
    src = axis - shift - warp * (axis - c) / half
    idx = np.clip(np.searchsorted(axis, src, side="right") - 1, 0, len(axis) - 2)
    w = np.clip((src - axis[idx]) / (axis[idx + 1] - axis[idx]), 0.0, 1.0)
    return idx, w


def instrument_op(X, axis, shift, warp, broad):
    """Same operator as transfer.instrument_op (np.interp semantics incl. edge clamping), vectorised."""
    X = np.atleast_2d(np.asarray(X, float))
    idx, w = _interp_plan(np.asarray(axis, float), shift, warp)
    out = X[:, idx] * (1.0 - w) + X[:, idx + 1] * w
    if broad > 1e-3:
        out = gaussian_filter1d(out, broad, axis=1, mode="nearest")
    return out


def _poly_basis(p, deg=2):
    t = np.linspace(-1, 1, p)
    return np.stack([t ** j for j in range(deg + 1)], 1)


class TTPA4:
    def __init__(self, context="fleet", broaden=True, gate=True, use_par=None, var=0.99999, kmax=40, deg=2,
                 n_null=19, seed=0, gate_par=False):
        assert context in ("fleet", "chem")
        self.context, self.broaden, self.gate, self.gate_par = context, broaden, gate, gate_par   # broaden: True/False/'test'
        self.use_par = (context == "fleet") if use_par is None else use_par
        self.var, self.kmax, self.deg, self.n_null, self.seed = var, kmax, deg, n_null, seed

    # -- source model of chemistry
    def _subspace(self, X):
        mu = X.mean(0)
        U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
        return mu, Vt[: min(self.k, len(s))].T

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
        self.step = float(np.mean(np.diff(self.axis)))
        return self

    # -- operator pieces (theta = (shift, warp, sigma); sigma = 0 means no broadening)
    def _src(self, X, th):
        return instrument_op(X, self.axis, float(th[0]), float(th[1]), max(float(th[2]), 0.0))

    def _tgt(self, X, th):
        return np.asarray(X, float)      # broadening is modelled on the source side only (see _box)

    def _basis(self, P, th):
        Q, _ = np.linalg.qr(np.column_stack([self._src(P.T, th).T, self.L]))
        return Q

    def _box(self, th):
        """Physical bounds: |shift|, |warp| <= 10 sampling steps; source-side broadening 0 <= sigma <= 5 points.
        Broadening is applied to the SOURCE basis only (target narrower than source is not modelled):
        smoothing the target would lower the residual ratio by removing noise, a degenerate solution."""
        lo = np.array([-10 * self.step, -10 * self.step, 0.0]); hi = np.array([10 * self.step, 10 * self.step, 5.0])
        th = np.asarray(th, float)
        c = np.clip(th, lo, hi)
        return c, float(((th - c) / np.maximum(hi - lo, 1e-12)) ** 2 @ np.ones(3))

    def _objective(self, th, P, Ua):
        th, pen = self._box(th)
        Q = self._basis(P, th)
        R = Ua - (Ua @ Q) @ Q.T
        return float((R ** 2).sum() / max((Ua ** 2).sum(), 1e-30)) + 1e3 * pen

    @staticmethod
    def _signal(Xa, k):
        Xc = Xa - Xa.mean(0)
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        kt = int(min(k, len(s)))
        return s[:kt, None] * Vt[:kt]

    def _fit(self, P, Ua, quick, broad, x0=None):
        """Nelder-Mead on (shift, warp[, sigma]). Returns (theta, R(theta))."""
        if broad:
            f = lambda t: self._objective(t, P, Ua)
            starts = ([np.asarray(x0, float)] if x0 is not None else [np.zeros(3)]) if quick else \
                [np.array([sh * self.step, 0.0, b]) for sh in (-2, 0, 2) for b in (0.0, 1.0, 2.5)]
        else:
            f = lambda t: self._objective(np.r_[t, 0.0], P, Ua)
            starts = [np.zeros(2)] if quick else [np.array([sh * self.step, 0.0]) for sh in (-3, -1, 0, 1, 3)]
        opts = dict(xatol=1e-3, fatol=1e-9, maxiter=150) if quick else dict(xatol=1e-4, fatol=1e-10, maxiter=400)
        best = min((minimize(f, x, method="Nelder-Mead", options=opts) for x in starts), key=lambda r: r.fun)
        th = np.r_[best.x, 0.0] if not broad else best.x
        return self._box(th)[0], float(best.fun)

    def _split_mean(self, P, mu, Xa, th):
        diff = Xa.mean(0) - self._src(mu[None, :], th)[0]
        Q = self._basis(P, th)
        d_par = Q @ (Q.T @ diff)
        return d_par, diff - d_par

    def _stats(self, Sp, Ap):
        """Test statistics for one (pseudo-)source / (pseudo-)target pair, computed identically for the real
        target and for every null draw (quick fits from theta = 0): R(0), R(shift,warp), R(shift,warp,sigma),
        out-of-span and in-span mean-shift energies."""
        z = np.zeros(3)
        mu_p, P_p = self._subspace(Sp)
        Up = self._signal(Ap, self.k)
        r0p = self._objective(z, P_p, Up)
        tpn, rpn = self._fit(P_p, Up, True, False)
        rpb = self._fit(P_p, Up, True, True, x0=[tpn[0], tpn[1], 1.0])[1] if self.broaden is not False else rpn
        dpa, dpp = self._split_mean(P_p, mu_p, Ap, z)
        return (r0p, rpn, rpb, float(np.mean(dpp ** 2)), float(np.mean(dpa ** 2)))

    def fit_target(self, Xa):
        """Estimation: theta with and without broadening from the full source subspace and the full target.
        Testing (exact Monte-Carlo, Proposition 3): the real target statistic and n_null null statistics are all
        computed from a pseudo-source of the same size m = n_S - n_a and a target set of the same size n_a, with
        the same quick fitting procedure; under exchangeability of target and source spectra the real statistic
        is exchangeable with the null ones, so 'adapt iff real > max(null)' has size <= 1/(n_null + 1)."""
        Xa = np.asarray(Xa, float)
        Ua = self._signal(Xa, self.k)
        z = np.zeros(3)
        r0 = self._objective(z, self.P, Ua)
        th_n, r_n = self._fit(self.P, Ua, False, False)
        th_b, r_b = self._fit(self.P, Ua, False, True) if self.broaden is not False else (th_n, r_n)
        st = dict(R0=r0, Rn=r_n, Rb=r_b, theta_n=th_n.tolist(), theta_b=th_b.tolist())
        null, real = [], None
        if self.gate:
            rng = np.random.default_rng(self.seed)
            ns = len(self.Xs); na = int(min(len(Xa), ns // 2)); m = ns - na
            ref = rng.permutation(ns)[:m]
            real = self._stats(self.Xs[ref], Xa[rng.permutation(len(Xa))[:na]])
            for _ in range(self.n_null):
                perm = rng.permutation(ns)
                null.append(self._stats(self.Xs[perm[na:]], self.Xs[perm[:na]]))
        self._Xa, self._th_n, self._th_b = Xa, th_n, th_b
        self._null = np.array(null, float).reshape(-1, 5)
        self._real = None if real is None else np.array(real, float)
        self.stats = st
        self.configure()
        return self

    def configure(self, gate=None, use_par=None, gate_par=None, broaden=None, gof="auto", chem_theta="test", chem_mean="perp"):
        """Derive the adaptation actually applied from the fitted statistics (ablations share one fit).
        broaden: False (shift+warp), True (always fit broadening), 'test' (broadening only if it significantly
        improves on shift+warp, nested Monte-Carlo test)."""
        gate = self.gate if gate is None else gate
        use_par = self.use_par if use_par is None else use_par
        gate_par = self.gate_par if gate_par is None else gate_par
        broaden = self.broaden if broaden is None else broaden
        if broaden == "auto":
            # 'chem' context = same instrument by definition: no line-shape change is admitted (band-width
            # changes of the chemistry would be confounded with broadening); 'fleet': nested test
            broaden = "test" if self.context == "fleet" else False
        st, N = self.stats, self._null
        # gate statistics come from the exchangeable test computation (self._real); estimation uses the full data
        r0, rn, rb, D_t, P_t = self._real if self._real is not None else (st["R0"], st["Rn"], st["Rb"], np.nan, np.nan)
        rel = lambda a, b: (a - b) / max(a, 1e-30)
        if broaden is True:
            G, Gn = rel(r0, rb), [rel(a, c) for a, _, c, _, _ in N]
            th_hat, broad_ok = self._th_b, True
        else:
            G, Gn = rel(r0, rn), [rel(a, b) for a, b, _, _, _ in N]
            broad_ok = False
            if broaden == "test":
                Gb, Gbn = rel(rn, rb), [rel(b, c) for _, b, c, _, _ in N]
                broad_ok = bool(Gb > max(Gbn)) if gate else True
                st.update(Gb=Gb, Gb_null_max=float(max(Gbn)) if len(Gbn) else None)
            th_hat = self._th_b if broad_ok else self._th_n
        theta_ok = bool(G > max(Gn)) if gate else True
        # goodness-of-fit gate ('chem' context by default): the operator is trusted only if, after it, the target
        # signal subspace is explained as well as a same-instrument pseudo-target is (no unexplained new chemistry)
        gof = (self.context == "chem") if gof == "auto" else bool(gof)
        r_fit = rb if (broaden is True or broad_ok) else rn
        r_null = [c if (broaden is True or broad_ok) else b for _, b, c, _, _ in N]
        gof_ok = bool(r_fit <= max(r_null)) if (gate and gof and len(r_null)) else True
        if self.context == "chem" and chem_theta == "abstain":
            theta_ok = False
        theta_ok = theta_ok and gof_ok
        self.theta = th_hat if theta_ok else np.zeros(3)
        st.update(gof=gof, gof_ok=gof_ok, R_fit=r_fit, R_null_max=float(max(r_null)) if len(r_null) else None)
        d_par, d_perp = self._split_mean(self.P, self.mu, self._Xa, self.theta)
        D = float(np.mean(d_perp ** 2)); Dp = float(np.mean(d_par ** 2))
        d_ok = bool(D_t > N[:, 3].max()) if gate else True
        # in-span mean shift: only in 'fleet' context (Proposition 2) and, if gate_par, only when it exceeds what
        # sampling variation of the mean composition produces on same-instrument pseudo-targets
        par_ok = bool(use_par and ((P_t > N[:, 4].max()) if (gate and gate_par) else True))
        if self.context == "chem" and chem_mean == "none":
            d_ok = par_ok = False          # population may have changed: no mean-shift correction at all
        self.d = (d_perp if d_ok else 0.0) + (d_par if par_ok else 0.0)
        st.update(G=G, G_null_max=float(max(Gn)) if len(Gn) else None, theta_hat=th_hat.tolist(), theta_used=self.theta.tolist(),
                  D_test=float(D_t), Dpar_test=float(P_t), G_test_R=[float(r0), float(rn), float(rb)],
                  D=D, D_null_max=float(N[:, 3].max()) if len(N) else None, Dpar=Dp, Dpar_null_max=float(N[:, 4].max()) if len(N) else None,
                  theta_gate=theta_ok, broad_gate=broad_ok, dperp_gate=d_ok, dpar_used=par_ok, use_par=use_par, broaden=str(broaden),
                  gate=gate, gate_par=gate_par, norm_dpar=float(np.sqrt(Dp)), norm_dperp=float(np.sqrt(D)), k=self.k)
        return self

    def source_as_target(self, Xs=None):
        Xs = self.Xs if Xs is None else Xs
        return self._src(Xs, self.theta) + self.d

    def target_input(self, X):
        return self._tgt(X, self.theta)


# ----------------------------------------------------------------------------- prior-art baselines
def _prep(hp, Xs, *others):
    P = Prep(hp["prep"]).fit(Xs)
    return [P.transform(Xs)] + [P.transform(o) for o in others]


def _pls_fit_predict(Z, y, Zt, k, kind):
    if kind == "cls":
        classes = np.unique(y)
        Y = (y[:, None] == classes[None, :]).astype(float)
        m = PLS(k).fit(Z, Y)
        return classes[m.predict(Zt, k).argmax(1)]
    return PLS(k).fit(Z, y).predict(Zt, k)


def udop_predict(Xs, ys, Xa, Xt, hp, kind, udop_ncomp=1, svd_ncomp=20):
    """uDOP with the authors' defaults (1 component removed, 20 SVD components), on preprocessed spectra."""
    from pycaltransfer.caltransfer import udop
    Zs, Za, Zt = _prep(hp, Xs, Xa, Xt)
    E = udop(Zs, Za, udop_ncomp=udop_ncomp, svd_ncomp=int(min(svd_ncomp, len(Zs) - 1, len(Za) - 1)))
    return _pls_fit_predict(Zs @ E, ys, Zt @ E, hp["k"], kind)


MAX_N_SUPERLINEAR = 2000   # TCA, OT and CT-VPdtw are fitted on at most this many (random) source spectra


def _cap(Xs, ys, n=MAX_N_SUPERLINEAR):
    if len(Xs) <= n:
        return Xs, ys
    i = np.sort(np.random.default_rng(0).choice(len(Xs), n, replace=False))
    return Xs[i], ys[i]


def tcr_predict(Xs, ys, Xa, Xt, hp, kind, mu=1.0):
    """TCR = linear TCA (pycaltransfer) + OLS on the transfer components (regression only)."""
    from pycaltransfer.caltransfer import linear_tca
    if kind != "reg":
        return None
    Xs, ys = _cap(Xs, ys)
    Zs, Za, Zt = _prep(hp, Xs, Xa, Xt)
    a = int(min(hp["k"], 20, len(Zs) + len(Za) - 2))
    R, xs_m, xt_m, _ = linear_tca(Zs, Za, a, mu)
    Ts = (Zs - xs_m) @ R
    A = np.column_stack([Ts, np.ones(len(Ts))])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    return np.column_stack([(Zt - xt_m) @ R, np.ones(len(Zt))]) @ coef


def ot_predict(Xs, ys, Xa, Xt, hp, kind, ncomp=20):
    """Exact OT (EMD) barycentric mapping of target -> source in a pooled PCA space; source model applied."""
    import ot
    Xs, ys = _cap(Xs, ys)
    Zs, Za, Zt = _prep(hp, Xs, Xa, Xt)
    Z = np.vstack([Zs, Za]); m = Z.mean(0)
    U, s, Vt = np.linalg.svd(Z - m, full_matrices=False)
    V = Vt[: int(min(ncomp, len(s) - 1))].T
    Ts, Ta, Tt = (Zs - m) @ V, (Za - m) @ V, (Zt - m) @ V
    mp = ot.da.EMDTransport()
    mp.fit(Xs=Ta, Xt=Ts)
    Tt2 = mp.transform(Xs=Tt)
    Zt2 = Zt + (Tt2 - Tt) @ V.T
    return _pls_fit_predict(Zs, ys, Zt2, hp["k"], kind)


def dann_predict(Xs, ys, Xa, Xt, hp, kind, steps=1000, seed=0):
    """DANN (gradient-reversal domain adversarial training), 1-D CNN feature extractor; regression only."""
    import torch
    import torch.nn as nn
    if kind != "reg":
        return None
    torch.manual_seed(seed); np.random.seed(seed)
    Zs, Za, Zt = _prep(hp, Xs, Xa, Xt)
    mz, sz = np.vstack([Zs, Za]).mean(0), np.vstack([Zs, Za]).std(0) + 1e-8
    f = lambda Z: torch.tensor((Z - mz) / sz, dtype=torch.float32)[:, None, :]
    my, sy = ys.mean(), ys.std() + 1e-12
    xs, xa, xt = f(Zs), f(Za), f(Zt)
    yt = torch.tensor((ys - my) / sy, dtype=torch.float32)

    class GRL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lam):
            ctx.lam = lam
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -ctx.lam * g, None

    feat = nn.Sequential(nn.Conv1d(1, 8, 9, stride=2, padding=4), nn.ReLU(), nn.Conv1d(8, 16, 9, stride=2, padding=4), nn.ReLU(),
                         nn.AdaptiveAvgPool1d(16), nn.Flatten(), nn.Linear(256, 64), nn.ReLU())
    reg = nn.Linear(64, 1)
    dis = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(list(feat.parameters()) + list(reg.parameters()) + list(dis.parameters()), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    bs = int(min(32, len(xs), len(xa)))
    for it in range(steps):
        p = it / steps
        lam = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        i = torch.randint(0, len(xs), (bs,)); j = torch.randint(0, len(xa), (bs,))
        hs, ha = feat(xs[i]), feat(xa[j])
        loss = ((reg(hs)[:, 0] - yt[i]) ** 2).mean()
        dl = dis(GRL.apply(torch.cat([hs, ha]), lam))[:, 0]
        loss = loss + bce(dl, torch.cat([torch.zeros(bs), torch.ones(bs)]))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return reg(feat(xt))[:, 0].numpy() * sy + my


def _wmat(M, path):
    M = np.atleast_2d(np.asarray(M, float))
    with open(path, "wb") as fh:
        np.asarray(M.shape, dtype="<i4").tofile(fh)
        np.ascontiguousarray(M, dtype="<f8").tofile(fh)


def _rmat(path):
    with open(path, "rb") as fh:
        r, c = np.fromfile(fh, dtype="<i4", count=2)
        return np.fromfile(fh, dtype="<f8", count=r * c).reshape(r, c)


def ctvpdtw_batch(jobs, width=21, order=2, span=5, S=1, A0=20, timeout=3600, deriv=0):
    """jobs: list of (Xs, ys, Xa, Xt). Runs the authors' R implementation once for the whole batch.
    Returns list of (pred or None, info dict)."""
    tmp = Path(tempfile.mkdtemp(prefix="ctvp_"))
    dirs = []
    for n, (Xs, ys, Xa, Xt) in enumerate(jobs):
        d = tmp / f"j{n}"; d.mkdir()
        _wmat(Xs, d / "Xs.bin"); _wmat(np.asarray(ys, float)[:, None], d / "ys.bin")
        _wmat(Xa, d / "Xa.bin"); _wmat(Xt, d / "Xt.bin")
        _wmat(np.array([[width, order, span, S, min(A0, len(Xs) - 2), deriv]], float), d / "cfg.bin")
        dirs.append(d)
    (tmp / "jobs.txt").write_text("\n".join(str(d) for d in dirs) + "\n")
    subprocess.run(["Rscript", str(RSRC / "ctvpdtw_run.R"), str(tmp / "jobs.txt")], check=False, timeout=timeout,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = []
    for d in dirs:
        if (d / "pred.bin").exists():
            info = _rmat(d / "info.bin")[0]
            out.append((_rmat(d / "pred.bin")[:, 0], dict(deriv=int(info[0]), lv=int(info[1]), rmsecv=float(info[2]))))
        else:
            err = (d / "error.txt").read_text()[:300] if (d / "error.txt").exists() else "no output"
            out.append((None, dict(error=err)))
    subprocess.run(["rm", "-rf", str(tmp)], check=False)
    return out
