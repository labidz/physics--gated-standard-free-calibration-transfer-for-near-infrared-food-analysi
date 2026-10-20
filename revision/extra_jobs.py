"""Post-hoc analyses for the revision (not part of the frozen protocol).
  python extra_jobs.py recovery | null | dipls | sens
Run from spectwin/experiments with PYTHONPATH=../src. Outputs CSV files to $OUT (default ../../out3)."""
import sys, os, json, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; OUT = Path(os.environ.get("OUT", ROOT.parent / "out3")); OUT.mkdir(exist_ok=True, parents=True)
import run_e12, run_e12_conf
from run_e12 import Family, fit_model
from run_e12_conf import idrc, icrisat, grain_years
from spectwin.baselines import pls_tune
from spectwin.transfer2 import TTPA4, instrument_op
FROZEN = dict(broaden="auto", gate_par=False, gof=True, chem_mean="none", chem_theta="abstain")
rmsep = lambda y, p: float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))

# ------------------------------------------------------------------ #5 parameter recovery
def recovery():
    w, y, _ = idrc(); X = w["A1.CAL"].astype(float); ax = np.arange(X.shape[1], dtype=float); sd = X.std()
    rows = []; t0 = time.time()
    truths = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 1.0, 1.0), (3.0, -2.0, 2.5)]   # shift, warp (channels), sigma (channels)
    for (s, wp, sg) in truths:
        for nA in (20, 60):
            for noise in (1e-3, 1e-2):
                for rep in range(8):
                    rng = np.random.default_rng(1000 + rep); perm = rng.permutation(len(X)); S, A = perm[:124], perm[124:124 + nA]
                    t = np.linspace(-1, 1, X.shape[1]); d = 0.02 * sd * (0.5 + t - t ** 2)
                    Xa = instrument_op(X[A], ax, s, wp, sg) + d + noise * sd * rng.standard_normal((nA, X.shape[1]))
                    tt = TTPA4("fleet", seed=rep, broaden="test", gate=False).fit_source(X[S], ax, k_min=8).fit_target(Xa)
                    tn, tb = tt.stats["theta_n"], tt.stats["theta_b"]
                    rows.append(dict(true_shift=s, true_warp=wp, true_sigma=sg, n_A=nA, noise_rel=noise, rep=rep,
                                     shift_nb=tn[0], warp_nb=tn[1], shift_b=tb[0], warp_b=tb[1], sigma_b=tb[2],
                                     R0=tt.stats["R0"], Rn=tt.stats["Rn"], Rb=tt.stats["Rb"], k=tt.k))
        print("recovery truth", (s, wp, sg), f"{time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "recovery.csv", index=False)

# ------------------------------------------------------------------ #7 null (same-instrument) draws
def null():
    w, y, _ = idrc(); ic = icrisat(); g = grain_years("regression_2")
    sets = [("IDRC A1 (248)", w["A1.CAL"].astype(float), y, np.arange(w["A1.CAL"].shape[1], dtype=float)),
            ("IDRC B1 (248)", w["B1.CAL"].astype(float), y, np.arange(w["B1.CAL"].shape[1], dtype=float)),
            ("ICRISAT FOSS (302)", ic["foss"], ic["y"], ic["wl"]), ("Grain 2022 handheld-1 (n=%d)" % len(g["Xs"]), g["Xs"], g["ys"], g["axis"])]
    rows = []; t0 = time.time()
    for name, X, yy, ax in sets:
        for frac in (1.0, 0.5):
            for draw in range(20):
                seed = 7000 + draw; rng = np.random.default_rng(seed); n = len(X); perm = rng.permutation(n)
                nS, nA = n // 2, n // 4; S, A, T = perm[:nS], perm[nS:nS + nA], perm[nS + nA:]
                S, A = S[: int(len(S) * frac)], A[: int(len(A) * frac)]
                hp = pls_tune(X[S], yy[S], np.arange(len(S)).astype(str), "reg")
                tt = TTPA4("fleet", seed=seed, broaden="test").fit_source(X[S], ax, k_min=hp["k"] + 5).fit_target(X[A]); tt.configure(**FROZEN); st = tt.stats
                p_nc = fit_model(X[S], yy[S], hp, "reg")(X[T]); p_tt = fit_model(tt.source_as_target(), yy[S], hp, "reg")(tt.target_input(X[T]))
                p_dmc = fit_model(X[S], yy[S], hp, "reg")(X[T] - X[A].mean(0) + X[S].mean(0))
                rows.append(dict(dataset=name, size_fraction=frac, draw=draw, n_S=len(S), n_A=len(A), n_T=len(T), operator=int(st["theta_gate"]), broadening=int(st["broad_gate"]),
                                 outspan=int(st["dperp_gate"]), inspan_gate_would_fire=int(st["Dpar_test"] > st["Dpar_null_max"]), inspan_applied=int(st["dpar_used"]),
                                 rmsep_NC=rmsep(yy[T], p_nc), rmsep_DMC=rmsep(yy[T], p_dmc), rmsep_TTPA=rmsep(yy[T], p_tt)))
            print("null", name, frac, f"{time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "null_draws.csv", index=False)

# ------------------------------------------------------------------ hook: iterate over the stored confirmatory designs
class FM(Family):
    JOB = None; RES = []
    def __init__(self, stage, name, kind="reg", context="fleet"):
        self.stage, self.name, self.kind, self.context = stage, name, kind, context
        self.path = ROOT / "results" / "e12" / stage / f"{name}.json"; self.runs = json.loads(self.path.read_text()); self.index = {(r["scenario"], r["rep"]): r for r in self.runs}; self.update = True
    def done(self): return False
    def add(self, scen, rep, Xs, ys, Xa, Xt, yt, axis, std=None, ya=None, groups_s=None, extra=None, dann=True):
        r = self.index[(scen, rep)]; Xs, Xa, Xt = (np.asarray(v, float) for v in (Xs, Xa, Xt))
        out = FM.JOB(Xs, np.asarray(ys, float), Xa, Xt, np.asarray(yt, float), axis, r["hp"], self.context, rep)
        FM.RES.append(dict(family=self.name, scenario=scen, rep=rep, **out)); print(self.name[:20], scen[:30], rep, flush=True)
    def finish(self): pass

def over_confirmatory(job, fname, names=("icrisat_handheld", "idrc_A", "idrc_B", "idrc_new", "grain_year", "cereal_species")):
    run_e12_conf.Family = FM; FM.JOB = job; FM.RES = []
    for n in names:
        run_e12_conf.conf(n); pd.DataFrame(FM.RES).to_csv(OUT / fname, index=False)

# ------------------------------------------------------------------ #9 sensitivity of subspace and baseline
def sens():
    cfgs = {"k_var99.9": dict(var=0.999), "k_LV+5_only": dict(var=0.0), "baseline_none": dict(none=True), "baseline_linear": dict(deg=1)}
    def job(Xs, ys, Xa, Xt, yt, axis, hp, context, rep):
        res = {}
        for name, c in cfgs.items():
            kw = {k: v for k, v in c.items() if k != "none"}
            tt = TTPA4(context, seed=rep, broaden="test", **kw).fit_source(Xs, axis, k_min=hp["k"] + 5)
            if c.get("none"): tt.L = np.zeros((len(axis), 0))
            tt.fit_target(Xa); tt.configure(**FROZEN)
            res[name] = rmsep(yt, fit_model(tt.source_as_target(), ys, hp, "reg")(tt.target_input(Xt)))
        return res
    over_confirmatory(job, "sensitivity_k_baseline.csv")

# ------------------------------------------------------------------ #10 di-PLS penalty sensitivity
def dipls():
    from diPLSlib.models import DIPLS
    from spectwin.prep import Prep
    def job(Xs, ys, Xa, Xt, yt, axis, hp, context, rep):
        res = {}; P = Prep(hp["prep"]).fit(Xs)
        for lam in (10.0, 100.0, 1000.0, 10000.0):
            try:
                m = DIPLS(A=int(min(hp["k"], 20)), l=lam, heuristic=False, rescale="Target"); m.fit(P.transform(Xs), ys, P.transform(Xs), P.transform(Xa))
                res[f"diPLS_l{int(lam)}"] = rmsep(yt, np.asarray(m.predict(P.transform(Xt))).ravel())
            except Exception as e:
                res[f"diPLS_l{int(lam)}"] = np.nan
        return res
    over_confirmatory(job, "dipls_penalty.csv")

if __name__ == "__main__":
    globals()[sys.argv[1]]()
