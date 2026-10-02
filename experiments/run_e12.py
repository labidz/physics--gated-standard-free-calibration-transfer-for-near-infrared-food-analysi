"""E12: standard-free calibration transfer, round 2.

  python experiments/run_e12.py dev  <family,...>   development families (the E3 families; corn: reps 0-4)
  python experiments/run_e12.py conf <family,...>   confirmatory families (fresh data; PROTOCOL Amendment A)

Every replicate evaluates, on identical (source S, unlabelled target A, target test T) splits:
  NC, DMC, CORAL, di-PLS, uDOP, TCR, OT-DA, DANN, CT-VPdtw, TTPA v3 (fleet/chem/primary),
  TTPA v4 and its ablations (no gate, no in-span correction, no broadening),
  and, where the design provides them, standards-based DS/PDS/SBC and the target-label oracle.
One JSON per family in results/e12/<stage>/.
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
warnings.filterwarnings("ignore")
import torch
torch.set_num_threads(1)
from run_e3 import fit_model, halves, score, _cxy  # noqa: F401  (imports apply the diPLSlib patch)
from spectwin.bench import load
from spectwin.baselines import pls_tune
from spectwin.prep import Prep
from spectwin.transfer import TTPA, ds_transform, pds_transform, coral_transform
from spectwin.transfer2 import TTPA4, udop_predict, tcr_predict, ot_predict, dann_predict, ctvpdtw_batch


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:  # a failing baseline is recorded as missing, never silently replaced
        print("   !", getattr(fn, "__name__", fn), type(e).__name__, str(e)[:120], flush=True)
        return None


import os
# TTPA v4 variants. All variants share ONE fit (operator with and without broadening on the target and on the
# same 19 Monte-Carlo pseudo-targets); they differ only in which fitted components configure() applies.
VARIANTS = {"TTPA4": dict(broaden="auto", gate_par=True, gof="auto"),
            "TTPA4-nogof": dict(broaden="auto", gate_par=True, gof=False),
            "TTPA4-abstain": dict(broaden="auto", gate_par=True, gof="auto", chem_theta="abstain"),
            "TTPA4-pargate-off": dict(broaden="auto", gate_par=False, gof="auto"),
            "TTPA4-nobroad": dict(broaden=False, gate_par=True, gof="auto"),
            "TTPA4-broadtest": dict(broaden="test", gate_par=True, gof="auto"),
            "TTPA4-nogate": dict(gate=False, broaden=False, gate_par=False, gof=False),
            "TTPA4-perp": dict(broaden="auto", gate_par=True, use_par=False, gof="auto"),
            # development round 4 (after the first full development evaluation; PROTOCOL Amendment A)
            "TTPA4-safe": dict(broaden="auto", gate_par=False, gof=True, chem_mean="none"),
            "TTPA4-safe-abstain": dict(broaden="auto", gate_par=False, gof=True, chem_mean="none", chem_theta="abstain")}


def ttpa4_all(Xs, ys, Xa, Xt, axis, hp, kind, context, seed):
    out, info = {}, {}
    tt = TTPA4(context, seed=seed, broaden="test").fit_source(Xs, axis, k_min=hp["k"] + 5).fit_target(Xa)
    for name, kw in VARIANTS.items():
        tt.configure(**kw)
        out[name] = fit_model(tt.source_as_target(), ys, hp, kind)(tt.target_input(Xt))
        info[name] = dict(tt.stats)
    return out, info


def run_one(Xs, ys, Xa, Xt, axis, kind, primary, std=None, ya=None, groups_s=None, seed=0, dann=True):
    gs = groups_s if groups_s is not None else np.arange(len(Xs)).astype(str)
    hp = pls_tune(Xs, ys, gs, kind)
    f = fit_model(Xs, ys, hp, kind)
    out = {"NC": f(Xt), "DMC": f(Xt - Xa.mean(0) + Xs.mean(0))}
    out["CORAL"] = safe(lambda: f(coral_transform(Xs, Xa)(Xt)))
    if kind == "reg":
        def _dipls():
            from diPLSlib.models import DIPLS
            P = Prep(hp["prep"]).fit(Xs)
            m = DIPLS(A=int(min(hp["k"], 20)), l=0, heuristic=True, rescale="Target")
            m.fit(P.transform(Xs), ys, P.transform(Xs), P.transform(Xa))
            return np.asarray(m.predict(P.transform(Xt))).ravel()
        out["diPLS"] = safe(_dipls)
    out["uDOP"] = safe(udop_predict, Xs, ys, Xa, Xt, hp, kind)
    out["TCR"] = safe(tcr_predict, Xs, ys, Xa, Xt, hp, kind)
    out["OT-DA"] = safe(ot_predict, Xs, ys, Xa, Xt, hp, kind)
    if dann:
        out["DANN"] = safe(dann_predict, Xs, ys, Xa, Xt, hp, kind, 500, seed)
    info = {}
    for mode in ["fleet", "chem"]:
        tt = TTPA(mode).fit_source(Xs, axis, k_min=hp["k"] + 5).fit_target(Xa)
        out[f"TTPA3-{mode}"] = fit_model(tt.source_as_target(), ys, hp, kind)(tt.target_input(Xt))
    out["TTPA3"] = out[f"TTPA3-{primary}"]
    o4, i4 = ttpa4_all(Xs, ys, Xa, Xt, axis, hp, kind, primary, seed)
    out.update(o4); info.update(i4)
    if std is not None:
        Ss, St = std
        out["DS (std)"] = safe(lambda: f(ds_transform(Ss, St)(Xt)))
        out["PDS (std)"] = safe(lambda: f(pds_transform(Ss, St)(Xt)))
        if kind == "reg":
            def _sbc():
                a, b = np.polyfit(f(St), f(Ss), 1)
                return a * f(Xt) + b
            out["SBC (std)"] = safe(_sbc)
    if ya is not None and len(Xa) >= 8:
        hpa = pls_tune(Xa, ya, np.arange(len(Xa)).astype(str), kind)
        out["Oracle (target labels)"] = fit_model(Xa, ya, hpa, kind)(Xt)
    return out, info, hp


class Family:
    """Collects replicates; CT-VPdtw is run once per family through R at the end."""

    def __init__(self, stage, name, kind="reg", context="fleet"):
        self.stage, self.name, self.kind, self.context = stage, name, kind, context
        self.path = ROOT / "results" / "e12" / stage / f"{name}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.runs, self.jobs = [], []
        self.update = os.environ.get("E12_UPDATE") == "1" and self.path.exists()
        if self.update:   # recompute only the TTPA v4 variants on the stored design; other methods untouched
            self.runs = json.loads(self.path.read_text())
            self.index = {(r["scenario"], r["rep"]): r for r in self.runs}

    def done(self):
        return self.path.exists() and not self.update

    def add(self, scen, rep, Xs, ys, Xa, Xt, yt, axis, std=None, ya=None, groups_s=None, extra=None, dann=True):
        t0 = time.time()
        Xs, Xa, Xt = (np.asarray(v, float) for v in (Xs, Xa, Xt))
        if self.update:
            r = self.index[(scen, rep)]
            for k in [k for k in r["pred"] if k.startswith("TTPA4")]:
                r["pred"].pop(k); r["info"].pop(k, None)
            o4, i4 = ttpa4_all(Xs, ys, Xa, Xt, axis, r["hp"], self.kind, self.context, rep)
            r["pred"].update({k: np.asarray(v).tolist() for k, v in o4.items()}); r["info"].update(i4)
            print(f"{scen[:40]:40s} r{rep} updated TTPA4 variants ({time.time() - t0:.0f}s)", flush=True)
            return
        out, info, hp = run_one(Xs, ys, Xa, Xt, axis, self.kind, self.context, std=std, ya=ya, groups_s=groups_s, seed=rep, dann=dann)
        r = dict(scenario=scen, rep=rep, kind=self.kind, context=self.context, hp=hp, info=info, y=np.asarray(yt).tolist(),
                 pred={k: (np.asarray(v).tolist() if v is not None else None) for k, v in out.items()}, n=[len(Xs), len(Xa), len(Xt)])
        if extra:
            r.update(extra)
        self.runs.append(r)
        if self.kind == "reg":
            from spectwin.transfer2 import _cap
            cs, cy = _cap(Xs, np.asarray(ys, float))
            self.jobs.append((len(self.runs) - 1, (cs, cy, Xa, Xt)))
        mk = "RMSEP" if self.kind == "reg" else "accuracy"
        show = ["NC", "DMC", "diPLS", "uDOP", "TCR", "OT-DA", "DANN", "TTPA3", "TTPA4"]
        ms = {k: score(self.kind, yt, out[k]) for k in show if out.get(k) is not None}
        print(f"{scen[:40]:40s} r{rep} " + " ".join(f"{k}={v[mk]:.4g}" for k, v in ms.items()) +
              f"  gates={info['TTPA4']['theta_gate']:d}{info['TTPA4']['dperp_gate']:d} ({time.time() - t0:.0f}s)", flush=True)

    def finish(self):
        if self.jobs and not self.update:
            t0 = time.time()
            res = ctvpdtw_batch([j for _, j in self.jobs])
            for (i, _), (p, inf) in zip(self.jobs, res):
                self.runs[i]["pred"]["CT-VPdtw"] = p.tolist() if p is not None else None
                self.runs[i]["info"]["CT-VPdtw"] = inf
            print(f"   CT-VPdtw for {len(self.jobs)} replicates ({time.time() - t0:.0f}s); failures: "
                  f"{sum(p is None for p, _ in res)}", flush=True)
        for r in self.runs:
            r["metrics"] = {k: score(self.kind, r["y"], v) for k, v in r["pred"].items() if v is not None}
        self.path.write_text(json.dumps(self.runs))


# ============================================================================ development families (E3 designs)
def dev(name):
    if name == "corn":
        F = Family("dev", "corn", context="fleet")
        if F.done(): return
        d = load("corn"); inst = d.domains["instrument"]
        for prop in ["moisture", "oil", "protein", "starch"]:
            y = d.targets[prop][:80]
            for src, tgt in [("m5", "mp5"), ("m5", "mp6"), ("mp5", "m5"), ("mp5", "mp6"), ("mp6", "m5"), ("mp6", "mp5")]:
                Xs_all = d.X[inst == src]; Xt_all = d.X[inst == tgt]
                for rep in range(5):
                    perm = np.random.default_rng(rep).permutation(80); S, A, T = perm[:40], perm[40:60], perm[60:]
                    F.add(f"corn {prop} {src}->{tgt}", rep, Xs_all[S], y[S], Xt_all[A], Xt_all[T], y[T], d.axis,
                          std=(Xs_all[A].astype(float), Xt_all[A].astype(float)), ya=y[A], extra=dict(prop=prop, pair=f"{src}->{tgt}"))
        F.finish()
    elif name == "ch2021":
        F = Family("dev", "ch2021", context="fleet")
        if F.done(): return
        d = load("ch2021"); inst = d.domains["instrument"]; role = d.domains["role"]; y = d.targets["y"]
        S = np.where(role == "cal")[0]
        std1 = {g: i for i, g in zip(np.where((role == "std") & (inst == "1"))[0], d.groups[(role == "std") & (inst == "1")])}
        for k in ["2", "3", "4", "5", "6", "7"]:
            A = np.where((role == "std") & (inst == k))[0]; T = np.where((role == "val") & (inst == k))[0]
            pairs = [(std1[g], i) for i, g in zip(A, d.groups[A]) if g in std1]
            F.add(f"ch2021 unit1->unit{k}", 0, d.X[S], y[S], d.X[A], d.X[T], y[T], d.axis,
                  std=(d.X[[a for a, _ in pairs]].astype(float), d.X[[b for _, b in pairs]].astype(float)))
        F.finish()
    elif name == "tablets":
        F = Family("dev", "tablets", context="fleet")
        if F.done(): return
        d = load("tablets"); inst = d.domains["instrument"]; y = d.targets["assay"]; g = d.groups
        cal = np.array([s.startswith("calibrate") for s in g]); val = np.array([s.startswith("validate") for s in g]); tst = np.array([s.startswith("test") for s in g])
        S = np.where(cal & (inst == "1"))[0]; A = np.where(val & (inst == "2"))[0]; T = np.where(tst & (inst == "2"))[0]; A1 = np.where(val & (inst == "1"))[0]
        F.add("tablets inst1->inst2 (non-food)", 0, d.X[S], y[S], d.X[A], d.X[T], y[T], d.axis, std=(d.X[A1].astype(float), d.X[A].astype(float)), ya=y[A])
        F.finish()
    elif name in ("wheat", "cassava", "grapes", "mango_season"):
        spec = {"wheat": ("wheat drift (2 months)", "wheat", "protein", "reg", "fleet", 10),
                "cassava": ("cassava new year", "cassava", "TBC", "reg", "chem", 10),
                "grapes": ("grapes new harvest dates", "grapes", "variety", "cls", "chem", 10),
                "mango_season": ("mango season 1-3 -> 4", "mango", "DM", "reg", "chem", 5)}[name]
        fam, ds, target, kind, ctx, reps = spec
        F = Family("dev", fam, kind=kind, context=ctx)
        if F.done(): return
        d = load(ds); y = d.targets[target]; y = y.astype(float) if kind == "reg" else y.astype(str)
        S = np.where(d.split == "train")[0]; Tpool = np.where(d.split == "test")[0]
        for rep in range(reps):
            a, t = halves(d.groups[Tpool], np.random.default_rng(rep)); A, T = Tpool[a], Tpool[t]
            F.add(fam, rep, d.X[S], y[S], d.X[A], d.X[T], y[T], d.axis, ya=y[A], groups_s=d.groups[S])
        F.finish()
    elif name == "mango_temp":
        fam = "mango fruit temperature low-mid -> high"
        F = Family("dev", fam, context="fleet")
        if F.done(): return
        d = load("mango"); t = d.domains["temperature"]; y = d.targets["DM"]; grp = np.unique(d.groups[t != "No"])
        for rep in range(5):
            g = np.random.default_rng(rep).permutation(grp); h = len(g) // 2
            G1, GA, GT = set(g[:h]), set(g[h:h + (len(g) - h) // 2]), set(g[h + (len(g) - h) // 2:])
            S = np.where(np.isin(t, ["Low", "Mid"]) & np.isin(d.groups, list(G1)))[0]
            A = np.where((t == "High") & np.isin(d.groups, list(GA)))[0]; T = np.where((t == "High") & np.isin(d.groups, list(GT)))[0]
            F.add(fam, rep, d.X[S], y[S], d.X[A], d.X[T], y[T], d.axis, ya=y[A], groups_s=d.groups[S])
        F.finish()
    elif name in ("plum_loco", "ch2018_lopo"):
        fam, ds, target, domain = {"plum_loco": ("plum leave-one-cultivar-out SSC", "plum", "SSC", "cultivar"),
                                   "ch2018_lopo": ("ch2018 leave-one-product-out protein", "ch2018", "protein", "product")}[name]
        F = Family("dev", fam, context="chem")
        if F.done(): return
        d = load(ds); y = d.targets[target].astype(float); dom = d.domains[domain]; ok = np.isfinite(y)
        for u in [u for u in np.unique(dom[ok]) if (dom[ok] == u).sum() >= 40]:
            S = np.where(ok & (dom != u))[0]; Tpool = np.where(ok & (dom == u))[0]
            a, t = halves(d.groups[Tpool], np.random.default_rng(0)); A, T = Tpool[a], Tpool[t]
            F.add(f"{fam} [{u}]", 0, d.X[S], y[S], d.X[A], d.X[T], y[T], d.axis, ya=y[A], groups_s=d.groups[S], extra=dict(domain=str(u)))
        F.finish()
    elif name == "ligno":
        F = Family("dev", "ligno", context="chem")
        if F.done(): return
        d = load("ligno"); role = d.domains["role"]; y = d.targets["glucose"]
        S = np.where(role == "offline")[0]; on = np.where(role == "online")[0]
        A = on[: int(0.1 * len(on))]; T = np.array([i for i in on[int(0.1 * len(on)):] if np.isfinite(y[i])])
        F.add("ligno off-line -> in-line (glucose)", 0, d.X[S], y[S], d.X[A], d.X[T], y[T], d.axis)
        F.finish()
    elif name == "simfood":
        F = Family("dev", "simfood", context="fleet")
        if F.done(): return
        SIMD = ROOT / "data" / "sim"
        for target_name, col, system in [("lipid", 1, "emulsion"), ("protein", 2, "powder")]:
            for su, tu in [(3, 4), (3, 5), (4, 5), (0, 1), (0, 2), (6, 7), (9, 10), (9, 11)]:
                zs = np.load(SIMD / f"simfood_vnir_unit{su:02d}.npz"); zt = np.load(SIMD / f"simfood_vnir_unit{tu:02d}.npz")
                S = np.where((zs["system"] == system) & (zs["split"] == "train"))[0]; Tpool = np.where((zs["system"] == system) & (zs["split"] == "test"))[0]
                y = zs["comp"][:, col]
                for rep in range(3):
                    perm = np.random.default_rng(rep).permutation(Tpool); A, T = perm[: len(perm) // 2], perm[len(perm) // 2:]
                    F.add(f"simfood {system}:{target_name} u{su}->u{tu}", rep, zs["X"][S], y[S], zt["X"][A], zt["X"][T], y[T], zs["axis"].astype(float),
                          std=(zs["X"][A].astype(float), zt["X"][A].astype(float)), ya=y[A])
        F.finish()
    else:
        raise ValueError(name)


DEV = ["corn", "ch2021", "tablets", "wheat", "cassava", "grapes", "mango_season", "mango_temp", "plum_loco", "ch2018_lopo", "ligno", "simfood"]

if __name__ == "__main__":
    stage, which = sys.argv[1], sys.argv[2].split(",")
    t0 = time.time()
    for w in which:
        if stage == "dev":
            dev(w)
        else:
            from run_e12_conf import conf
            conf(w)
        print(f"== {stage}:{w} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
