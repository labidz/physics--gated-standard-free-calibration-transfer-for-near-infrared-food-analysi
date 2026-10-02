"""E3: calibration transfer without labels or standards (SpecTwin-TTPA) vs reference methods.

Resumable: one JSON per scenario family in results/e3/.
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
from spectwin.bench import load
from spectwin.baselines import pls_tune
from spectwin.chemo import PLS, reg_metrics, cls_metrics
from spectwin.prep import Prep
from spectwin.transfer import TTPA, ds_transform, pds_transform, coral_transform

import sklearn.utils.validation as _v
import diPLSlib.models as _dm
_orig = _dm.check_X_y
def _cxy(*a, **k):
    k.pop("force_all_finite", None); return _orig(*a, **k)
_dm.check_X_y = _cxy
_orig_ca = _dm.check_array
def _ca(*a, **k):
    k.pop("force_all_finite", None); return _orig_ca(*a, **k)
_dm.check_array = _ca

OUT = ROOT / "results" / "e3"; OUT.mkdir(parents=True, exist_ok=True)


def fit_model(X, y, hp, kind):
    P = Prep(hp["prep"]).fit(X)
    if kind == "cls":
        classes = np.unique(y); Y = (y[:, None] == classes[None, :]).astype(float)
        m = PLS(hp["k"]).fit(P.transform(X), Y)
        return lambda Z: classes[m.predict(P.transform(Z), hp["k"]).argmax(1)]
    m = PLS(hp["k"]).fit(P.transform(X), y)
    return lambda Z: m.predict(P.transform(Z), hp["k"])


def run_one(Xs, ys, Xa, Xt, yt, axis, kind, primary, std=None, ya=None, groups_s=None):
    """Return dict method -> predictions on Xt. std = (Xs_std, Xt_std) paired standards (optional)."""
    gs = groups_s if groups_s is not None else np.arange(len(Xs)).astype(str)
    hp = pls_tune(Xs, ys, gs, kind)
    f = fit_model(Xs, ys, hp, kind)
    out = {"NC": f(Xt), "DMC": f(Xt - Xa.mean(0) + Xs.mean(0))}
    try:
        out["CORAL"] = f(coral_transform(Xs, Xa)(Xt))
    except Exception:
        out["CORAL"] = None
    if kind == "reg":
        try:
            from diPLSlib.models import DIPLS
            P = Prep(hp["prep"]).fit(Xs)
            mdl = DIPLS(A=int(min(hp["k"], 20)), l=0, heuristic=True, rescale="Target")
            mdl.fit(P.transform(Xs), ys, P.transform(Xs), P.transform(Xa))
            out["diPLS"] = np.asarray(mdl.predict(P.transform(Xt))).ravel()
        except Exception as e:
            out["diPLS"] = None
    info = {}
    for mode in ["fleet", "chem"]:
        tt = TTPA(mode).fit_source(Xs, axis, k_min=hp["k"] + 5).fit_target(Xa)
        g = fit_model(tt.source_as_target(), ys, hp, kind)
        out[f"TTPA-{mode}"] = g(tt.target_input(Xt))
        info[mode] = dict(theta=tt.theta.tolist(), k=tt.k, obj=tt.obj, obj0=tt.obj0)
    out["SpecTwin-TTPA"] = out[f"TTPA-{primary}"]
    if std is not None:
        Ss, St = std
        out["DS (std)"] = f(ds_transform(Ss, St)(Xt))
        pds = pds_transform(Ss, St)
        out["PDS (std)"] = f(pds(Xt))
        info["pds_window"] = pds.window
        if kind == "reg":
            a, b = np.polyfit(f(St), f(Ss), 1)
            out["SBC (std)"] = a * f(Xt) + b
    if ya is not None and len(Xa) >= 8:
        hpa = pls_tune(Xa, ya, np.arange(len(Xa)).astype(str), kind)
        out["Oracle (target labels)"] = fit_model(Xa, ya, hpa, kind)(Xt)
    out["_source_in_domain_hp"] = hp
    return out, info


def score(kind, y, p):
    if p is None:
        return None
    return reg_metrics(y, p) if kind == "reg" else cls_metrics(y, p)


def save(name, runs):
    (OUT / f"{name}.json").write_text(json.dumps(runs))


def halves(groups, rng):
    ug = np.unique(groups); rng.shuffle(ug)
    a = set(ug[: len(ug) // 2])
    m = np.array([g in a for g in groups])
    return np.where(m)[0], np.where(~m)[0]


def record(runs, scen, rep, kind, yt, out, info, extra=None):
    r = dict(scenario=scen, rep=rep, kind=kind, info=info, hp=out.pop("_source_in_domain_hp"), metrics={}, y=np.asarray(yt).tolist())
    for k, p in out.items():
        r["metrics"][k] = score(kind, yt, p)
    r["pred"] = {k: (np.asarray(p).tolist() if p is not None else None) for k, p in out.items()}
    if extra:
        r.update(extra)
    runs.append(r)
    mk = "RMSEP" if kind == "reg" else "accuracy"
    print(f"{scen:34s} rep{rep}: " + "  ".join(f"{k}={v[mk]:.4g}" for k, v in r["metrics"].items() if v), flush=True)


# ----------------------------------------------------------------------------- scenarios
def corn():
    if (OUT / "corn.json").exists():
        return
    d = load("corn"); inst = d.domains["instrument"]; runs = []
    for prop in ["moisture", "oil", "protein", "starch"]:
        y = d.targets[prop][:80]
        for src, tgt in [("m5", "mp5"), ("m5", "mp6"), ("mp5", "m5"), ("mp5", "mp6"), ("mp6", "m5"), ("mp6", "mp5")]:
            Xs_all = d.X[inst == src].astype(float); Xt_all = d.X[inst == tgt].astype(float)
            for rep in range(10):
                perm = np.random.default_rng(rep).permutation(80); S, A, T = perm[:40], perm[40:60], perm[60:]
                out, info = run_one(Xs_all[S], y[S], Xt_all[A], Xt_all[T], y[T], d.axis, "reg", "fleet",
                                    std=(Xs_all[A], Xt_all[A]), ya=y[A])
                record(runs, f"corn {prop} {src}->{tgt}", rep, "reg", y[T], out, info, dict(prop=prop, pair=f"{src}->{tgt}"))
    save("corn", runs)


def ch2021():
    if (OUT / "ch2021.json").exists():
        return
    d = load("ch2021"); inst = d.domains["instrument"]; role = d.domains["role"]; y = d.targets["y"]; runs = []
    S = np.where(role == "cal")[0]
    std_ids_1 = {g: i for i, g in zip(np.where((role == "std") & (inst == "1"))[0], d.groups[(role == "std") & (inst == "1")])}
    for k in ["2", "3", "4", "5", "6", "7"]:
        A = np.where((role == "std") & (inst == k))[0]
        T = np.where((role == "val") & (inst == k))[0]
        pairs = [(std_ids_1[g], i) for i, g in zip(A, d.groups[A]) if g in std_ids_1]
        Ss = d.X[[a for a, _ in pairs]].astype(float); St = d.X[[b for _, b in pairs]].astype(float)
        out, info = run_one(d.X[S].astype(float), y[S], d.X[A].astype(float), d.X[T].astype(float), y[T], d.axis, "reg", "fleet", std=(Ss, St))
        record(runs, f"ch2021 unit1->unit{k}", 0, "reg", y[T], out, info, dict(unit=k))
    save("ch2021", runs)


def tablets():
    if (OUT / "tablets.json").exists():
        return
    d = load("tablets"); inst = d.domains["instrument"]; y = d.targets["assay"]; g = d.groups; runs = []
    cal = np.array([s.startswith("calibrate") for s in g]); val = np.array([s.startswith("validate") for s in g]); tst = np.array([s.startswith("test") for s in g])
    S = np.where(cal & (inst == "1"))[0]; A = np.where(val & (inst == "2"))[0]; T = np.where(tst & (inst == "2"))[0]
    A1 = np.where(val & (inst == "1"))[0]
    out, info = run_one(d.X[S].astype(float), y[S], d.X[A].astype(float), d.X[T].astype(float), y[T], d.axis, "reg", "fleet",
                        std=(d.X[A1].astype(float), d.X[A].astype(float)), ya=y[A])
    record(runs, "tablets inst1->inst2 (non-food)", 0, "reg", y[T], out, info)
    save("tablets", runs)


def holdout_family(name, dsname, target, kind, S, Tpool, primary, reps=5, group_key=None):
    if (OUT / f"{name}.json").exists():
        return
    d = load(dsname); y = d.targets[target]; y = y.astype(float) if kind == "reg" else y.astype(str); runs = []
    grp = d.groups if group_key is None else d.domains[group_key]
    for rep in range(reps):
        a, t = halves(grp[Tpool], np.random.default_rng(rep))
        A, T = Tpool[a], Tpool[t]
        out, info = run_one(d.X[S].astype(float), y[S], d.X[A].astype(float), d.X[T].astype(float), y[T], d.axis, kind, primary,
                            ya=y[A], groups_s=d.groups[S])
        record(runs, name, rep, kind, y[T], out, info)
    save(name, runs)


def leave_one_out_family(name, dsname, target, domain, primary, min_n=40, max_domains=None):
    if (OUT / f"{name}.json").exists():
        return
    d = load(dsname); y = d.targets[target].astype(float); dom = d.domains[domain]; runs = []
    ok = np.isfinite(y)
    doms = [u for u in np.unique(dom[ok]) if (dom[ok] == u).sum() >= min_n]
    for u in doms[:max_domains]:
        S = np.where(ok & (dom != u))[0]; Tpool = np.where(ok & (dom == u))[0]
        a, t = halves(d.groups[Tpool], np.random.default_rng(0))
        A, T = Tpool[a], Tpool[t]
        out, info = run_one(d.X[S].astype(float), y[S], d.X[A].astype(float), d.X[T].astype(float), y[T], d.axis, "reg", primary,
                            ya=y[A], groups_s=d.groups[S])
        record(runs, f"{name} [{u}]", 0, "reg", y[T], out, info, dict(domain=str(u)))
    save(name, runs)


def ligno():
    if (OUT / "ligno.json").exists():
        return
    d = load("ligno"); role = d.domains["role"]; y = d.targets["glucose"]; runs = []
    S = np.where(role == "offline")[0]; on = np.where(role == "online")[0]
    A = on[: int(0.1 * len(on))]
    T = np.array([i for i in on[int(0.1 * len(on)):] if np.isfinite(y[i])])
    out, info = run_one(d.X[S].astype(float), y[S], d.X[A].astype(float), d.X[T].astype(float), y[T], d.axis, "reg", "chem")
    record(runs, "ligno off-line -> in-line (glucose)", 0, "reg", y[T], out, info)
    save("ligno", runs)


def simfood():
    """Controlled unit->unit transfer on SimFood-RT (known instrument parameters)."""
    if (OUT / "simfood.json").exists():
        return
    SIMD = ROOT / "data" / "sim"
    units = json.loads((SIMD / "units.json").read_text()); runs = []
    pairs = [(3, 4), (3, 5), (4, 5), (0, 1), (0, 2), (6, 7), (9, 10), (9, 11)]
    for target_name, col, system in [("lipid", 1, "emulsion"), ("protein", 2, "powder")]:
        for su, tu in pairs:
            zs = np.load(SIMD / f"simfood_vnir_unit{su:02d}.npz"); zt = np.load(SIMD / f"simfood_vnir_unit{tu:02d}.npz")
            sysl, split, comp = zs["system"], zs["split"], zs["comp"]
            S = np.where((sysl == system) & (split == "train"))[0]; Tpool = np.where((sysl == system) & (split == "test"))[0]
            y = comp[:, col]
            for rep in range(3):
                perm = np.random.default_rng(rep).permutation(Tpool); A, T = perm[: len(perm) // 2], perm[len(perm) // 2:]
                out, info = run_one(zs["X"][S].astype(float), y[S], zt["X"][A].astype(float), zt["X"][T].astype(float), y[T],
                                    zs["axis"].astype(float), "reg", "fleet", std=(zs["X"][A].astype(float), zt["X"][A].astype(float)), ya=y[A])
                true_rel = units[tu]["shift"] - units[su]["shift"]
                if str(zs["axis_unit"]) == "cm-1":
                    true_rel = None
                record(runs, f"simfood {system}:{target_name} u{su}->u{tu}", rep, "reg", y[T], out, info,
                       dict(true_rel_shift_nm=true_rel, est_shift=info["fleet"]["theta"][0], cls=units[su]["cls"]))
    save("simfood", runs)


if __name__ == "__main__":
    which = sys.argv[1].split(",") if len(sys.argv) > 1 else ["corn", "ch2021", "tablets", "wheat", "cassava", "grapes", "mango_season", "mango_temp", "plum_loco", "ch2018_lopo", "ligno"]
    t0 = time.time()
    for w in which:
        if w == "corn": corn()
        elif w == "ch2021": ch2021()
        elif w == "tablets": tablets()
        elif w == "wheat":
            d = load("wheat"); holdout_family("wheat drift (2 months)", "wheat", "protein", "reg", np.where(d.split == "train")[0], np.where(d.split == "test")[0], "fleet", reps=10)
        elif w == "cassava":
            d = load("cassava"); holdout_family("cassava new year", "cassava", "TBC", "reg", np.where(d.split == "train")[0], np.where(d.split == "test")[0], "chem", reps=10)
        elif w == "grapes":
            d = load("grapes"); holdout_family("grapes new harvest dates", "grapes", "variety", "cls", np.where(d.split == "train")[0], np.where(d.split == "test")[0], "chem", reps=10)
        elif w == "mango_season":
            d = load("mango"); holdout_family("mango season 1-3 -> 4", "mango", "DM", "reg", np.where(d.split == "train")[0], np.where(d.split == "test")[0], "chem", reps=5)
        elif w == "mango_temp":
            name = "mango fruit temperature low-mid -> high"
            if not (OUT / f"{name}.json").exists():
                d = load("mango"); t = d.domains["temperature"]; y = d.targets["DM"]; runs = []
                grp = np.unique(d.groups[t != "No"])
                for rep in range(5):
                    g = np.random.default_rng(rep).permutation(grp); h = len(g) // 2
                    G1, GA, GT = set(g[:h]), set(g[h:h + (len(g) - h) // 2]), set(g[h + (len(g) - h) // 2:])
                    S = np.where(np.isin(t, ["Low", "Mid"]) & np.isin(d.groups, list(G1)))[0]
                    A = np.where((t == "High") & np.isin(d.groups, list(GA)))[0]
                    T = np.where((t == "High") & np.isin(d.groups, list(GT)))[0]
                    out, info = run_one(d.X[S].astype(float), y[S], d.X[A].astype(float), d.X[T].astype(float), y[T], d.axis, "reg", "fleet",
                                        ya=y[A], groups_s=d.groups[S])
                    record(runs, name, rep, "reg", y[T], out, info)
                save(name, runs)
        elif w == "plum_loco":
            leave_one_out_family("plum leave-one-cultivar-out SSC", "plum", "SSC", "cultivar", "chem")
        elif w == "ch2018_lopo":
            leave_one_out_family("ch2018 leave-one-product-out protein", "ch2018", "protein", "product", "chem")
        elif w == "ligno": ligno()
        elif w == "simfood": simfood()
        print(f"== {w} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
