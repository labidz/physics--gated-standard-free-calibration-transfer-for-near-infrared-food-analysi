"""E12 confirmatory families (PROTOCOL Amendment A). Data never used in any development step.

C1 IDRC-2016 wheat, manufacturer A fleet  : A1 -> A2, A3, A.Test (same 248 samples), fleet context
C2 IDRC-2016 wheat, manufacturer B fleet  : B1 -> B2, B3, B.Test (same 248 samples), fleet context
C3 IDRC-2016 wheat, new units + new samples: A1 -> vliadA, B1 -> vliadB, C.Test -> vliadC, fleet context
C4 ICRISAT cereals, benchtop FOSS DS2500 -> handheld Hone EVT5 (1350-2498 nm), fleet context
C5 Sugarcane, ASD LabSpec -> 7 portable spectrometers (each on its own grid and range), fleet context
C6 Grain, new harvest year 2022 -> 2023 on two handheld spectrometers (population shift), chem context
C7 Cereals, new species (sorghum <-> pearl millet, FOSS DS2500) (population shift), chem context
P  Published protocols: IDRC B1->B2, B1->B3, B3->B2 with Kennard-Stone 198/50 (as in CTAI, Molecules 2019);
   corn m5->mp6 Kennard-Stone 64/16 (as in CTAI) -- corn is a development dataset, reported separately.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import rdata
from run_e12 import Family, halves

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "conf"


def _rda(name):
    d = rdata.read_rda(RAW / name)
    return {str(k): np.asarray(v) for k, v in list(d.values())[0].items()}


def idrc():
    w = _rda("idrc2016_wheat.rda")
    return w, w["Y.CAL"][:, 0].astype(float), w["y.Valid"][:, 0].astype(float)


def icrisat():
    f = pd.read_csv(RAW / "icrisat_foss.csv", sep=";", decimal=",")
    e = pd.read_csv(RAW / "icrisat_evt5.csv", sep=";", decimal=",")
    wl_f = np.array([float(c) for c in f.columns if re.fullmatch(r"\d+(\.\d+)?", str(c))])
    wl_e_cols = [c for c in e.columns if re.fullmatch(r"\d+(\.\d+)?", str(c))]
    wl_e = np.array([float(c) for c in wl_e_cols])
    Xf = f[[c for c in f.columns if re.fullmatch(r"\d+(\.\d+)?", str(c))]].apply(pd.to_numeric).values.astype(float)
    names = f["Record Name"].astype(str).str.upper().tolist()
    prot = f["Protein"].astype(str).str.replace(",", ".").astype(float).values
    species = f["Species"].astype(str).values
    pos = {n: i for i, n in enumerate(names)}
    ep = e.groupby("Record Name")["Protein"].first().astype(str).str.replace(",", ".").astype(float)
    keys = {}
    for n in e["Record Name"].astype(str).unique():
        u = n.upper()
        for cand in (u, u.replace("-FT-", "-FTM-")):
            c = [x for x in names if cand == x or cand.endswith("-" + x)]
            if c:
                m = max(c, key=len)
                if abs(ep[n] - prot[pos[m]]) < 0.02:          # identity check on the reference value
                    keys[n] = m
                break
    e = e[e["Record Name"].isin(keys)]
    Xe = e[wl_e_cols].apply(pd.to_numeric).values.astype(float)
    g = e["Record Name"].map(keys).values
    order = np.argsort(wl_e); wl_e, Xe = wl_e[order], Xe[:, order]
    sel = (wl_e >= wl_f.min()) & (wl_e <= wl_f.max())
    wl, Xe = wl_e[sel], Xe[:, sel]
    uniq = sorted(set(g))
    He = np.stack([Xe[g == u].mean(0) for u in uniq])                    # mean of the 3-12 handheld scans
    idx = np.array([pos[u] for u in uniq])
    Fe = np.stack([np.interp(wl, wl_f, x) for x in Xf[idx]])             # benchtop read on the handheld grid
    return dict(wl=wl, foss=Fe, evt5=He, y=prot[idx], species=species[idx], names=np.array(uniq),
                foss_full=Xf, wl_full=wl_f, y_full=prot, species_full=species, names_full=np.array(names))


SUGAR_DEV = {"F750": "F750.CSV", "SCiO": "Scio.CSV", "Tellspec": "Tellspec.CSV", "NIRscan-Nano": "Nirscan_nano.CSV",
             "MicroNIR-1700": ["R1_1700.CSV", "R2_1700.CSV", "R3_1700.CSV"], "MicroNIR-2200": "2200.CSV", "NIRONE-2.2": "Nirone22.CSV"}


def sugarcane():
    def rd(fn):
        df = pd.read_csv(RAW / "sugarcane" / fn, sep=None, engine="python")
        ax = np.array([float(str(c).strip()) for c in df.columns[1:]])
        X = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").values.astype(float)
        s = df.iloc[:, 0].astype(int).values
        o = np.argsort(ax)
        return s, ax[o], X[:, o]
    ref = pd.read_csv(RAW / "sugarcane" / "References.csv", sep=None, engine="python")
    sl, axl, Xl = rd("Labspec.CSV")
    ys = {k: pd.to_numeric(ref.set_index("Sample").loc[sl, k], errors="coerce").values.astype(float) for k in ["TS", "CP", "ADF", "IVOMD"]}
    devs = {}
    for name, fn in SUGAR_DEV.items():
        parts = [rd(f) for f in (fn if isinstance(fn, list) else [fn])]
        s, ax = parts[0][0], parts[0][1]
        X = np.mean([p[2][[list(p[0]).index(v) for v in sl]] for p in parts], axis=0)
        devs[name] = (ax, X, np.stack([np.interp(ax, axl, x) for x in Xl]))   # device grid, device X, LabSpec on that grid
    return sl, ys, devs


def grain_years(sub):
    base = RAW / "nirs4all" / sub
    rd = lambda f: pd.read_csv(base / f, sep=None, engine="python")
    Xtr, Xte = rd("Xtrain.csv"), rd("Xtest.csv")
    wn = np.array([float(c) for c in Xtr.columns]); o = np.argsort(wn)
    return dict(axis=wn[o], Xs=Xtr.values[:, o].astype(float), ys=rd("Ytrain.csv").iloc[:, 0].values.astype(float),
                gs=rd("Mtrain.csv")["Sample_ID"].astype(str).values, Xt=Xte.values[:, o].astype(float),
                yt=rd("Ytest.csv").iloc[:, 0].values.astype(float), gt=rd("Mtest.csv")["Sample_ID"].astype(str).values)


def kennard_stone(X, n):
    """Kennard & Stone (1969) selection on Euclidean distance; returns (selected, remaining) indices."""
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    sel = [int(i), int(j)]
    rest = [k for k in range(len(X)) if k not in sel]
    while len(sel) < n:
        dmin = D[np.ix_(rest, sel)].min(1)
        k = rest[int(np.argmax(dmin))]
        sel.append(k); rest.remove(k)
    return np.array(sel), np.array(rest)


# ============================================================================ families
def conf(name):
    if name in ("idrc_A", "idrc_B"):
        man = name[-1]
        F = Family("conf", f"IDRC-2016 wheat fleet {man} (same samples)", context="fleet")
        if F.done(): return
        w, y, _ = idrc(); ax = np.arange(w[f"{man}1.CAL"].shape[1], dtype=float)
        Xm = w[f"{man}1.CAL"].astype(float)
        for slave in [f"{man}2.CAL", f"{man}3.CAL", f"{man}.Test"]:
            Xsl = w[slave].astype(float)
            for rep in range(10):
                perm = np.random.default_rng(100 + rep).permutation(len(y)); S, A, T = perm[:124], perm[124:186], perm[186:]
                F.add(f"IDRC {man}1->{slave.replace('.CAL', '')}", rep, Xm[S], y[S], Xsl[A], Xsl[T], y[T], ax,
                      std=(Xm[A], Xsl[A]), ya=y[A], extra=dict(pair=f"{man}1->{slave}"))
        F.finish()
    elif name == "idrc_new":
        F = Family("conf", "IDRC-2016 wheat new units + new samples", context="fleet")
        if F.done(): return
        w, y, yv = idrc()
        for master, slave in [("A1.CAL", "vliadA"), ("B1.CAL", "vliadB"), ("C.Test", "vliadC")]:
            Xm, Xsl = w[master].astype(float), w[slave].astype(float); ax = np.arange(Xm.shape[1], dtype=float)
            for rep in range(10):
                perm = np.random.default_rng(200 + rep).permutation(len(yv)); A, T = perm[:75], perm[75:]
                F.add(f"IDRC {master.replace('.CAL', '')}->{slave}", rep, Xm, y, Xsl[A], Xsl[T], yv[T], ax, ya=yv[A])
        F.finish()
    elif name == "icrisat_handheld":
        F = Family("conf", "ICRISAT cereals benchtop -> handheld", context="fleet")
        if F.done(): return
        d = icrisat(); n = len(d["y"])
        for rep in range(10):
            perm = np.random.default_rng(300 + rep).permutation(n); S, A, T = perm[: n // 2], perm[n // 2: 3 * n // 4], perm[3 * n // 4:]
            F.add("ICRISAT DS2500 -> EVT5 protein", rep, d["foss"][S], d["y"][S], d["evt5"][A], d["evt5"][T], d["y"][T], d["wl"],
                  std=(d["foss"][A], d["evt5"][A]), ya=d["y"][A])
        F.finish()
    elif name == "sugarcane":
        F = Family("conf", "Sugarcane lab -> portable spectrometers", context="fleet")
        if F.done(): return
        sl, ys, devs = sugarcane(); n = len(sl)
        for dev, (ax, Xd, Xl) in devs.items():
            for target, y in ys.items():
                ok = np.where(np.isfinite(y))[0]; m = len(ok)
                for rep in range(5):
                    perm = ok[np.random.default_rng(400 + rep).permutation(m)]; S, A, T = perm[: m // 2], perm[m // 2: 3 * m // 4], perm[3 * m // 4:]
                    F.add(f"sugarcane LabSpec->{dev} {target}", rep, Xl[S], y[S], Xd[A], Xd[T], y[T], ax,
                          std=(Xl[A], Xd[A]), ya=y[A], extra=dict(device=dev, target=target))
        F.finish()
    elif name == "grain_year":
        F = Family("conf", "Grain new harvest year (population shift)", context="chem")
        if F.done(): return
        for sub, dev in [("regression_2", "handheld-1 (908-1676 nm)"), ("regression_3", "handheld-2 (933-1659 nm)")]:
            g = grain_years(sub)
            for rep in range(10):
                a, t = halves(g["gt"], np.random.default_rng(500 + rep))
                F.add(f"grain 2022->2023 {dev}", rep, g["Xs"], g["ys"], g["Xt"][a], g["Xt"][t], g["yt"][t], g["axis"],
                      ya=g["yt"][a], groups_s=g["gs"])
        F.finish()
    elif name == "cereal_species":
        F = Family("conf", "Cereals new species (population shift)", context="chem")
        if F.done(): return
        d = icrisat(); sp = d["species_full"]; X, y, ax = d["foss_full"], d["y_full"], d["wl_full"]
        sg = np.array([s.startswith("SG") for s in sp]); pm = np.array([s.startswith("PM") for s in sp])
        for src, tgt, lab in [(sg, pm, "sorghum->pearl millet"), (pm, sg, "pearl millet->sorghum")]:
            S = np.where(src)[0]; Tpool = np.where(tgt)[0]
            for rep in range(5):
                perm = np.random.default_rng(600 + rep).permutation(Tpool); A, T = perm[: len(perm) // 2], perm[len(perm) // 2:]
                F.add(f"cereal {lab}", rep, X[S], y[S], X[A], X[T], y[T], ax, ya=y[A])
        F.finish()
    elif name == "published_idrc":
        F = Family("conf", "Published protocol IDRC-2016 (KS 198-50)", context="fleet")
        if F.done(): return
        w, y, _ = idrc(); ax = np.arange(397, dtype=float)
        for m, s in [("B1.CAL", "B2.CAL"), ("B1.CAL", "B3.CAL"), ("B3.CAL", "B2.CAL")]:
            Xm, Xs_ = w[m].astype(float), w[s].astype(float)
            cal, tst = kennard_stone(Xm, 198)
            F.add(f"IDRC KS {m[:2]}->{s[:2]}", 0, Xm[cal], y[cal], Xs_[cal], Xs_[tst], y[tst], ax, std=(Xm[cal], Xs_[cal]), ya=y[cal])
        F.finish()
    elif name == "published_corn":
        F = Family("conf", "Published protocol corn m5->mp6 (KS 64-16)", context="fleet")
        if F.done(): return
        from spectwin.bench import load
        d = load("corn"); inst = d.domains["instrument"]
        Xm, Xs_ = d.X[inst == "m5"].astype(float), d.X[inst == "mp6"].astype(float)
        cal, tst = kennard_stone(Xm, 64)
        for prop in ["moisture", "oil", "protein", "starch"]:
            y = d.targets[prop][:80].astype(float)
            F.add(f"corn KS m5->mp6 {prop}", 0, Xm[cal], y[cal], Xs_[cal], Xs_[tst], y[tst], d.axis, std=(Xm[cal], Xs_[cal]), ya=y[cal])
        F.finish()
    else:
        raise ValueError(name)


CONF = ["idrc_A", "idrc_B", "idrc_new", "icrisat_handheld", "sugarcane", "grain_year", "cereal_species", "published_idrc", "published_corn"]
