"""FoodSpecBench: harmonised loaders for public food / agrifood spectral datasets.

Every dataset is converted to one schema and saved as data/bench/<name>.npz:

    X        (n, p) float32   spectra (signal type given by `signal`)
    axis     (p,)   float64   native spectral axis
    groups   (n,)   str       sample / replicate group (grouped CV never splits a group)
    split    (n,)   str       official split: 'train' | 'test' | '' (none)
    y__<t>   (n,)   float or str   targets (NaN / '' when missing)
    d__<k>   (n,)   str       domain labels (instrument, season, cultivar, ...)

plus a JSON card (axis_unit, modality, signal, task types, source, citation, licence, notes).
Raw files come from GitHub mirrors (CRAN packages, JchemoData.jl, research repos) and PyPI.
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "bench"


@dataclass
class SpecSet:
    name: str
    X: np.ndarray
    axis: np.ndarray
    axis_unit: str            # 'nm' | 'cm-1' | 'raman_shift'
    modality: str             # 'vnir' | 'mir' | 'raman'
    signal: str               # 'absorbance' | 'deriv1' | 'snv_sg2' | 'snv' | 'raman'
    targets: dict
    target_type: dict         # name -> 'reg' | 'cls'
    groups: np.ndarray
    split: np.ndarray
    domains: dict = field(default_factory=dict)
    card: dict = field(default_factory=dict)

    def save(self, out: Path = OUT):
        out.mkdir(parents=True, exist_ok=True)
        S = lambda v: np.array([str(x) for x in np.asarray(v, dtype=object).ravel()], dtype=str)
        arrays = dict(X=np.asarray(self.X, dtype=np.float32), axis=np.asarray(self.axis, dtype=np.float64),
                      groups=S(self.groups), split=S(self.split))
        for k, v in self.targets.items():
            arrays[f"y__{k}"] = S(v) if self.target_type[k] == "cls" else np.asarray(v, dtype=np.float64)
        for k, v in self.domains.items():
            arrays[f"d__{k}"] = S(v)
        np.savez_compressed(out / f"{self.name}.npz", **arrays)
        card = dict(name=self.name, n=int(self.X.shape[0]), p=int(self.X.shape[1]),
                    axis_unit=self.axis_unit, modality=self.modality, signal=self.signal,
                    axis_range=[float(self.axis.min()), float(self.axis.max())],
                    targets=self.target_type, domains=list(self.domains.keys()),
                    has_official_split=bool((self.split != "").any()), **self.card)
        (out / f"{self.name}.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
        return card


def load(name: str, root: Path = OUT) -> SpecSet:
    z = np.load(root / f"{name}.npz", allow_pickle=False)
    card = json.loads((root / f"{name}.json").read_text())
    targets = {k[3:]: z[k] for k in z.files if k.startswith("y__")}
    domains = {k[3:]: z[k] for k in z.files if k.startswith("d__")}
    return SpecSet(name=name, X=z["X"], axis=z["axis"], axis_unit=card["axis_unit"],
                   modality=card["modality"], signal=card["signal"], targets=targets,
                   target_type=card["targets"], groups=z["groups"], split=z["split"],
                   domains=domains, card=card)


# ----------------------------------------------------------------------------- helpers
def _rdata(path):
    import rdata
    from rdata.conversion import DEFAULT_CLASS_MAP
    cm = dict(DEFAULT_CLASS_MAP)
    cm["data.frame"] = lambda obj, attrs: dict(obj)
    return rdata.conversion.convert(rdata.parser.parse_file(str(path)), cm)


def _axis_from_coords(da, strip="X"):
    vals = da.coords[da.dims[1]].values
    return np.array([float(str(v).lstrip(strip)) for v in vals])


def _clean_cols(X, axis):
    """Drop padded / constant / non-finite columns."""
    ok = np.isfinite(X).all(0) & (np.nanstd(X, 0) > 1e-9)
    return X[:, ok], axis[ok]


def _sorted(X, axis):
    o = np.argsort(axis)
    return X[:, o], axis[o]


def _refl_to_abs(R):
    return -np.log10(np.clip(R, 1e-4, None))


def _nan(n):
    return np.full(n, np.nan)


def _dup_groups(X, prefix):
    _, inv = np.unique(np.round(X, 6), axis=0, return_inverse=True)
    return np.array([f"{prefix}{i}" for i in inv.ravel()])


# ----------------------------------------------------------------------------- loaders
def mango():
    df = pd.read_csv(RAW / "passos/NAnderson2020MendeleyMangoNIRData.csv")
    spec = [c for c in df.columns if re.fullmatch(r"\d+", str(c))]
    X, axis = _clean_cols(df[spec].values.astype(float), np.array(spec, float))
    split = np.where(df["Set"].values == "Val Ext", "test", "train")
    return SpecSet(
        "mango", X, axis, "nm", "vnir", "absorbance",
        {"DM": df["DM"].values.astype(float)}, {"DM": "reg"},
        groups=np.array([f"pop{p}_s{s}" for p, s in zip(df["Pop"], df["Season"])]), split=split,
        domains={"season": df["Season"].astype(str).values, "cultivar": df["Cultivar"].values,
                 "region": df["Region"].values, "temperature": df["Temp"].values,
                 "set": df["Set"].values},
        card=dict(source="Anderson et al. (2020, 2021), Postharvest Biol. Technol.; Mendeley Data 46htwnp833 (via github.com/dario-passos)",
                  licence="CC BY 4.0 (Mendeley Data)", instrument="Felix F750 handheld Vis-NIR (interactance)",
                  notes="Official split: Cal+Tuning (seasons 1-3) -> Val Ext (season 4). Groups = population x season."))


def plum():
    df = pd.read_parquet(RAW / "plumspectra/plumspectra_analysis_ready.parquet")
    if "qc_primary_include" in df:
        keep = df["qc_primary_include"].astype(str).str.lower().isin(["true", "1", "yes"]).values
        df = df[keep].reset_index(drop=True)
    wl_col = [c for c in df.columns if "wavelength" in c.lower()][0]
    ab_col = [c for c in df.columns if "absorb" in c.lower()][0]
    axis = np.asarray(df[wl_col].iloc[0], float)
    X = np.stack([np.asarray(v, float) for v in df[ab_col].values])
    X, axis = _clean_cols(*_sorted(X, axis))
    names = {"soluble_solids_pct": "SSC", "ph": "pH", "fruit_weight_g": "weight",
             "skin_break_force_g_mean": "SRF", "flesh_force_mean_g_mean": "MFF",
             "loading_stiffness_g_per_rawpos_mean": "LS", "adhesive_force_g_mean": "AF",
             "loading_work_g_rawpos_mean": "LW"}
    targets = {v: df[k].values.astype(float) for k, v in names.items() if k in df}
    # Pre-specified physical-plausibility screen for manually recorded traits (release contains
    # transcription errors, e.g. pH = 4352, SSC = 123.8 %, weight = 14132 g). Values are set to NaN, not corrected.
    bounds = {"pH": (2.0, 7.0), "SSC": (3.0, 35.0), "weight": (5.0, 300.0)}
    n_flag = {}
    for t, (lo, hi) in bounds.items():
        bad = ~((targets[t] >= lo) & (targets[t] <= hi))
        n_flag[t] = int(bad.sum())
        targets[t][bad] = np.nan
    return SpecSet(
        "plum", X, axis, "nm", "vnir", "absorbance", targets, {k: "reg" for k in targets},
        groups=df["sample_id"].astype(str).values, split=np.array([""] * len(df)),
        domains={"cultivar": df["cultivar_code"].astype(str).values,
                 "batch": df["batch_id"].astype(str).values,
                 "fold": df["outer_fold"].astype(str).values},
        card=dict(source="PlumSPECTRA-data (github.com/707728642li/PlumSPECTRA-data, commit cf4959d, 2026-08-20)",
                  licence="Public; reuse terms pending institutional confirmation (per data card)",
                  instrument="NIR 901-1701 nm (228 bands)",
                  notes=f"Primary QC cohort. Frozen 5 outer folds used as the CV outer loop. Implausible trait values set to NaN: {n_flag}."))


def ch2021():
    from .io_jld2 import read_jld2
    d = read_jld2(str(RAW / "jchemodata/challenge2021.jld2"))["dat"]
    def xa(df):
        return df.values.astype(float), np.array([float(c) for c in df.columns])
    Xc, axis = xa(d["Xtrain"]); Xs, _ = xa(d["Xstd"]); Xt, _ = xa(d["Xtest"])
    X = _refl_to_abs(np.vstack([Xc, Xs, Xt]))   # published as relative reflectance/intensity
    y = np.concatenate([d["Ytrain"]["y"].values.astype(float), _nan(len(Xs)), d["Ytest"]["y"].values.astype(float)])
    inst = np.concatenate([np.full(len(Xc), "1"), d["Ystd"]["inst"].astype(int).astype(str).values,
                           d["Ytest"]["inst"].astype(int).astype(str).values])
    role = np.array(["cal"] * len(Xc) + ["std"] * len(Xs) + ["val"] * len(Xt))
    ids = np.concatenate([d["Ytrain"]["id"].astype(str).values, d["Ystd"]["id"].astype(str).values,
                          d["Ytest"]["id"].astype(str).values])
    split = np.where(role == "cal", "train", np.where((role == "val") & (inst == "1"), "test", ""))
    return SpecSet(
        "ch2021", X, axis, "nm", "vnir", "absorbance", {"y": y}, {"y": "reg"},
        groups=ids, split=split, domains={"instrument": inst, "role": role},
        card=dict(source="Chemometrics 2021 challenge (chemom2021.sciencesconf.org), via JchemoData.jl",
                  licence="Challenge data, redistributed in JchemoData.jl", instrument="7 Vis-NIR units of one brand (300-1100 nm)",
                  notes="Fruit; property unspecified by organisers. Relative reflectance converted to log10(1/R). E1 = cal (unit 1) -> val unit 1. Units 2-7 used for transfer (E3). 245 transfer standards on all units."))


def ch2018():
    from .io_jld2 import read_jld2
    d = read_jld2(str(RAW / "jchemodata/challenge2018.jld2"))["dat"]
    X = _refl_to_abs(d["X"].values.astype(float)); axis = np.array([float(c) for c in d["X"].columns])
    Y = d["Y"]
    split = np.where(Y["test"].astype(int).values == 1, "test", "train")
    return SpecSet(
        "ch2018", X, axis, "nm", "vnir", "absorbance",
        {"protein": Y["conc"].values.astype(float), "product": Y["typ"].astype(str).values},
        {"protein": "reg", "product": "cls"},
        groups=np.array([f"s{i}" for i in range(len(X))]), split=split,
        domains={"product": Y["typ"].astype(str).values, "label": Y["label"].astype(str).values},
        card=dict(source="Chemometrics 2018 challenge (Paris), corrected by Lesnoff (JchemoData.jl)",
                  licence="Challenge data, redistributed in JchemoData.jl", instrument="NIR 1120-2478 nm",
                  notes="Reflectance converted to log10(1/R). Protein in 10 food/feed products (wheat, milk powder & whey, soya meal, sunflower seed, silage...)."))


def corn():
    from .io_jld2 import read_jld2
    d = read_jld2(str(RAW / "jchemodata/corn.jld2"))["dat"]
    axis = np.array([float(c) for c in d["Xm5"].columns])
    Xs, inst, grp = [], [], []
    for k in ["m5", "mp5", "mp6"]:
        Xs.append(d[f"X{k}"].values.astype(float)); inst += [k] * 80; grp += [f"c{i}" for i in range(80)]
    X = np.vstack(Xs); Y = d["Y"]
    targets = {c: np.tile(Y[c].values.astype(float), 3) for c in ["moisture", "oil", "protein", "starch"]}
    nbs = {k: d[f"X{k}nbs"].values.astype(float) for k in ["m5", "mp5", "mp6"]}
    np.savez_compressed(OUT / "corn_nbs.npz", **nbs)
    return SpecSet(
        "corn", X, axis, "nm", "vnir", "absorbance", targets, {c: "reg" for c in targets},
        groups=np.array(grp), split=np.array([""] * len(X)), domains={"instrument": np.array(inst)},
        card=dict(source="Eigenvector corn data (Cargill), via JchemoData.jl", licence="Public demonstration data",
                  instrument="3 NIR spectrometers m5, mp5, mp6 (1100-2498 nm)",
                  notes="Same 80 samples on 3 instruments; NBS glass standards in corn_nbs.npz."))


def wheat():
    from .io_jld2 import read_jld2
    d = read_jld2(str(RAW / "jchemodata/wheatkernels_cui.jld2"))["dat"]
    Xa = d["Xtrain"].values.astype(float); Xb = d["Xtest"].values.astype(float)
    axis = np.array([float(c) for c in d["Xtrain"].columns])
    X = np.vstack([Xa, Xb]); y = np.concatenate([d["Ytrain"]["prot"].values, d["Ytest"]["prot"].values]).astype(float)
    split = np.array(["train"] * len(Xa) + ["test"] * len(Xb))
    return SpecSet(
        "wheat", X, axis, "nm", "vnir", "absorbance", {"protein": y}, {"protein": "reg"},
        groups=np.array([f"k{i}" for i in range(len(X))]), split=split, domains={"time": np.where(split == "test", "t+2months", "t0")},
        card=dict(source="Pedersen et al. (2002); Cui & Fearn (2018) dataset 3 (ucphchemometrics.com), via JchemoData.jl",
                  licence="Public research data", instrument="NIT single-kernel 850-1048 nm",
                  notes="Test kernels stored ~2 months longer (temporal drift)."))


def cassava():
    d = _rdata(RAW / "cran_rchemo/cassav.rda")["cassav"]
    Xa = np.asarray(d["Xtrain"], float); Xb = np.asarray(d["Xtest"], float)
    axis = _axis_from_coords(d["Xtrain"])
    X = np.vstack([Xa, Xb]); y = np.concatenate([d["ytrain"], d["ytest"]]).astype(float)
    year = np.concatenate([np.asarray(d["year"]).astype(str), np.array(["test"] * len(Xb))])
    split = np.array(["train"] * len(Xa) + ["test"] * len(Xb))
    return SpecSet(
        "cassava", X, axis, "nm", "vnir", "absorbance", {"TBC": y}, {"TBC": "reg"},
        groups=np.array([f"r{i}" for i in range(len(X))]), split=split, domains={"year": year},
        card=dict(source="Davrieux et al. (2016); Lesnoff et al. (2020); CRAN rchemo", licence="GPL (R package data)",
                  instrument="FOSS NIRSystems 400-2498 nm", notes="Beta-carotene in cassava roots; breeding years 2009-2013."))


def sorghum():
    from .io_jld2 import read_jld2
    d = read_jld2(str(RAW / "jchemodata/sorghum.jld2"))["dat"]
    X = d["X"].values.astype(float); axis = np.array([float(c) for c in d["X"].columns]); Y = d["Y"]
    rng = np.random.default_rng(0); te = rng.random(len(X)) < 0.25
    targets = {c.upper(): Y[c].values.astype(float) for c in ["cp", "ndf", "adf", "dm"]}
    return SpecSet(
        "sorghum", X, axis, "nm", "vnir", "deriv1", targets, {k: "reg" for k in targets},
        groups=np.array([f"g{i}" for i in range(len(X))]), split=np.where(te, "test", "train"),
        card=dict(source="JchemoData.jl (sorghum)", licence="Redistributed in JchemoData.jl",
                  instrument="NIR 1100-2498 nm", notes="Published as first-derivative spectra (r = 0.90 with d/dlambda of log(1/R)). No official split: fixed random 75/25 holdout (seed 0)."))


def forages():
    from .io_jld2 import read_jld2
    d = read_jld2(str(RAW / "jchemodata/forages2.jld2"))["dat"]
    X = d["X"].values.astype(float); axis = np.array([float(c) for c in d["X"].columns]); Y = d["Y"]
    split = np.where(Y["test"].astype(int).values == 1, "test", "train")
    return SpecSet(
        "forages", X, axis, "nm", "vnir", "snv_sg2",
        {"DM": Y["dm"].values.astype(float), "NDF": Y["ndf"].values.astype(float)}, {"DM": "reg", "NDF": "reg"},
        groups=np.array([f"f{i}" for i in range(len(X))]), split=split, domains={"type": Y["typ"].astype(str).values},
        card=dict(source="CIRAD UMR Selmet, via JchemoData.jl (forages2)", licence="Released preprocessed only (SNV + SG 2nd derivative)",
                  instrument="FOSS NIRSystems 1100-2498 nm", notes="Published only after SNV + Savitzky-Golay 2nd derivative."))


def pears():
    X = pd.read_csv(RAW / "passos/spectra_pears.csv", header=None).values.astype(float)
    y = pd.read_csv(RAW / "passos/brix_pears.csv", header=None).values.ravel().astype(float)
    axis = np.loadtxt(RAW / "passos/lambda.txt")
    X = _refl_to_abs(X)
    keep = axis >= 450   # silicon detector very noisy below ~450 nm
    rng = np.random.default_rng(1); te = rng.random(len(X)) < 0.25
    return SpecSet(
        "pears", X[:, keep], axis[keep], "nm", "vnir", "absorbance", {"Brix": y}, {"Brix": "reg"},
        groups=np.array([f"p{i}" for i in range(len(X))]), split=np.where(te, "test", "train"),
        card=dict(source="Kaneko & Funatsu (2015) SVR example data (github.com/dario-passos)", licence="Research example data",
                  instrument="Vis-NIR 433-1147 nm", notes="Reflectance converted to log10(1/R); <450 nm dropped. Fixed random 75/25 holdout (seed 1)."))


def grapes():
    from .io_jld2 import read_jld2
    d = read_jld2(str(RAW / "jchemodata/grapes.jld2"))["dat"]
    I = d["X"].values.astype(float); axis = np.array([float(c) for c in d["X"].columns]); Y = d["Y"]
    split = np.where(Y["test"].astype(int).values == 1, "test", "train")
    keep = (axis >= 400) & (axis <= 1100)                     # counts < ~10 below 400 nm (noise)
    Iref = I[split == "train"].max(0)                          # per-wavelength reference = brightest training spectrum
    X = -np.log10(np.clip(I, 1.0, None) / Iref)[:, keep]; axis = axis[keep]
    return SpecSet(
        "grapes", X, axis, "nm", "vnir", "absorbance", {"variety": Y["lab"].astype(str).values}, {"variety": "cls"},
        groups=np.array([f"b{i}" for i in range(len(X))]), split=split, domains={"date": Y["date"].astype(str).values},
        card=dict(source="Roger et al. (2005), Chemom. Intell. Lab. Syst.; via JchemoData.jl", licence="Research data",
                  instrument="Zeiss MMS1 transmission 303-1147 nm", notes="Published as raw detector counts; converted to pseudo-absorbance -log10(I / I_ref), I_ref = per-wavelength maximum of training spectra; 400-1100 nm kept. Official split by acquisition batch."))


def tecator():
    d = _rdata(RAW / "cran_caret/tecator.RData")
    X = np.asarray(d["absorp"], float); Y = np.asarray(d["endpoints"], float)
    axis = np.linspace(850, 1050, 100)
    split = np.array(["train"] * 172 + ["test"] * 43)
    return SpecSet(
        "tecator", X, axis, "nm", "vnir", "absorbance",
        {"moisture": Y[:, 0], "fat": Y[:, 1], "protein": Y[:, 2]}, {"moisture": "reg", "fat": "reg", "protein": "reg"},
        groups=_dup_groups(X, "t"), split=split,
        card=dict(source="Tecator / StatLib (Borggaard & Thodberg 1992), CRAN caret", licence="Public (StatLib)",
                  instrument="Tecator Infratec NIT 850-1050 nm", notes="Conventional split: first 172 train, last 43 test; duplicate spectra grouped."))


def cookie():
    d = _rdata(RAW / "cran_ppls/cookie.rda")["cookie"]
    X = np.column_stack([np.asarray(v, float) for v in d["NIR"].values()]) if isinstance(d["NIR"], dict) else np.asarray(d["NIR"], float)
    C = d["constituents"]
    targets = {k: np.asarray(C[k], float) for k in ["fat", "sucrose", "dry_flour", "water"]}
    axis = np.linspace(1100, 2498, X.shape[1])
    return SpecSet(
        "cookie", X, axis, "nm", "vnir", "absorbance", targets, {k: "reg" for k in targets},
        groups=np.array([f"d{i}" for i in range(len(X))]), split=np.array(["train"] * 40 + ["test"] * 32),
        card=dict(source="Osborne et al. (1984); Brown, Fearn & Vannucci (2001); CRAN ppls", licence="GPL (R package data)",
                  instrument="NIR reflectance 1100-2498 nm", notes="Designed biscuit-dough recipes (near closure). Official 40/32 split."))


def bioethanol():
    d = _rdata(RAW / "cran_chemometrics/NIR.rda")["NIR"]
    Xdf = d["xNIR"]; X = np.column_stack([np.asarray(Xdf[c], float) for c in Xdf]) if isinstance(Xdf, dict) else Xdf.values
    cols = list(Xdf.keys()) if isinstance(Xdf, dict) else list(Xdf.columns)
    axis = np.array([float(str(c).lstrip("X")) for c in cols])
    Y = d["yGlcEtOH"]
    get = (lambda k: np.asarray(Y[k], float)) if isinstance(Y, dict) else (lambda k: Y[k].values.astype(float))
    return SpecSet(
        "bioethanol", X, axis, "nm", "vnir", "deriv1", {"glucose": get("Glucose"), "ethanol": get("Ethanol")},
        {"glucose": "reg", "ethanol": "reg"}, groups=np.array([f"m{i}" for i in range(len(X))]), split=np.array([""] * len(X)),
        card=dict(source="Liebmann et al. (2009) Anal. Chim. Acta; CRAN chemometrics", licence="GPL (R package data)",
                  instrument="NIR transmission 1115-2285 nm", notes="Published as first-derivative spectra; rye/wheat/corn mashes."))


def ligno():
    import glob, zipfile
    w = glob.glob("/tmp/claude-0/-home-claude/*/scratchpad/probe/wheels/chemotools-*.whl") + glob.glob(str(RAW / "pypi/chemotools-*.whl"))
    z = zipfile.ZipFile(w[0])
    rd = lambda n: pd.read_csv(z.open("chemotools/datasets/data/" + n))
    Xa = rd("train_spectra.csv"); ya = rd("train_hplc.csv")["glucose"].values.astype(float)
    Xb = rd("fermentation_spectra.csv"); hplc = rd("fermentation_hplc.csv")
    axis = np.array([float(c) for c in Xa.columns])
    X = np.vstack([Xa.values, Xb.values]).astype(float)
    keep = (axis >= 900) & (axis <= 1700)      # outside: ATR cut-off noise / zero padding
    X, axis = X[:, keep], axis[keep]
    t_on = np.arange(len(Xb)) * (hplc["time"].max() / (len(Xb) - 1))  # spectra every ~1.25 min over the run
    y = np.concatenate([ya, _nan(len(Xb))])
    targets = {"glucose": y}
    for c in ["xylose", "acetic_acid", "ethanol", "furfural"]:
        targets[c] = _nan(len(X))
    for _, r in hplc.iterrows():   # attach HPLC values to the nearest on-line spectrum
        j = len(Xa) + int(np.argmin(np.abs(t_on - r["time"])))
        for c in ["glucose", "xylose", "acetic_acid", "ethanol", "furfural"]:
            targets[c][j] = r[c]
    time = np.concatenate([_nan(len(Xa)), t_on])
    role = np.array(["offline"] * len(Xa) + ["online"] * len(Xb))
    np.save(OUT / "ligno_time.npy", time)
    return SpecSet(
        "ligno", X, axis, "cm-1", "mir", "absorbance", targets, {k: "reg" for k in targets},
        groups=np.array([f"o{i}" for i in range(len(X))]), split=np.where(role == "offline", "train", "test"),
        domains={"role": role},
        card=dict(source="DTU PROSYS 2019 lignocellulosic ethanol fermentation (chemotools / dtuprosys, PyPI)", licence="MIT (package data)",
                  instrument="ATR-MIR flow cell, 428-1833 cm-1 (900-1700 cm-1 kept)", notes="Train: semi-synthetic off-line standards (glucose). Test: on-line spectra, HPLC at 34 times (nearest spectrum). Spectrum timing inferred as uniform over the run."))


def mayonnaise():
    d = _rdata(RAW / "cran_pls/mayonnaise.RData")["mayonnaise"]
    X = np.asarray(d["NIR"], float); oil = np.asarray(d["oil.type"]).astype(int).astype(str)
    tr = np.asarray(d["train"]).astype(bool)
    axis = np.linspace(1100, 2500, X.shape[1])
    groups = np.array([f"s{i // 3}" for i in range(len(X))])
    return SpecSet(
        "mayonnaise", X, axis, "nm", "vnir", "absorbance", {"oil": oil}, {"oil": "cls"},
        groups=groups, split=np.where(tr, "train", "test"), domains={"oil": oil},
        card=dict(source="Indahl et al. (1999) Chemom. Intell. Lab. Syst.; CRAN pls", licence="GPL (R package data)",
                  instrument="NIR 1100-2500 nm", notes="54 samples x triplicate scans; 6 oil types; official train flag."))


def oregano():
    from .io_jld2 import read_jld2
    d = read_jld2(str(RAW / "jchemodata/oregano_occ.jld2"))["dat"]
    Xa = d["Xtrain"].values.astype(float); Xb = d["Xtest"].values.astype(float)
    wn = np.array([float(c) for c in d["Xtrain"].columns])
    X = np.vstack([Xa, Xb]); cls = np.concatenate([d["Ytrain"]["cla"].astype(str).values, d["Ytest"]["cla"].astype(str).values])
    axis_nm = 1e7 / wn
    X, axis_nm = _sorted(X, axis_nm)
    return SpecSet(
        "oregano", X, axis_nm, "nm", "vnir", "snv", {"class": cls}, {"class": "cls"},
        groups=np.array([f"h{i}" for i in range(len(X))]), split=np.array(["train"] * len(Xa) + ["test"] * len(Xb)),
        card=dict(source="Rodionova & Pomerantsev (2020); Kucheryavskiy et al. (2024); via JchemoData.jl", licence="Research data (mda.tools)",
                  instrument="Nicolet 6700 FT-NIR 9000-4000 cm-1 (axis converted to nm)", notes="SNV-preprocessed. Train: oregano only. Test: oregano, non-target herbs, suspicious."))


def fishoil():
    d = _rdata(RAW / "cran_EMSC/fishoil.rda")["fishoil"]
    X = np.asarray(d["Raman"], float); axis = np.asarray(d["Raman"].coords[d["Raman"].dims[1]].values, float)
    keep = (axis >= 300) & (axis <= 3200)
    return SpecSet(
        "fishoil", X[:, keep], axis[keep], "raman_shift", "raman", "raman", {"iodine": np.asarray(d["Iodine"], float)}, {"iodine": "reg"},
        groups=np.array([f"o{g}" for g in np.asarray(d["replicates"])]), split=np.array([""] * len(X)),
        card=dict(source="Afseth et al.; CRAN EMSC (Liland)", licence="GPL (R package data)", instrument="Raman",
                  notes="42 oils x 3 replicates; strong fluorescence background. Range 300-3200 cm-1 kept."))


def emulsions():
    d = _rdata(RAW / "cran_multiblock/emulsions.rda")["emulsions"]
    P = np.asarray(d["PUFA"], float)
    out = []
    for mod, key, unit, sig in [("vnir", "NIR", "nm", "absorbance"), ("raman", "Raman", "raman_shift", "raman")]:
        X = np.asarray(d[key], float); axis = np.asarray(d[key].coords[d[key].dims[1]].values, float)
        out.append(SpecSet(
            f"emulsions_{key.lower()}", X, axis, unit, mod, sig, {"PUFAtot": P[:, 0], "PUFAfat": P[:, 1]},
            {"PUFAtot": "reg", "PUFAfat": "reg"}, groups=np.array([f"e{i}" for i in range(len(X))]), split=np.array([""] * len(X)),
            card=dict(source="Afseth et al. (2005); CRAN multiblock (Liland)", licence="GPL (R package data)",
                      instrument=f"{key} on the same 69 fat/water/protein emulsions", notes="Same samples measured by NIR and Raman.")))
    return out


def carbs():
    d = _rdata(RAW / "cran_mdatools/carbs.RData")["carbs"]
    D = np.asarray(d["D"], float); S = np.asarray(d["S"], float); C = np.asarray(d["C"], float)
    axis = np.asarray(d["D"].coords[d["D"].dims[1]].values, float)
    X = np.vstack([D, S.T])
    names = ["fructose", "lactose", "ribose"]
    Cfull = np.vstack([C, np.eye(3) * C.max(0)])
    targets = {n: Cfull[:, i] for i, n in enumerate(names)}
    role = np.array(["mixture"] * len(D) + ["pure"] * 3)
    X, axis = _sorted(X, axis)
    return SpecSet(
        "carbs", X, axis, "raman_shift", "raman", "raman", targets, {n: "reg" for n in names},
        groups=np.array([f"c{i}" for i in range(len(X))]), split=np.array([""] * len(X)), domains={"role": role},
        card=dict(source="CRAN mdatools (Kucheryavskiy)", licence="MIT (package data)", instrument="Raman",
                  notes="21 mixtures with known concentrations + 3 pure-component spectra (identifiability ground truth)."))


def potato():
    d = _rdata(RAW / "cran_multiblock/potato.rda")["potato"]
    X = np.asarray(d["NIRraw"], float); axis = np.asarray(d["NIRraw"].coords[d["NIRraw"].dims[1]].values, float)
    ch = d["Chemical"]; co = d["Compression"]
    chn = [str(v) for v in ch.coords[ch.dims[1]].values]; con = [str(v) for v in co.coords[co.dims[1]].values]
    targets = {"starch": np.asarray(ch, float)[:, chn.index("Sta.")], "PEU": np.asarray(ch, float)[:, chn.index("PEU")],
               "modulus20": np.asarray(co, float)[:, con.index("Mo20")], "stress20": np.asarray(co, float)[:, con.index("ST20")]}
    return SpecSet(
        "potato", X, axis, "nm", "vnir", "snv", targets, {k: "reg" for k in targets},
        groups=np.array([f"t{i}" for i in range(len(X))]), split=np.array([""] * len(X)),
        card=dict(source="Thybo et al.; models.life.ku.dk Texture_Potatoes; CRAN multiblock", licence="GPL (R package data)",
                  instrument="Vis-NIR 400-2498 nm (raw tubers)", notes="n=26: mechanism / case analysis only (LOOCV)."))


def cereal():
    d = _rdata(RAW / "cran_chemometrics/cereal.rda")["cereal"]
    X = np.asarray(d["X"], float); axis = _axis_from_coords(d["X"]); Y = np.asarray(d["Y"], float)
    names = [str(v) for v in d["Y"].coords[d["Y"].dims[1]].values]
    targets = {re.sub(r"\W", "", n): Y[:, i] for i, n in enumerate(names)}
    return SpecSet(
        "cereal", X, axis, "nm", "vnir", "deriv1", targets, {k: "reg" for k in targets},
        groups=np.array([f"c{i}" for i in range(len(X))]), split=np.array([""] * len(X)),
        card=dict(source="Varmuza & Filzmoser (2009); CRAN chemometrics", licence="GPL (R package data)",
                  instrument="NIR 1126-2278 nm", notes="n=15: small-n check only (LOOCV)."))


def tablets():
    import scipy.io as sio
    s = sio.loadmat(RAW / "passos/nir_shootout_2002.mat", squeeze_me=True, struct_as_record=False)
    cands = [np.asarray(a, float).ravel() for a in np.asarray(s["calibrate_1"].axisscale, dtype=object).ravel()]
    axis = next(a for a in cands if a.size == 650)
    Xs, inst, split, ys, grp = [], [], [], [], []
    for part, sp in [("calibrate", "train"), ("validate", "train"), ("test", "test")]:
        Y = np.asarray(s[f"{part}_Y"].data, float)
        for k in ["1", "2"]:
            D = np.asarray(s[f"{part}_{k}"].data, float)
            Xs.append(D); inst += [k] * len(D); split += [sp] * len(D); ys.append(Y)
            grp += [f"{part}{i}" for i in range(len(D))]
    X = np.vstack(Xs); Y = np.vstack(ys)
    return SpecSet(
        "tablets", X, axis, "nm", "vnir", "absorbance", {"assay": Y[:, 2]}, {"assay": "reg"},
        groups=np.array(grp), split=np.array(split), domains={"instrument": np.array(inst)},
        card=dict(source="IDRC 2002 shootout (pharmaceutical tablets), via github.com/dario-passos", licence="IDRC shootout data",
                  instrument="2 NIR transmission instruments 600-1898 nm", notes="NON-FOOD control for instrument transfer only."))


def minerals():
    df = pd.read_csv(RAW / "nirpy/Raman_spectra_mixtures.csv")
    fr = df[["Albite", "Quartz", "Orthoclase"]].values.astype(float)
    spec = [c for c in df.columns if c not in ("Unnamed: 0", "Albite", "Quartz", "Orthoclase")]
    X = df[spec].values.astype(float); axis = np.array(spec, float)
    keep = (axis >= 150) & (axis <= 1300)
    return SpecSet(
        "minerals", X[:, keep], axis[keep], "raman_shift", "raman", "raman",
        {"albite": fr[:, 0], "quartz": fr[:, 1], "orthoclase": fr[:, 2]}, {"albite": "reg", "quartz": "reg", "orthoclase": "reg"},
        groups=np.array([f"m{i}" for i in range(len(X))]), split=np.array([""] * len(X)),
        card=dict(source="nirpyresearch example data (github.com/nevernervous78/nirpyresearch)", licence="Research example data",
                  instrument="Raman", notes="NON-FOOD control: 3-mineral mixtures with known fractions (closure) for identifiability."))


LOADERS = [mango, plum, ch2021, ch2018, corn, wheat, cassava, sorghum, forages, pears, grapes, tecator, cookie,
           bioethanol, ligno, mayonnaise, oregano, fishoil, emulsions, carbs, potato, cereal, tablets, minerals]


def build_all(verbose=True):
    OUT.mkdir(parents=True, exist_ok=True)
    cards = []
    for fn in LOADERS:
        res = fn()
        for s in (res if isinstance(res, list) else [res]):
            assert np.isfinite(s.X).all(), f"{s.name}: non-finite X"
            assert len(s.groups) == len(s.X) == len(s.split)
            for k, v in s.targets.items():
                assert len(v) == len(s.X), (s.name, k)
            c = s.save()
            cards.append(c)
            if verbose:
                tg = ", ".join(f"{k}({t})" for k, t in c["targets"].items())
                print(f"{c['name']:16s} n={c['n']:6d} p={c['p']:5d} {c['axis_unit']:11s} {c['axis_range'][0]:8.1f}-{c['axis_range'][1]:8.1f} "
                      f"{c['signal']:10s} split={'yes' if c['has_official_split'] else 'no ':3s} | {tg}")
    (OUT / "registry.json").write_text(json.dumps(cards, indent=2, ensure_ascii=False))
    return cards


if __name__ == "__main__":
    build_all()
