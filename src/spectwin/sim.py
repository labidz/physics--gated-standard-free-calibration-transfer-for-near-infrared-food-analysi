"""SimFood-RT: physics-based simulation of food NIR spectra on virtual instrument fleets.

Absorption  : water and ethanol from published optical constants (Segelstein 1981; Kedenburg et al.
              2012; Wang et al.; Sani & Dell'Oro 2016 via refractiveindex.info, CC0), mu_a = 4 pi k / lambda.
              Lipid, protein and carbohydrate are SYNTHETIC band models (Gaussian bands at literature
              NIR band positions; magnitudes approximate) - adequate because SimFood-RT is used for
              mechanism / identifiability checks where the ground truth is whatever is simulated.
Scattering  : oil-in-water emulsions - Mie theory (miepython) for log-normal droplet size distributions;
              powders - power-law reduced scattering mu_s' = a (lambda / 1000 nm)^-b, g = 0.8;
              clear broths - no scattering (Beer-Lambert transmission).
Transport   : adding-doubling (iadpython) total diffuse reflectance of a 10 mm slab.
Instruments : 4 classes x 3 units with unit-specific axis shift, warp, Gaussian line width,
              stray light, smooth photometric offset and noise.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
OPT = ROOT / "data" / "raw" / "optical"
SIM = ROOT / "data" / "sim"

WL = np.arange(400.0, 2501.0, 4.0)          # simulation grid (nm)


def _nk(fname):
    d = yaml.safe_load((OPT / fname).read_text())
    rows = []
    for blk in d["DATA"]:
        if "tabulated" in blk["type"]:
            arr = np.array([[float(v) for v in r.split()] for r in blk["data"].strip().splitlines()])
            rows.append(arr)
    a = rows[0]
    wl_nm = a[:, 0] * 1000.0
    k = a[:, -1]
    return wl_nm, k


def mua_from_k(fname, wl=WL):
    w, k = _nk(fname)
    o = np.argsort(w)
    kk = np.interp(wl, w[o], k[o])
    return 4 * np.pi * kk / (wl * 1e-6)   # per mm (lambda in mm)


def _bands(wl, spec):
    out = np.zeros_like(wl)
    for c, fwhm, amp in spec:
        s = fwhm / 2.355
        out += amp * np.exp(-0.5 * ((wl - c) / s) ** 2)
    return out


# Synthetic absorption coefficient spectra (per mm) of pure constituents: (centre nm, FWHM nm, peak).
LIPID = [(930, 40, 0.012), (1040, 40, 0.004), (1210, 45, 0.11), (1395, 40, 0.09), (1725, 35, 0.55),
         (1765, 35, 0.45), (2140, 40, 0.30), (2310, 35, 1.20), (2350, 35, 1.00)]
PROTEIN = [(1020, 50, 0.006), (1180, 50, 0.02), (1510, 50, 0.20), (1690, 45, 0.15), (1980, 60, 0.30),
           (2050, 55, 0.60), (2180, 50, 0.70), (2300, 45, 0.55)]
CARB = [(1200, 50, 0.03), (1440, 60, 0.25), (1580, 60, 0.15), (1940, 70, 0.30), (2080, 60, 0.65),
        (2270, 50, 0.45), (2320, 45, 0.40)]


def constituent_mua(wl=WL):
    water = mua_from_k("main_H2O_nk_Segelstein.yml", wl)
    eth = mua_from_k("organic_C2H6O_-_ethanol_nk_Sani.yml", wl)
    return dict(water=water, lipid=_bands(wl, LIPID), protein=_bands(wl, PROTEIN), carbohydrate=_bands(wl, CARB), ethanol=eth)


# ----------------------------------------------------------------------------- scattering
_MIE = None


def mie_table(m=1.46 / 1.33):
    global _MIE
    if _MIE is None:
        import miepython
        x = np.logspace(-2, np.log10(400), 1500)
        try:
            qext, qsca, qback, g = miepython.efficiencies_mx(m, x)
        except AttributeError:
            qext, qsca, qback, g = miepython.mie(m, x)
        _MIE = (x, np.asarray(qsca), np.asarray(g))
    return _MIE


def emulsion_scattering(wl, phi, d50_um, sg=1.6, n_med=1.33):
    """Volume-weighted Mie scattering (per mm) and anisotropy for log-normal droplets."""
    x_t, q_t, g_t = mie_table()
    d = np.exp(np.linspace(np.log(d50_um) - 3 * np.log(sg), np.log(d50_um) + 3 * np.log(sg), 41))
    pdf = np.exp(-0.5 * ((np.log(d) - np.log(d50_um)) / np.log(sg)) ** 2)
    vol = pdf * d ** 3
    vol /= vol.sum()
    mus = np.zeros_like(wl)
    gnum = np.zeros_like(wl)
    for di, vi in zip(d, vol):
        x = np.pi * di * n_med / (wl * 1e-3)
        q = np.interp(x, x_t, q_t)
        g = np.interp(x, x_t, g_t)
        s = 1.5 * phi * vi * q / (di * 1e-3)       # per mm
        mus += s
        gnum += s * g
    return mus, gnum / np.maximum(mus, 1e-12)


def powder_scattering(wl, a, b, g=0.8):
    musp = a * (wl / 1000.0) ** (-b)
    return musp / (1 - g), np.full_like(wl, g)


def diffuse_reflectance(mua, mus, g, d_mm=10.0, n=1.40):
    import iadpython
    mut = mua + mus
    a = np.where(mut > 0, mus / np.maximum(mut, 1e-12), 0.0)
    b = mut * d_mm
    s = iadpython.Sample(a=a, b=b, g=g, d=d_mm, n=n, n_above=1.0, n_below=1.0)
    UR1, UT1, URU, UTU = s.rt()
    return np.asarray(UR1)


# ----------------------------------------------------------------------------- instruments
CLASSES = {
    "sivnir": dict(lo=400.0, hi=1100.0, step=3.0, fwhm=10.0, noise=5e-4, shift=1.0, warp=0.5, stray=(1e-4, 1e-3), unit="nm"),
    "dispnir": dict(lo=1100.0, hi=2498.0, step=2.0, fwhm=10.0, noise=2e-4, shift=0.8, warp=0.5, stray=(1e-4, 5e-4), unit="nm"),
    "ftnir": dict(lo=1000.0, hi=2500.0, step=None, fwhm=None, noise=1e-4, shift=0.2, warp=0.1, stray=(5e-5, 2e-4), unit="cm-1"),
    "mems": dict(lo=1350.0, hi=2500.0, step=8.0, fwhm=16.0, noise=2e-3, shift=3.0, warp=1.5, stray=(5e-4, 3e-3), unit="nm"),
}


def make_unit(cls, k, rng):
    c = CLASSES[cls]
    return dict(cls=cls, unit=k, shift=float(rng.uniform(-c["shift"], c["shift"])), warp=float(rng.uniform(-c["warp"], c["warp"])),
                fwhm_scale=float(rng.uniform(0.85, 1.2)), stray=float(rng.uniform(*c["stray"])),
                off_coef=rng.normal(0, 0.01, 3).tolist(), noise=c["noise"])


def measure(A_fine, unit, rng):
    """Apparent absorbance on the simulation grid -> one unit's measured spectra (n, p_unit)."""
    c = CLASSES[unit["cls"]]
    if c["unit"] == "cm-1":   # FT-NIR: uniform wavenumber grid, 8 cm-1 resolution
        nu = np.arange(4000.0, 10000.0 + 1e-9, 4.0)
        axis_nm = 1e7 / nu
        fwhm_nm = (axis_nm ** 2) * 8.0 * unit["fwhm_scale"] / 1e7
        axis_out = nu
    else:
        axis_nm = np.arange(c["lo"], c["hi"] + 1e-9, c["step"])
        fwhm_nm = np.full_like(axis_nm, c["fwhm"] * unit["fwhm_scale"])
        axis_out = axis_nm
    mid = 0.5 * (axis_nm.min() + axis_nm.max())
    half = 0.5 * (axis_nm.max() - axis_nm.min())
    true_nm = axis_nm + unit["shift"] + unit["warp"] * (axis_nm - mid) / half     # wavelength actually seen by pixel
    # Gaussian line shape: weighted average of the fine spectrum around each true wavelength
    W = np.exp(-0.5 * ((WL[None, :] - true_nm[:, None]) / (fwhm_nm[:, None] / 2.355)) ** 2)
    W /= W.sum(1, keepdims=True)
    T = 10.0 ** (-A_fine) @ W.T                       # convolve in transmittance/reflectance space
    T = (T + unit["stray"]) / (1 + unit["stray"])
    A = -np.log10(np.clip(T, 1e-6, None))
    t = (axis_nm - mid) / half
    A = A + np.polyval(unit["off_coef"], t)[None, :]
    A = A + rng.normal(0, unit["noise"], A.shape)
    return A.astype(np.float32), axis_out


# ----------------------------------------------------------------------------- systems
def sample_system(system, n, rng):
    mu = constituent_mua()
    comp, scat, A = [], [], []
    for i in range(n):
        if system == "emulsion":
            f = rng.uniform(0.03, 0.40); p = rng.uniform(0.02, 0.10); cbh = rng.uniform(0.0, 0.06)
            w = 1 - f - p - cbh
            fr = dict(water=w, lipid=f, protein=p, carbohydrate=cbh, ethanol=0.0)
            d50 = float(np.exp(rng.uniform(np.log(0.3), np.log(5.0))))
            mus, g = emulsion_scattering(WL, f, d50)
            sc = dict(d50_um=d50)
        elif system == "powder":
            w = rng.uniform(0.02, 0.15); p = rng.uniform(0.05, 0.40); f = rng.uniform(0.01, 0.30)
            cbh = max(1 - w - p - f, 0.05)
            s_ = w + p + f + cbh
            fr = dict(water=w / s_, lipid=f / s_, protein=p / s_, carbohydrate=cbh / s_, ethanol=0.0)
            a = float(rng.uniform(5, 50)); b = float(rng.uniform(0.3, 1.5))
            mus, g = powder_scattering(WL, a, b)
            sc = dict(musp_1000=a, power=b)
        else:  # broth: clear liquid, transmission 1 mm
            e = rng.uniform(0.0, 0.10); s = rng.uniform(0.0, 0.10)
            fr = dict(water=1 - e - s, lipid=0.0, protein=0.0, carbohydrate=s, ethanol=e)
            mus, g = None, None
            sc = dict(path_mm=1.0)
        mua = sum(fr[k] * mu[k] for k in fr)
        if mus is None:
            Afine = mua * 1.0 / np.log(10)
        else:
            R = diffuse_reflectance(mua, mus, g)
            Afine = -np.log10(np.clip(R, 1e-6, None))
        comp.append([fr[k] for k in ["water", "lipid", "protein", "carbohydrate", "ethanol"]])
        scat.append(sc)
        A.append(Afine)
    return np.array(comp), scat, np.array(A)


def generate(n_per_system=1000, seed=0, systems=("emulsion", "powder", "broth")):
    SIM.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    units = [make_unit(c, k, np.random.default_rng(100 + 10 * i + k)) for i, c in enumerate(CLASSES) for k in range(3)]
    comps, scats, As, sys_lab = [], [], [], []
    for s in systems:
        c, sc, A = sample_system(s, n_per_system, rng)
        comps.append(c); scats += sc; As.append(A); sys_lab += [s] * len(c)
    comp = np.vstack(comps); Afine = np.vstack(As); sys_lab = np.array(sys_lab)
    split = np.where(rng.random(len(comp)) < 0.8, "train", "test")
    np.savez_compressed(SIM / "simfood_truth.npz", comp=comp, Afine=Afine.astype(np.float32), wl=WL, system=sys_lab, split=split,
                        comp_names=np.array(["water", "lipid", "protein", "carbohydrate", "ethanol"]))
    (SIM / "simfood_scatter.json").write_text(json.dumps(scats))
    for uid, u in enumerate(units):
        X, axis = measure(Afine, u, np.random.default_rng(1000 + uid))
        np.savez_compressed(SIM / f"simfood_vnir_unit{uid:02d}.npz", X=X, axis=axis, axis_unit=CLASSES[u["cls"]]["unit"],
                            unit_id=uid, split=split, system=sys_lab, comp=comp, params=json.dumps(u))
    (SIM / "units.json").write_text(json.dumps(units, indent=1))
    return units
