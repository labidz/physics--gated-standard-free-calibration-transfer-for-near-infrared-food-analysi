"""E4/E5: identifiability and interpretability.

(a) SimFood-RT (exact ground truth): recovery of composition fractions, pure-component shapes and
    instrument parameters by a small SpecTwin (K = number of constituents) vs PCA, NMF, MCR-ALS.
(b) Identifiability through pooling (Theorem 3): recovery on emulsions when the dictionary is
    learned from emulsions only, + powders, + broths.
(c) Local identifiability certificate (Theorem 2): Spearman(kappa(J), per-sample fraction error).
(d) Real ground truth: carbs (Raman; pure spectra + concentrations), minerals (Raman; fractions),
    cookie (NIR; designed recipe fractions).
(e) Band assignments of the pooled twin's dictionary vs literature NIR bands.
"""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
torch.set_num_threads(1)
from scipy.optimize import linear_sum_assignment, nnls
from scipy.stats import spearmanr
from sklearn.decomposition import NMF, PCA
from spectwin.bench import load
from spectwin.grid import to_global, global_axis
from spectwin.twin import SpecTwin, TwinConfig, Corpus, train_ssl, encode_np, MODS, SIGNALS
from spectwin.twin_run import load_twin

OUT = ROOT / "results" / "e4"; OUT.mkdir(parents=True, exist_ok=True)
SIM = ROOT / "data" / "sim"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800


def affine_r2(Z, C):
    """Mean R^2 of predicting each true fraction from the latent (affine identifiability)."""
    A = np.column_stack([Z, np.ones(len(Z))])
    r2 = []
    for j in range(C.shape[1]):
        if C[:, j].std() < 1e-9:
            continue
        coef, *_ = np.linalg.lstsq(A, C[:, j], rcond=None)
        e = C[:, j] - A @ coef
        r2.append(1 - (e ** 2).sum() / ((C[:, j] - C[:, j].mean()) ** 2).sum())
    return float(np.mean(r2))


def perm_corr(Z, C):
    """Mean |correlation| after optimal one-to-one matching (identifiability up to permutation/scale)."""
    keep = [j for j in range(C.shape[1]) if C[:, j].std() > 1e-9]
    C = C[:, keep]
    M = np.zeros((Z.shape[1], C.shape[1]))
    for i in range(Z.shape[1]):
        for j in range(C.shape[1]):
            if Z[:, i].std() > 1e-12:
                M[i, j] = abs(np.corrcoef(Z[:, i], C[:, j])[0, 1])
    r, c = linear_sum_assignment(-M)
    return float(M[r, c].mean()), r, c


def mcr_als(X, k, iters=200, seed=0):
    rng = np.random.default_rng(seed)
    Xp = X - X.min()
    C = rng.random((len(X), k))
    for _ in range(iters):
        S = np.array([nnls(C, Xp[:, j])[0] for j in range(Xp.shape[1])])          # p x k
        C = np.array([nnls(S, Xp[i])[0] for i in range(len(Xp))])                # n x k
    return C, S


def small_twin(datasets, K, steps=STEPS, modality="vnir", seed=0, q=1):
    cfg = TwinConfig(K=K, q=q, w_res=10.0)
    m = SpecTwin(cfg)
    c = Corpus()
    for (name, Xg, valid, sig, did) in datasets:
        c.add(name, Xg, valid, modality, sig, did)
    train_ssl(m, c, steps=steps, B=64, lr=2e-3, seed=seed, log=None)
    return m, c


def eval_latents(m, Xg, valid, did, scale, modality="vnir", sig="absorbance"):
    lat = encode_np(m, Xg, valid, modality, sig, did, scale)
    return lat


def baselines_recovery(Xtr, Xte, Cte, K):
    out = {}
    pca = PCA(K).fit(Xtr); out["PCA"] = (affine_r2(pca.transform(Xte), Cte), perm_corr(pca.transform(Xte), Cte)[0])
    off = min(Xtr.min(), Xte.min())
    nmf = NMF(K, init="nndsvda", max_iter=2000, random_state=0).fit(Xtr - off)
    Zn = nmf.transform(Xte - off); Zn = Zn / (Zn.sum(1, keepdims=True) + 1e-12)
    out["NMF"] = (affine_r2(Zn, Cte), perm_corr(Zn, Cte)[0])
    Cm, S = mcr_als(np.vstack([Xtr[:300], Xte]), K, iters=60)
    Zm = Cm[min(300, len(Xtr)):]; Zm = Zm / (Zm.sum(1, keepdims=True) + 1e-12)
    out["MCR-ALS"] = (affine_r2(Zm, Cte), perm_corr(Zm, Cte)[0])
    return out


def sim_unit(uid):
    z = np.load(SIM / f"simfood_vnir_unit{uid:02d}.npz")
    return z


def part_a_b_c(res):
    truth = np.load(SIM / "simfood_truth.npz")
    names = list(truth["comp_names"])
    systems = {"emulsion": [0, 1, 2, 3], "powder": [0, 1, 2, 3], "broth": [0, 3, 4]}
    z = sim_unit(3)                       # dispersive-NIR unit 0 (1100-2498 nm)
    Xg, valid = to_global(z["X"], z["axis"], str(z["axis_unit"]), "vnir")
    syst, split, comp = z["system"], z["split"], z["comp"]
    # (a) per-system recovery
    for s, cols in systems.items():
        tr = np.where((syst == s) & (split == "train"))[0]; te = np.where((syst == s) & (split == "test"))[0]
        K = len(cols)
        m, c = small_twin([(s, Xg[tr], valid, "absorbance", 0)], K)
        lat = eval_latents(m, Xg[te], valid, 0, c.items[0]["scale"])
        Cte = comp[te][:, cols]
        r = dict(system=s, K=K, SpecTwin=(affine_r2(lat["c"], Cte), perm_corr(lat["c"], Cte)[0]))
        r.update(baselines_recovery(z["X"][tr].astype(float), z["X"][te].astype(float), Cte, K))
        # (c) local identifiability certificate: Jacobian of the decoder wrt latent parameters
        kap = jacobian_condition(m, Xg[te[:150]], valid, c.items[0]["scale"])
        err = per_sample_fraction_error(lat["c"][:150], Cte[:150])
        rho, pv = spearmanr(kap, err)
        r["kappa_spearman"] = (float(rho), float(pv)); r["kappa_median"] = float(np.median(kap))
        res["a_recovery"].append(r)
        print("E4a", s, {k: (np.round(v, 3) if isinstance(v, tuple) else v) for k, v in r.items()}, flush=True)
    # (b) pooling: dictionary learned on emulsion only / + powder / + broth; evaluate emulsion test
    te = np.where((syst == "emulsion") & (split == "test"))[0]
    Cte = comp[te][:, [0, 1, 2, 3]]
    for pool in [["emulsion"], ["emulsion", "powder"], ["emulsion", "powder", "broth"]]:
        sets = []
        for i, s in enumerate(pool):
            tr = np.where((syst == s) & (split == "train"))[0]
            sets.append((s, Xg[tr], valid, "absorbance", i))
        m, c = small_twin(sets, 5)
        lat = eval_latents(m, Xg[te], valid, 0, c.items[0]["scale"])
        r = dict(pool="+".join(pool), affine_r2=affine_r2(lat["c"], Cte), perm_corr=perm_corr(lat["c"], Cte)[0])
        res["b_pooling"].append(r)
        print("E4b", r, flush=True)
    # instrument recovery: twin on the three dispersive-NIR units (separate instrument parameters)
    sets, truths = [], []
    units = json.loads((SIM / "units.json").read_text())
    for j, uid in enumerate([3, 4, 5]):
        zz = sim_unit(uid)
        Xu, vu = to_global(zz["X"], zz["axis"], str(zz["axis_unit"]), "vnir")
        tr = np.where(zz["split"] == "train")[0]
        sets.append((f"u{uid}", Xu[tr], vu, "absorbance", j)); truths.append(units[uid]["shift"])
    m, c = small_twin(sets, 5, steps=STEPS)
    gax = global_axis("vnir")
    est = m.ds_inst.detach().numpy()[:3, 0]
    # convert grid-point shift to nm at 1800 nm (log grid spacing)
    nm_per_pt = 1800 * (np.log(gax[-1]) - np.log(gax[0])) / (len(gax) - 1)
    est_nm = est * nm_per_pt
    rel_true = np.array(truths) - truths[0]; rel_est = est_nm - est_nm[0]
    res["instrument"] = dict(true_shift_nm=list(map(float, truths)), est_shift_nm=list(map(float, est_nm)),
                             rel_true=list(map(float, rel_true)), rel_est=list(map(float, rel_est)))
    print("E4 instrument", res["instrument"], flush=True)


def jacobian_condition(m, Xg, valid, scale):
    """Condition number of d(decoded spectrum)/d(w, a, gamma) at the encoder's estimate, per sample."""
    V = torch.tensor(valid)
    out = []
    for i in range(len(Xg)):
        x = torch.tensor(Xg[i:i + 1])
        v = V[None, :]
        sig = torch.tensor([SIGNALS.index("absorbance")]); mod = torch.tensor([0])
        with torch.no_grad():
            lat = m.encode(x, v, mod, sig, torch.tensor([scale]))
        K, P = m.cfg.K, m.cfg.P_abs
        theta0 = torch.cat([lat["w"][0], lat["a"][0:1], lat["gam"][0, :P]]).detach()

        def f(theta):
            l2 = dict(lat)
            l2 = {k: (t.clone() if torch.is_tensor(t) else t) for k, t in lat.items()}
            l2["w"] = theta[:K][None]
            l2["a"] = theta[K:K + 1]
            g = l2["gam"].clone(); g[0, :P] = theta[K + 1:K + 1 + P]; l2["gam"] = g
            xp, _ = m.decode(l2, v, 0, sig, torch.tensor([0]))
            return xp[0][V]
        J = torch.autograd.functional.jacobian(f, theta0).numpy()
        s = np.linalg.svd(J, compute_uv=False)
        out.append(s[0] / max(s[-1], 1e-12))
    return np.array(out)


def per_sample_fraction_error(Z, C):
    A = np.column_stack([Z, np.ones(len(Z))])
    coef, *_ = np.linalg.lstsq(A, C, rcond=None)
    return np.sqrt(((A @ coef - C) ** 2).mean(1))


def part_d(res):
    # carbs: Raman mixtures with pure spectra
    d = load("carbs"); mix = d.domains["role"] == "mixture"
    Xg, valid = to_global(d.X, d.axis, d.axis_unit, "raman")
    C = np.column_stack([d.targets[k] for k in ["fructose", "lactose", "ribose"]])
    m, c = small_twin([("carbs", Xg[mix], valid, "raman", 0)], 3, modality="raman", steps=STEPS)
    lat = encode_np(m, Xg[mix], valid, "raman", "raman", 0, c.items[0]["scale"])
    Cm = C[mix] / C[mix].sum(1, keepdims=True)
    E = m.dictionary(MODS.index("raman")).detach().numpy()[:, valid]
    pure = Xg[~mix][:, valid]
    ang = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            a, b = E[i] - E[i].mean(), pure[j] - pure[j].mean()
            ang[i, j] = np.degrees(np.arccos(np.clip(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1)))
    r_, c_ = linear_sum_assignment(ang)
    pc, *_ = perm_corr(lat["c"], Cm)
    base = baselines_recovery(d.X[mix].astype(float), d.X[mix].astype(float), Cm, 3)
    res["d_real"].append(dict(dataset="carbs (Raman)", SpecTwin_affine_r2=affine_r2(lat["c"], Cm), SpecTwin_perm_corr=pc,
                              atom_vs_pure_angle_deg=[float(ang[i, j]) for i, j in zip(r_, c_)],
                              baselines={k: v for k, v in base.items()}))
    print("E4d carbs", res["d_real"][-1], flush=True)
    # minerals (non-food Raman, closure)
    d = load("minerals"); Xg, valid = to_global(d.X, d.axis, d.axis_unit, "raman")
    C = np.column_stack([d.targets[k] for k in ["albite", "quartz", "orthoclase"]])
    m, c = small_twin([("minerals", Xg, valid, "raman", 0)], 3, modality="raman", steps=STEPS)
    lat = encode_np(m, Xg, valid, "raman", "raman", 0, c.items[0]["scale"])
    base = baselines_recovery(d.X.astype(float), d.X.astype(float), C, 3)
    res["d_real"].append(dict(dataset="minerals (Raman, non-food)", SpecTwin_affine_r2=affine_r2(lat["c"], C), SpecTwin_perm_corr=perm_corr(lat["c"], C)[0], baselines=base))
    print("E4d minerals", res["d_real"][-1], flush=True)
    # cookie dough: designed recipe
    d = load("cookie"); Xg, valid = to_global(d.X, d.axis, d.axis_unit, "vnir")
    C = np.column_stack([d.targets[k] for k in ["fat", "sucrose", "dry_flour", "water"]]); C = C / C.sum(1, keepdims=True)
    m, c = small_twin([("cookie", Xg, valid, "absorbance", 0)], 4, steps=STEPS)
    lat = encode_np(m, Xg, valid, "vnir", "absorbance", 0, c.items[0]["scale"])
    base = baselines_recovery(d.X.astype(float), d.X.astype(float), C, 4)
    res["d_real"].append(dict(dataset="cookie dough (NIR)", SpecTwin_affine_r2=affine_r2(lat["c"], C), SpecTwin_perm_corr=perm_corr(lat["c"], C)[0], baselines=base))
    print("E4d cookie", res["d_real"][-1], flush=True)


LIT_BANDS = {  # common NIR band assignments (nm); Workman & Weyer (2012); Osborne et al. (1993)
    "O-H 3rd overtone (water)": 760, "C-H 3rd overtone (CH2/CH3, lipid)": 930, "O-H 2nd overtone (water)": 970,
    "N-H 2nd overtone (protein)": 1020, "O-H comb. (water)": 1190, "C-H 2nd overtone (lipid)": 1210,
    "C-H comb. (lipid)": 1395, "O-H 1st overtone (water)": 1450, "N-H 1st overtone (protein)": 1510,
    "O-H 1st overtone (starch/sugar)": 1580, "C-H 1st overtone (lipid)": 1725, "C-H 1st overtone (CH2, lipid)": 1765,
    "O-H comb. (water)_2": 1940, "N-H comb. (protein)": 2055, "O-H/C-O comb. (carbohydrate)": 2100,
    "N-H/C=O comb. (protein, amide)": 2180, "C-H/C-H comb. (starch/cellulose)": 2270, "C-H comb. (lipid)_2": 2310,
    "C-H comb. (lipid)_3": 2350, "chlorophyll a (Vis)": 675, "carotenoids (Vis)": 480,
}


def part_e(res):
    m = load_twin("vnir", "real")
    E = m.dictionary(0).detach().numpy(); ax = global_axis("vnir")
    lit = np.array(list(LIT_BANDS.values()), float)
    peaks = []
    from scipy.signal import find_peaks
    for k in range(E.shape[0]):
        pk, prop = find_peaks(E[k], prominence=0.15 * E[k].max())
        peaks += [float(ax[p]) for p in pk if 600 <= ax[p] <= 2450]
    peaks = np.array(peaks)
    dist = np.array([np.min(np.abs(lit - p)) for p in peaks])
    rng = np.random.default_rng(0)
    rand = rng.uniform(600, 2450, (2000, len(peaks)))
    rdist = np.array([[np.min(np.abs(lit - p)) for p in row] for row in rand[:200]])
    res["e_bands"] = dict(n_peaks=int(len(peaks)), median_dist_nm=float(np.median(dist)) if len(dist) else None,
                          frac_within_15nm=float(np.mean(dist <= 15)) if len(dist) else None,
                          random_median_dist_nm=float(np.median(rdist)), random_frac_within_15nm=float(np.mean(rdist <= 15)),
                          peaks_nm=sorted(map(float, np.round(peaks, 1))))
    print("E5 bands", {k: v for k, v in res["e_bands"].items() if k != "peaks_nm"}, flush=True)


if __name__ == "__main__":
    parts = sys.argv[1].split(",") if len(sys.argv) > 1 else ["abc", "d", "e"]
    fp = OUT / "e4_results.json"
    res = json.loads(fp.read_text()) if fp.exists() else {"a_recovery": [], "b_pooling": [], "d_real": []}
    t0 = time.time()
    if "abc" in parts:
        res["a_recovery"], res["b_pooling"] = [], []
        part_a_b_c(res); fp.write_text(json.dumps(res, indent=1))
    if "d" in parts:
        res["d_real"] = []
        part_d(res); fp.write_text(json.dumps(res, indent=1))
    if "e" in parts:
        part_e(res); fp.write_text(json.dumps(res, indent=1))
    print(f"E4 done in {(time.time() - t0) / 60:.1f} min", flush=True)
