"""E4 supplement: non-amortised physics factorisation (SpecTwin decoder fitted by direct optimisation).

Tests the identifiability of the measurement model class itself, separately from encoder amortisation:
    x_i = a_i + sum_p gamma_ip L_p + sum_k w_ik E_k,   w >= 0, E >= 0 smooth
(+ fitted on training spectra; test latents re-optimised with E fixed).
"""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
from run_e4 import affine_r2, perm_corr
from spectwin.bench import load

OUT = ROOT / "results" / "e4"; SIM = ROOT / "data" / "sim"


def legendre(p, P):
    if P < 0:
        return torch.zeros((0, p), dtype=torch.float32)
    t = np.linspace(-1, 1, p)
    L = [np.ones(p), t, 0.5 * (3 * t ** 2 - 1), 0.5 * (5 * t ** 3 - 3 * t)]
    pm2, pm1 = L[2], L[3]
    for n in range(4, P + 1):
        pn = ((2 * n - 1) * t * pm1 - (n - 1) * pm2) / n
        L.append(pn); pm2, pm1 = pm1, pn
    return torch.tensor(np.stack(L[:P + 1]), dtype=torch.float32)


def smoother(p, s=1.5):
    i = np.arange(p); K = np.exp(-0.5 * ((i[:, None] - i[None, :]) / s) ** 2); K /= K.sum(1, keepdims=True)
    return torch.tensor(K, dtype=torch.float32)


def fit_factorisation(X, K, P=3, steps=3000, E=None, seed=0, lr=0.03):
    torch.manual_seed(seed)
    X = torch.tensor(X, dtype=torch.float32)
    n, p = X.shape
    mu, sd = X.mean(), X.std()
    Xs = (X - mu) / sd
    L = legendre(p, P)
    S = smoother(p)
    learn_E = E is None
    E_raw = torch.nn.Parameter(torch.randn(K, p) * 0.1 - 1.0) if learn_E else None
    W_raw = torch.nn.Parameter(torch.randn(n, K) * 0.1 - 1.0)
    B = torch.nn.Parameter(torch.zeros(n, P + 1))
    params = [W_raw, B] + ([E_raw] if learn_E else [])
    opt = torch.optim.Adam(params, lr=lr)
    for it in range(steps):
        Em = (F.softplus(E_raw) @ S.T) if learn_E else E
        Em = Em / Em.norm(dim=1, keepdim=True).clamp_min(1e-8) * np.sqrt(p)
        rec = F.softplus(W_raw) @ Em + B @ L
        loss = ((rec - Xs) ** 2).mean() + 1e-4 * F.softplus(W_raw).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Em = (F.softplus(E_raw) @ S.T) if learn_E else E
        Em = Em / Em.norm(dim=1, keepdim=True).clamp_min(1e-8) * np.sqrt(p)
        Wt = F.softplus(W_raw).numpy()
    return Wt / (Wt.sum(1, keepdims=True) + 1e-12), Em.detach(), float(loss)


def run():
    res = []
    truth = np.load(SIM / "simfood_truth.npz")
    systems = {"emulsion": [0, 1, 2, 3], "powder": [0, 1, 2, 3], "broth": [0, 3, 4]}
    z = np.load(SIM / "simfood_vnir_unit03.npz")
    for s, cols in systems.items():
        tr = np.where((z["system"] == s) & (z["split"] == "train"))[0][:600]
        te = np.where((z["system"] == s) & (z["split"] == "test"))[0]
        K = len(cols)
        Xall = z["X"].astype(float)
        Wtr, E, l1 = fit_factorisation(Xall[tr], K)
        Wte, _, l2 = fit_factorisation(Xall[te], K, E=E)
        C = z["comp"][te][:, cols]
        r = dict(system=s, K=K, affine_r2=affine_r2(Wte, C), perm_corr=perm_corr(Wte, C)[0], train_loss=l1, test_loss=l2)
        # same model WITHOUT the baseline terms (pure non-negative factorisation with smoothness)
        Wtr0, E0, _ = fit_factorisation(Xall[tr] - Xall[tr].min(), K, P=-1)
        Wte0, _, _ = fit_factorisation(Xall[te] - Xall[tr].min(), K, P=-1, E=E0)
        r["no_baseline_affine_r2"] = affine_r2(Wte0, C); r["no_baseline_perm_corr"] = perm_corr(Wte0, C)[0]
        res.append(r); print("E4-NA", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    # real ground truth: cookie (designed recipe)
    d = load("cookie"); C = np.column_stack([d.targets[k] for k in ["fat", "sucrose", "dry_flour", "water"]]); C = C / C.sum(1, keepdims=True)
    W, E, l = fit_factorisation(d.X.astype(float), 4)
    res.append(dict(system="cookie (NIR, real)", K=4, affine_r2=affine_r2(W, C), perm_corr=perm_corr(W, C)[0]))
    print("E4-NA cookie", res[-1], flush=True)
    for name, ks, mod in [("carbs", ["fructose", "lactose", "ribose"], "raman"), ("minerals", ["albite", "quartz", "orthoclase"], "raman")]:
        d = load(name); C = np.column_stack([d.targets[k] for k in ks])
        rows = np.where(d.domains["role"] == "mixture")[0] if name == "carbs" else np.arange(len(d.X))
        Cn = C[rows] / C[rows].sum(1, keepdims=True)
        W, E, l = fit_factorisation(d.X[rows].astype(float), 3, P=6)
        res.append(dict(system=f"{name} (Raman, real)", K=3, affine_r2=affine_r2(W, Cn), perm_corr=perm_corr(W, Cn)[0]))
        print("E4-NA", name, res[-1], flush=True)
    (OUT / "e4_nonamortised.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    run()
