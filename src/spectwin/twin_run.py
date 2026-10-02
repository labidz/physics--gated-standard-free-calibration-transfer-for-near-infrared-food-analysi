"""Shared helpers for running SpecTwin on benchmark datasets (latent caching, adaptation, heads)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .bench import load
from .chemo import RidgePath, group_kfold
from .grid import GRIDS, native_to_modality_axis, to_global
from .twin import SpecTwin, TwinConfig, adapt, encode_np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "results" / "models"
CACHE = ROOT / "results" / "cache"

PRETRAIN = {"vnir": ["mango", "ch2021", "ch2018", "wheat", "cassava", "sorghum", "forages", "pears", "grapes",
                     "tecator", "cookie", "mayonnaise", "oregano"],
            "raman": ["carbs", "minerals"], "mir": ["ligno"]}


def load_twin(modality, variant="real"):
    ck = torch.load(MODELS / f"twin_{modality}_{variant}.pt", weights_only=False)
    cfg = TwinConfig(**ck["cfg"])
    m = SpecTwin(cfg)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


def ds_ids():
    return json.loads((MODELS / "ds_ids.json").read_text())


def scale_of(Xg_train, valid):
    return float(Xg_train[:, valid].std()) + 1e-8


def latents_for(dataset, train_rows, rows, variant="real", adapt_steps=200, seed=0, cache_key=None, model_cache={}):
    """Encode `rows` of `dataset` with the twin; adapt (unlabelled, train_rows only) if the dataset was not in the corpus."""
    ds = load(dataset)
    mod = ds.modality
    key = (mod, variant)
    if key not in model_cache:
        model_cache[key] = load_twin(mod, variant)
    base = model_cache[key]
    did = ds_ids()[dataset]
    Xg, valid = to_global(ds.X, ds.axis, ds.axis_unit, mod)
    sc = scale_of(Xg[train_rows], valid)
    path = None
    if cache_key is not None:
        path = CACHE / f"twin_{variant}" / f"{dataset}__{cache_key}.npz"
        if path.exists():
            z = np.load(path)
            return {k: z[k] for k in z.files}, valid
    in_corpus = dataset in PRETRAIN.get(mod, [])
    model = base if in_corpus else adapt(base, Xg[train_rows], valid, mod, ds.signal, did, steps=adapt_steps, seed=seed)
    lat = encode_np(model, Xg[rows], valid, mod, ds.signal, did, sc)
    lat["rows"] = np.asarray(rows)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **lat)
    return lat, valid


def phys_features(lat, P=3):
    w = lat["w"]
    return np.column_stack([w, np.log(w.sum(1, keepdims=True) + 1e-8), lat["a"][:, None], lat["gam"][:, :P], lat["u"],
                            np.log(lat["sd"][:, None] + 1e-12), lat["dshift"][:, None], lat["dsig"][:, None]])


def ridge_feat(Ftr, ytr, gtr, Fte, kind, seed=0, alphas=np.logspace(-6, 3, 10)):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12
    A, B = (Ftr - mu) / sd, (Fte - mu) / sd
    if kind == "cls":
        classes = np.unique(ytr)
        Y = (ytr[:, None] == classes[None, :]).astype(float)
        yi = np.searchsorted(classes, ytr)
    else:
        Y = ytr.astype(float)
    score = np.zeros(len(alphas))
    for tr, va in group_kfold(gtr, 5, seed):
        r = RidgePath().fit(A[tr], Y[tr])
        for j, a in enumerate(alphas):
            p = r.predict(A[va], a)
            score[j] += (p.argmax(1) != yi[va]).sum() if kind == "cls" else ((p - Y[va]) ** 2).sum()
    a = alphas[int(np.argmin(score))]
    p = RidgePath().fit(A, Y).predict(B, a)
    return (classes[p.argmax(1)] if kind == "cls" else p), dict(alpha=float(a))


def mlp_head(Ftr, ytr, gtr, Fte, kind, seed=0, max_steps=3000, bs=64, patience=30):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    A = torch.tensor((Ftr - mu) / sd, dtype=torch.float32)
    B = torch.tensor((Fte - mu) / sd, dtype=torch.float32)
    tr, va = group_kfold(gtr, 7, seed)[0]
    if kind == "cls":
        classes = np.unique(ytr)
        Y = torch.tensor(np.searchsorted(classes, ytr), dtype=torch.long)
        n_out, lossf = len(classes), nn.CrossEntropyLoss()
    else:
        ym, ys = float(np.mean(ytr)), float(np.std(ytr) + 1e-12)
        Y = torch.tensor((ytr - ym) / ys, dtype=torch.float32)
        n_out, lossf = 1, nn.MSELoss()
    net = nn.Sequential(nn.Linear(A.shape[1], 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_out))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    best, state, wait, step = np.inf, None, 0, 0
    epoch_steps = max(1, math.ceil(len(tr) / bs))
    while step < max_steps:
        net.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            o = net(A[b])
            loss = lossf(o, Y[b]) if kind == "cls" else lossf(o[:, 0], Y[b])
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
        net.eval()
        with torch.no_grad():
            o = net(A[va])
            v = (lossf(o, Y[va]) if kind == "cls" else lossf(o[:, 0], Y[va])).item()
        if v < best - 1e-6:
            best, wait, state = v, 0, {k: t.clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        o = net(B).numpy()
    return (classes[o.argmax(1)] if kind == "cls" else o[:, 0] * ys + ym), dict(steps=step)


def corrected_native(ds, lat, valid, rows):
    """Twin nuisance correction on the NATIVE grid (no information loss):
    x_corr = (x - mu - sd * (a + sum_p gamma_p L_p(t))) / (sd * sum(w))
    i.e. additive offset, low-order baseline and multiplicative scatter estimated by the twin are removed.
    Spectra already published as derivatives / SNV are returned unchanged."""
    X = ds.X[rows].astype(float)
    if ds.signal not in ("absorbance", "raman"):
        return X
    g = GRIDS[ds.modality]
    ax = native_to_modality_axis(ds.axis, ds.axis_unit, ds.modality)
    if g["kind"] == "log":
        j = (np.log(ax) - np.log(g["lo"])) / (np.log(g["hi"]) - np.log(g["lo"])) * (g["G"] - 1)
    else:
        j = (ax - g["lo"]) / (g["hi"] - g["lo"]) * (g["G"] - 1)
    vi = np.where(valid)[0]
    t = np.clip(2 * (j - vi.min()) / max(vi.max() - vi.min(), 1) - 1, -1, 1)
    P = 6 if ds.modality == "raman" else 3
    polys = [t, 0.5 * (3 * t ** 2 - 1), 0.5 * (5 * t ** 3 - 3 * t)]
    pm2, pm1 = polys[1], polys[2]
    for n in range(4, P + 1):
        pn = ((2 * n - 1) * t * pm1 - (n - 1) * pm2) / n
        polys.append(pn); pm2, pm1 = pm1, pn
    Lb = np.stack(polys[:P])                                    # P x p
    base = lat["a"][:, None] + lat["gam"][:, :P] @ Lb          # n x p (standardised units)
    tot = lat["w"].sum(1, keepdims=True)
    return (X - lat["mu"][:, None] - lat["sd"][:, None] * base) / (lat["sd"][:, None] * np.maximum(tot, 1e-6))


def corrected_global(Xg, lat, valid, modality, signal):
    """Same nuisance correction on the global (grid-agnostic) axis; used when train/test grids differ (E6)."""
    X = np.asarray(Xg, float)[:, valid]
    if signal not in ("absorbance", "raman"):
        return X
    vi = np.where(valid)[0]
    t = 2 * (vi - vi.min()) / max(vi.max() - vi.min(), 1) - 1
    P = 6 if modality == "raman" else 3
    polys = [t, 0.5 * (3 * t ** 2 - 1), 0.5 * (5 * t ** 3 - 3 * t)]
    pm2, pm1 = polys[1], polys[2]
    for n in range(4, P + 1):
        pn = ((2 * n - 1) * t * pm1 - (n - 1) * pm2) / n
        polys.append(pn); pm2, pm1 = pm1, pn
    base = lat["a"][:, None] + lat["gam"][:, :P] @ np.stack(polys[:P])
    tot = lat["w"].sum(1, keepdims=True)
    return (X - lat["mu"][:, None] - lat["sd"][:, None] * base) / (lat["sd"][:, None] * np.maximum(tot, 1e-6))
