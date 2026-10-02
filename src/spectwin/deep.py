"""Deep-learning baselines: 1D-CNN (Cui & Fearn / Bjerrum style) and a patch Transformer (no physics, no pretraining)."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn

from .chemo import group_kfold
from .prep import snv


def set_threads(n=1):
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)


# ----------------------------------------------------------------------------- models
class CNN1D(nn.Module):
    def __init__(self, p, n_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(1, 8, 9, padding=4), nn.ELU(),
            nn.Conv1d(8, 16, 9, stride=2, padding=4), nn.ELU(),
            nn.Conv1d(16, 32, 9, stride=2, padding=4), nn.ELU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ELU(), nn.Linear(64, n_out))

    def forward(self, x):
        return self.f(x[:, None, :])


class PatchTransformer(nn.Module):
    def __init__(self, p, n_out, n_patch=32, d=64, layers=2, heads=4):
        super().__init__()
        self.ps = int(math.ceil(p / n_patch))
        self.pad = self.ps * n_patch - p
        self.embed = nn.Linear(self.ps, d)
        self.pos = nn.Parameter(torch.randn(1, n_patch + 1, d) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_out))

    def forward(self, x):
        if self.pad:
            x = nn.functional.pad(x, (0, self.pad), mode="replicate")
        B = x.shape[0]
        t = self.embed(x.view(B, -1, self.ps))
        t = torch.cat([self.cls.expand(B, -1, -1), t], 1) + self.pos
        return self.head(self.enc(t)[:, 0])


# ----------------------------------------------------------------------------- training
def _augment(X, rng):
    """Bjerrum et al. (2017) style: offset, slope, multiplicative scatter on raw spectra."""
    n, p = X.shape
    sd = X.std() + 1e-12
    t = np.linspace(-1, 1, p)[None, :]
    off = rng.normal(0, 0.05 * sd, (n, 1))
    slope = rng.normal(0, 0.05 * sd, (n, 1))
    mult = 1 + rng.normal(0, 0.05, (n, 1))
    return X * mult + off + slope * t


def train_net(arch, Xtr, ytr, gtr, kind, seed=0, max_epochs=200, patience=20, max_steps=None, bs=64, lr=1e-3, augment=True):
    """Train with an inner grouped 15% validation split for early stopping. Returns a predict function."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    folds = group_kfold(gtr, 7, seed)          # ~15% validation
    tr, va = folds[0]
    if kind == "cls":
        classes = np.unique(ytr)
        yi = np.searchsorted(classes, ytr)
        n_out = len(classes)
        Y = yi
    else:
        mu, sd = float(np.mean(ytr)), float(np.std(ytr) + 1e-12)
        Y = (ytr.astype(float) - mu) / sd
        n_out = 1
    p = Xtr.shape[1]
    if max_steps is None:
        max_steps = 4000 if arch == "cnn" else 2000
    net = (CNN1D if arch == "cnn" else PatchTransformer)(p, n_out)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss() if kind == "cls" else nn.MSELoss()
    Xva = torch.tensor(snv(Xtr[va]), dtype=torch.float32)
    Yva = torch.tensor(Y[va], dtype=torch.long if kind == "cls" else torch.float32)
    steps_per_epoch = max(1, int(math.ceil(len(tr) / bs)))
    epochs = int(min(max_epochs, max(20, math.ceil(max_steps / steps_per_epoch))))
    best, best_state, wait = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            xb = _augment(Xtr[b], rng) if augment else Xtr[b]
            xb = torch.tensor(snv(xb), dtype=torch.float32)
            yb = torch.tensor(Y[b], dtype=torch.long if kind == "cls" else torch.float32)
            out = net(xb)
            loss = lossf(out, yb) if kind == "cls" else lossf(out[:, 0], yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            o = net(Xva)
            v = lossf(o, Yva).item() if kind == "cls" else lossf(o[:, 0], Yva).item()
        if v < best - 1e-6:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        with torch.no_grad():
            o = net(torch.tensor(snv(X), dtype=torch.float32)).numpy()
        if kind == "cls":
            return classes[o.argmax(1)]
        return o[:, 0] * sd + mu

    info = dict(epochs=ep + 1, best_val=float(best), n_params=int(sum(t.numel() for t in net.parameters())))
    return predict, info


def m_deep(arch, Xtr, ytr, gtr, Xte, kind, seed=0):
    t = time.time()
    pred, info = train_net(arch, Xtr, ytr, gtr, kind, seed=seed)
    info["sec"] = time.time() - t
    return pred(Xte), info
