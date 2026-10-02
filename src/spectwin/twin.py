"""SpecTwin: a self-supervised, physics-factorised measurement twin for spectroscopy.

x = P_d[ G_psi( a + b * sqrt(G) * E^T c + L gamma + R^T u ) ] + noise

E   non-negative, smooth absorptivity dictionary shared across datasets (per modality), K atoms
c   composition fractions (softmax, closure)          b  multiplicative scatter / path length
a   additive offset    L gamma  Legendre baseline over the measured range    R^T u  small residual
G   instrument operator: axis shift, Gaussian line-shape broadening, stray light (absorbance only)
P_d the preprocessing a data publisher already applied (SG 1st derivative, SNV + SG 2nd derivative, SNV)

The encoder is a transformer over patches of the global (physical-wavelength) grid; it is the
amortised inverse of the measurement chain. Training: masked physical reconstruction + chemistry
invariance and instrument equivariance under simulated physical perturbations.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics import SGConv, gaussian_smoother, legendre_basis, masked_snv, shift_broaden, stray_light

MODS = ["vnir", "mir", "raman"]
SIGNALS = ["absorbance", "raman", "deriv1", "snv_sg2", "snv"]


@dataclass
class TwinConfig:
    K: int = 64
    q: int = 8
    P_abs: int = 3
    P_raman: int = 6
    d: int = 96
    layers: int = 3
    heads: int = 4
    G: int = 1024
    patch: int = 16
    max_datasets: int = 96
    smooth_sigma: float = 1.0
    w_res: float = 1.0
    w_inv: float = 1.0
    w_eq: float = 0.5
    w_ent: float = 0.001
    w_inst: float = 0.01
    w_dinst: float = 0.1
    mask_frac: float = 0.15
    # ablation switches
    use_physics: bool = True
    use_instrument: bool = True
    use_scatter: bool = True
    use_prep_op: bool = True
    use_invariance: bool = True
    aug: dict = field(default_factory=lambda: dict(shift=2.0, sigma=2.0, logb=0.2, offset=0.2, slope=0.1, stray=0.0005))


def edge_fill(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Replace invalid (out-of-range) grid values by the nearest valid edge value."""
    B, G = x.shape
    idx = torch.arange(G, device=x.device)[None, :].expand(B, G)
    big = torch.full_like(idx, 10 ** 9)
    jmin = torch.where(valid, idx, big).min(1).values
    jmax = torch.where(valid, idx, -big).max(1).values
    left = x.gather(1, jmin[:, None]).expand(B, G)
    right = x.gather(1, jmax[:, None]).expand(B, G)
    out = torch.where(idx < jmin[:, None], left, x)
    return torch.where(idx > jmax[:, None], right, out)


class SpecTwin(nn.Module):
    def __init__(self, cfg: TwinConfig = TwinConfig()):
        super().__init__()
        self.cfg = cfg
        d, G, K = cfg.d, cfg.G, cfg.K
        T = G // cfg.patch
        self.T = T
        self.Pmax = max(cfg.P_abs, cfg.P_raman)
        self.embed = nn.Linear(2 * cfg.patch, d)
        self.pos = nn.Parameter(torch.randn(1, T, d) * 0.02)
        self.mod_emb = nn.Embedding(len(MODS), d)
        self.sig_emb = nn.Embedding(len(SIGNALS), d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.scal = nn.Linear(2, d)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.layers)
        self.norm = nn.LayerNorm(d)
        self.n_lat = K + 2 + self.Pmax + cfg.q
        self.head = nn.Linear(d, self.n_lat + 3)
        centers = torch.linspace(0, G - 1, K)
        width = 1.5 * G / K
        bumps = torch.exp(-0.5 * ((torch.arange(G)[None, :] - centers[:, None]) / width) ** 2) + 1e-3
        inv_sp = torch.log(torch.expm1(bumps.clamp_min(1e-4)))
        self.E_raw = nn.Parameter(inv_sp[None].repeat(len(MODS), 1, 1) + 0.05 * torch.randn(len(MODS), K, G))
        self.R = nn.Parameter(torch.randn(len(MODS), cfg.q, G) * 0.01)
        self.register_buffer("S", gaussian_smoother(G, cfg.smooth_sigma))
        self.ds_inst = nn.Parameter(torch.zeros(cfg.max_datasets, 3))
        if not cfg.use_physics:
            self.bb_dec = nn.Sequential(nn.Linear(self.n_lat, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, G))
        self.sg1 = SGConv(1)
        self.sg2 = SGConv(2)

    # ------------------------------------------------------------------ encoder
    def encode(self, x, valid, mod, sig, scale):
        cfg = self.cfg
        B, G = x.shape
        v = valid.float()
        n = v.sum(1).clamp_min(1)
        mu = (x * v).sum(1) / n
        sd = (((x - mu[:, None]) ** 2) * v).sum(1).div(n).clamp_min(1e-16).sqrt()
        z = (x - mu[:, None]) / sd[:, None] * v
        zt = z.view(B, self.T, cfg.patch)
        vt = v.view(B, self.T, cfg.patch)
        tok = self.embed(torch.cat([zt, vt], -1)) + self.pos + self.mod_emb(mod)[:, None, :] + self.sig_emb(sig)[:, None, :]
        pad = vt.sum(-1) == 0
        cls = self.cls.expand(B, -1, -1) + self.scal(torch.stack([torch.log(sd / scale), mu / scale], 1))[:, None, :]
        h = self.enc(torch.cat([cls, tok], 1), src_key_padding_mask=torch.cat([torch.zeros(B, 1, dtype=torch.bool), pad], 1))
        h = self.norm(h[:, 0])
        o = self.head(h)
        K, P, q = cfg.K, self.Pmax, cfg.q
        lat = dict(h=h, mu=mu, sd=sd,
                   w=F.softplus(o[:, :K]),
                   c=F.softplus(o[:, :K]) / F.softplus(o[:, :K]).sum(1, keepdim=True).clamp_min(1e-8),
                   logb=o[:, K].clamp(-5, 5) if cfg.use_scatter else torch.zeros_like(o[:, K]),
                   a=o[:, K + 1] if cfg.use_scatter else torch.zeros_like(o[:, K + 1]),
                   gam=o[:, K + 2:K + 2 + P] if cfg.use_scatter else torch.zeros_like(o[:, K + 2:K + 2 + P]),
                   u=o[:, K + 2 + P:K + 2 + P + q],
                   dshift=4 * torch.tanh(o[:, -3] / 4) if cfg.use_instrument else torch.zeros_like(o[:, -3]),
                   dsig=3 * torch.sigmoid(o[:, -2] - 2) if cfg.use_instrument else torch.zeros_like(o[:, -2]),
                   dstray=0.002 * torch.sigmoid(o[:, -1] - 6) if cfg.use_instrument else torch.zeros_like(o[:, -1]))
        return lat

    # ------------------------------------------------------------------ decoder
    def dictionary(self, mod_idx: int):
        E = F.softplus(self.E_raw[mod_idx]) @ self.S.T
        return E / E.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def canonical_std(self, lat, mod_idx: int):
        """Scatter-, baseline- and instrument-free spectrum in standardised units: (b E^T c + R^T u) / b."""
        E = self.dictionary(mod_idx)
        tot = lat["w"].sum(1, keepdim=True).clamp_min(1e-6)
        chem = (lat["c"] @ E) * math.sqrt(self.cfg.G)
        res = lat["u"] @ self.R[mod_idx]
        return chem + res / tot, chem

    def decode(self, lat, valid, mod_idx: int, sig, ds):
        cfg = self.cfg
        if cfg.use_physics:
            E = self.dictionary(mod_idx)
            chem = (lat["w"] @ E) * math.sqrt(cfg.G)
            P = cfg.P_raman if MODS[mod_idx] == "raman" else cfg.P_abs
            L = legendre_basis(valid, P)
            base = lat["a"][:, None] + (lat["gam"][:, :P, None] * L).sum(1)
            res = lat["u"] @ self.R[mod_idx]
            full = chem + base + res
        else:
            z = torch.cat([lat["w"], lat["logb"][:, None], lat["a"][:, None], lat["gam"], lat["u"]], 1)
            full = self.bb_dec(z)
            res = torch.zeros_like(full)
        x = lat["mu"][:, None] + lat["sd"][:, None] * full
        if cfg.use_instrument and cfg.use_physics:
            p = self.ds_inst[ds]
            shift = p[:, 0] + lat["dshift"]
            sig_b = torch.sqrt(F.softplus(p[:, 1] - 2) ** 2 + lat["dsig"] ** 2 + 1e-6)
            x = shift_broaden(x, shift, sig_b)
            is_abs = sig == SIGNALS.index("absorbance")
            if is_abs.any():
                s = 0.002 * torch.sigmoid(p[:, 2] - 6) + lat["dstray"]
                xs = stray_light(x, s)
                x = torch.where(is_abs[:, None], xs, x)
        return x, res

    def apply_prep(self, x, valid, sig):
        """Apply the publisher's preprocessing operator P_d (per sample)."""
        if not self.cfg.use_prep_op:
            return x
        out = x
        for s_name, fn in [("deriv1", lambda t: self.sg1(t)),
                           ("snv_sg2", lambda t: self.sg2(masked_snv(edge_fill(t, valid), valid))),
                           ("snv", lambda t: masked_snv(t, valid))]:
            m = sig == SIGNALS.index(s_name)
            if m.any():
                out = torch.where(m[:, None], fn(x), out)
        return out

    # ------------------------------------------------------------------ losses
    def rec_loss(self, x_pred, x, valid, sig):
        raw = (sig == SIGNALS.index("absorbance")) | (sig == SIGNALS.index("raman"))
        v = valid.float()
        n = v.sum(1).clamp_min(1)
        mu = (x * v).sum(1) / n
        sd = (((x - mu[:, None]) ** 2) * v).sum(1).div(n).clamp_min(1e-16).sqrt()
        sd = torch.maximum(sd, 0.1 * sd.median())          # floor: nearly flat spectra must not dominate
        e_raw = (((x_pred - x) / sd[:, None]) ** 2 * v).sum(1) / n
        # preprocessed signals: compare scale-free (SNV) shapes, excluding 4 edge points
        ve = F.max_pool1d((~valid).float()[:, None, :], 9, 1, 4)[:, 0] < 0.5
        ve = ve & valid
        e_pp = ((masked_snv(x_pred, ve) - masked_snv(x, ve)) ** 2 * ve.float()).sum(1) / ve.float().sum(1).clamp_min(1)
        return torch.where(raw, e_raw, e_pp).mean()

    def augment(self, x, valid, sig, gen=None):
        a = self.cfg.aug
        B, G = x.shape
        dev = x.device
        r = lambda *s: torch.rand(*s, generator=gen, device=dev)
        rn = lambda *s: torch.randn(*s, generator=gen, device=dev)
        v = valid.float()
        n = v.sum(1).clamp_min(1)
        mu = (x * v).sum(1) / n
        sd = (((x - mu[:, None]) ** 2) * v).sum(1).div(n).clamp_min(1e-16).sqrt()
        raw = (sig == SIGNALS.index("absorbance")) | (sig == SIGNALS.index("raman"))
        w_shift = (2 * r(B) - 1) * a["shift"]
        w_sig = r(B) * a["sigma"]
        b = torch.exp((2 * r(B) - 1) * a["logb"])
        off = rn(B) * a["offset"] * sd
        slope = rn(B) * a["slope"] * sd
        t = torch.linspace(-1, 1, G, device=dev)[None, :]
        xf = edge_fill(x, valid)
        xa = torch.where(raw[:, None], b[:, None] * xf + off[:, None] + slope[:, None] * t, xf)
        xa = shift_broaden(xa, w_shift, w_sig)
        is_abs = sig == SIGNALS.index("absorbance")
        s = r(B) * a["stray"]
        xa = torch.where(is_abs[:, None], stray_light(xa, s), xa)
        return xa * v, dict(shift=w_shift, sig=w_sig)

    def ssl_step(self, x, valid, mod_idx, sig, ds, scale, gen=None):
        cfg = self.cfg
        B = x.shape[0]
        mod = torch.full((B,), mod_idx, dtype=torch.long)
        # clean branch with random patch masking
        pm = torch.rand(B, self.T, generator=gen) < cfg.mask_frac
        vin = valid & ~pm.repeat_interleave(cfg.patch, 1)
        lat = self.encode(x * vin.float(), vin, mod, sig, scale)
        xp, res = self.decode(lat, valid, mod_idx, sig, ds)
        L = {"rec": self.rec_loss(self.apply_prep(xp, valid, sig), x, valid, sig)}
        L["res"] = (res ** 2).mean()
        L["ent"] = lat["w"].sum(1).mean()
        if cfg.use_invariance:
            xa, w = self.augment(x, valid, sig, gen)
            lat2 = self.encode(xa, valid, mod, sig, scale)
            xp2, res2 = self.decode(lat2, valid, mod_idx, sig, ds)
            L["rec_aug"] = self.rec_loss(self.apply_prep(xp2, valid, sig), xa, valid, sig)
            L["res"] = L["res"] + (res2 ** 2).mean()
            L["inv"] = ((lat2["c"] - lat["c"].detach()) ** 2).sum(1).mean()
            if cfg.use_instrument and cfg.use_physics:
                L["eq"] = (((lat2["dshift"] - lat["dshift"].detach() - w["shift"]) / cfg.aug["shift"]) ** 2).mean() + \
                          (((lat2["dsig"] - torch.sqrt(lat["dsig"].detach() ** 2 + w["sig"] ** 2)) / cfg.aug["sigma"]) ** 2).mean()
        L["inst"] = (self.ds_inst ** 2).mean()
        if cfg.use_instrument and cfg.use_physics:   # pin per-sample instrument deviations of clean spectra to ~0 (gauge)
            L["dinst"] = ((lat["dshift"] / cfg.aug["shift"]) ** 2 + (lat["dsig"] / cfg.aug["sigma"]) ** 2 + (lat["dstray"] / 0.0005) ** 2).mean()
        total = L["rec"] + L.get("rec_aug", 0) + cfg.w_res * L["res"] + cfg.w_ent * L["ent"] + \
            cfg.w_inv * L.get("inv", 0) + cfg.w_eq * L.get("eq", 0) + cfg.w_inst * L["inst"] + cfg.w_dinst * L.get("dinst", 0)
        return total, {k: float(v) for k, v in L.items()}


# ---------------------------------------------------------------------------- data + training
class Corpus:
    """Spectra on the global grid, grouped by dataset, for SSL training."""

    def __init__(self):
        self.items = []

    def add(self, name, Xg, valid, modality, signal, ds_id, weight=1.0):
        Xg = torch.tensor(np.asarray(Xg, np.float32))
        v = Xg[:, torch.tensor(valid)]
        scale = float(v.std()) + 1e-8
        self.items.append(dict(name=name, X=Xg, valid=torch.tensor(valid), mod=MODS.index(modality),
                               sig=SIGNALS.index(signal), ds=ds_id, scale=scale, n=len(Xg), w=weight))

    def batch(self, mod_idx, B, rng):
        its = [it for it in self.items if it["mod"] == mod_idx]
        p = np.array([math.sqrt(it["n"]) * it["w"] for it in its])
        p /= p.sum()
        which = rng.choice(len(its), B, p=p)
        xs, vs, sg, ds, sc = [], [], [], [], []
        for k in which:
            it = its[k]
            j = rng.integers(it["n"])
            xs.append(it["X"][j]); vs.append(it["valid"]); sg.append(it["sig"]); ds.append(it["ds"]); sc.append(it["scale"])
        return (torch.stack(xs), torch.stack(vs), torch.tensor(sg), torch.tensor(ds), torch.tensor(sc, dtype=torch.float32))


def train_ssl(model: SpecTwin, corpus: Corpus, steps=3000, B=64, lr=1e-3, seed=0, params=None, log_every=250, log=print):
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    params = list(model.parameters()) if params is None else params
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    warm = min(200, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    mods = sorted({it["mod"] for it in corpus.items})
    mp = np.array([sum(math.sqrt(it["n"]) for it in corpus.items if it["mod"] == m) for m in mods])
    mp /= mp.sum()
    model.train()
    hist = []
    for step in range(steps):
        m = mods[rng.choice(len(mods), p=mp)]
        x, v, sg, ds, sc = corpus.batch(m, B, rng)
        loss, parts = model.ssl_step(x, v, m, sg, ds, sc, gen)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        hist.append(parts)
        if log and (step % log_every == 0 or step == steps - 1):
            avg = {k: np.mean([h.get(k, np.nan) for h in hist[-log_every:]]) for k in parts}
            log(f"step {step:5d} " + " ".join(f"{k}={v:.4f}" for k, v in avg.items()))
    model.eval()
    return hist


@torch.no_grad()
def encode_np(model: SpecTwin, Xg, valid, modality, signal, ds_id, scale, bs=256):
    """Latents + canonical spectra for a set of spectra (eval mode)."""
    model.eval()
    mi = MODS.index(modality)
    out = {k: [] for k in ["c", "w", "logb", "a", "gam", "u", "dshift", "dsig", "dstray", "mu", "sd", "h", "canon", "chem", "recon_err"]}
    V = torch.tensor(valid)
    for i in range(0, len(Xg), bs):
        x = torch.tensor(np.asarray(Xg[i:i + bs], np.float32))
        B = len(x)
        v = V[None, :].expand(B, -1)
        sg = torch.full((B,), SIGNALS.index(signal))
        ds = torch.full((B,), ds_id)
        lat = model.encode(x, v, torch.full((B,), mi), sg, torch.full((B,), float(scale)))
        xp, _ = model.decode(lat, v, mi, sg, ds)
        xp = model.apply_prep(xp, v, sg)
        raw = signal in ("absorbance", "raman")
        if raw:
            err = (((xp - x) / lat["sd"][:, None]) ** 2 * v.float()).sum(1) / v.float().sum(1)
        else:
            err = ((masked_snv(xp, v) - masked_snv(x, v)) ** 2 * v.float()).sum(1) / v.float().sum(1)
        if model.cfg.use_physics:
            canon, chem = model.canonical_std(lat, mi)
        else:
            canon = chem = lat["h"]
        for k in ["c", "w", "logb", "a", "gam", "u", "dshift", "dsig", "dstray", "mu", "sd", "h"]:
            out[k].append(lat[k].numpy())
        out["canon"].append(canon.numpy())
        out["chem"].append(chem.numpy())
        out["recon_err"].append(err.numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def adapt(model: SpecTwin, Xg, valid, modality, signal, ds_id, steps=300, lr=3e-4, seed=0, B=64, freeze_dict=True, log=None):
    """Unlabelled self-supervised adaptation to one dataset (training spectra only). Returns a new model."""
    m = copy.deepcopy(model)
    c = Corpus()
    c.add("adapt", Xg, valid, modality, signal, ds_id)
    params = [p for n, p in m.named_parameters() if not (freeze_dict and n in ("E_raw", "R"))]
    train_ssl(m, c, steps=steps, B=B, lr=lr, seed=seed, params=params, log=log, log_every=max(steps, 1))
    return m
