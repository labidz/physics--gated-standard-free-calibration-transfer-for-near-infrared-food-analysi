"""Differentiable measurement-physics operators on the global grid (torch)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import savgol_coeffs


def shift_broaden(A: torch.Tensor, shift: torch.Tensor, sigma: torch.Tensor, pad: int = 64) -> torch.Tensor:
    """Fractional axis shift and Gaussian line-shape broadening (instrument operator), via FFT.

    A: (B, G); shift, sigma: (B,) in global-grid units. Positive shift moves features to higher index.
    """
    B, G = A.shape
    Ap = F.pad(A[:, None, :], (pad, pad), mode="reflect")[:, 0]
    L = Ap.shape[1]
    f = torch.fft.rfftfreq(L, device=A.device, dtype=A.dtype)
    H = torch.exp(-2 * math.pi ** 2 * sigma[:, None] ** 2 * f[None, :] ** 2)
    phase = -2 * math.pi * f[None, :] * shift[:, None]
    spec = torch.fft.rfft(Ap)
    spec = spec * torch.complex(H * torch.cos(phase), H * torch.sin(phase))
    out = torch.fft.irfft(spec, n=L)
    return out[:, pad:pad + G]


def stray_light(A: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Apparent absorbance with stray-light fraction s: -log10((10^-A + s) / (1 + s))."""
    s = s[:, None]
    return -torch.log10((torch.pow(10.0, -A.clamp(-3, 6)) + s) / (1 + s))


def legendre_basis(valid: torch.Tensor, P: int) -> torch.Tensor:
    """Legendre polynomials P1..P_P over each spectrum's valid range. valid: (B, G) bool -> (B, P, G)."""
    B, G = valid.shape
    idx = torch.arange(G, dtype=torch.float32, device=valid.device)[None, :].expand(B, G)
    big = torch.full_like(idx, 1e9)
    jmin = torch.where(valid, idx, big).min(1).values
    jmax = torch.where(valid, idx, -big).max(1).values
    t = (2 * (idx - jmin[:, None]) / (jmax - jmin).clamp_min(1)[:, None] - 1).clamp(-1, 1)
    polys = [t]
    if P >= 2:
        polys.append(0.5 * (3 * t ** 2 - 1))
    if P >= 3:
        polys.append(0.5 * (5 * t ** 3 - 3 * t))
    p_prev2, p_prev1 = polys[-2] if len(polys) > 1 else torch.ones_like(t), polys[-1]
    for n in range(4, P + 1):  # Bonnet recursion
        pn = ((2 * n - 1) * t * p_prev1 - (n - 1) * p_prev2) / n
        polys.append(pn)
        p_prev2, p_prev1 = p_prev1, pn
    return torch.stack(polys[:P], 1)


def masked_snv(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    v = valid.float()
    n = v.sum(1, keepdim=True).clamp_min(1)
    mu = (x * v).sum(1, keepdim=True) / n
    sd = (((x - mu) ** 2) * v).sum(1, keepdim=True).div(n).clamp_min(1e-16).sqrt()
    return (x - mu) / sd * v


class SGConv(torch.nn.Module):
    """Savitzky-Golay derivative as a fixed convolution on the global grid."""

    def __init__(self, deriv: int, window: int = 9, poly: int = 2):
        super().__init__()
        k = savgol_coeffs(window, poly, deriv=deriv, use="conv")
        self.register_buffer("k", torch.tensor(k[::-1].copy(), dtype=torch.float32)[None, None, :])
        self.pad = window // 2

    def forward(self, x):
        return F.conv1d(F.pad(x[:, None, :], (self.pad, self.pad), mode="replicate"), self.k)[:, 0]


def gaussian_smoother(G: int, sigma: float) -> torch.Tensor:
    """Fixed (G x G) Gaussian smoothing matrix used to keep dictionary atoms smooth."""
    i = np.arange(G)
    K = np.exp(-0.5 * ((i[:, None] - i[None, :]) / sigma) ** 2)
    K /= K.sum(1, keepdims=True)
    return torch.tensor(K, dtype=torch.float32)
