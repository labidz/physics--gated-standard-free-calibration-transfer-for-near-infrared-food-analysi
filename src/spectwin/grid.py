"""Global physical grids and grid-agnostic resampling (quadrature) operators.

Each modality has one global axis. Any native spectrum is mapped onto it by a fixed linear
operator W (G x p): bin-averaging where the native grid is denser than the global cell
(a quadrature of the cell integral) and linear interpolation where it is coarser. Global
points outside the measured range are marked invalid (mask = 0).
"""
from __future__ import annotations

import numpy as np

GRIDS = {
    "vnir": dict(lo=300.0, hi=2500.0, G=1024, kind="log", unit="nm"),
    "mir": dict(lo=400.0, hi=4000.0, G=1024, kind="log", unit="cm-1"),
    "raman": dict(lo=150.0, hi=3300.0, G=1024, kind="lin", unit="raman_shift"),
}


def global_axis(modality: str) -> np.ndarray:
    g = GRIDS[modality]
    if g["kind"] == "log":
        return np.exp(np.linspace(np.log(g["lo"]), np.log(g["hi"]), g["G"]))
    return np.linspace(g["lo"], g["hi"], g["G"])


def native_to_modality_axis(axis: np.ndarray, axis_unit: str, modality: str) -> np.ndarray:
    """Express a native axis in the unit of the modality's global grid."""
    target = GRIDS[modality]["unit"]
    if axis_unit == target:
        return np.asarray(axis, float)
    if axis_unit == "cm-1" and target == "nm":
        return 1e7 / np.asarray(axis, float)
    if axis_unit == "nm" and target == "cm-1":
        return 1e7 / np.asarray(axis, float)
    raise ValueError((axis_unit, modality))


def resample_operator(axis: np.ndarray, modality: str):
    """Return (W, valid) with W of shape (G, p) such that x_global = W @ x_native."""
    a = np.asarray(axis, float)
    order = np.argsort(a)
    a_sorted = a[order]
    g = global_axis(modality)
    G = len(g)
    if GRIDS[modality]["kind"] == "log":
        lg = np.log(g)
        mid = np.exp((lg[1:] + lg[:-1]) / 2)
    else:
        mid = (g[1:] + g[:-1]) / 2
    edges = np.concatenate([[g[0] - (mid[0] - g[0])], mid, [g[-1] + (g[-1] - mid[-1])]])
    W = np.zeros((G, len(a)))
    valid = np.zeros(G, bool)
    lo, hi = a_sorted[0], a_sorted[-1]
    for j in range(G):
        inside = np.where((a_sorted >= edges[j]) & (a_sorted < edges[j + 1]))[0]
        if len(inside) > 0:
            W[j, order[inside]] = 1.0 / len(inside)
            valid[j] = True
        elif lo <= g[j] <= hi:
            k = np.searchsorted(a_sorted, g[j])
            k0, k1 = max(k - 1, 0), min(k, len(a_sorted) - 1)
            if k0 == k1:
                W[j, order[k0]] = 1.0
            else:
                w = (g[j] - a_sorted[k0]) / (a_sorted[k1] - a_sorted[k0])
                W[j, order[k0]] = 1 - w
                W[j, order[k1]] = w
            valid[j] = True
    return W, valid


def to_global(X: np.ndarray, axis: np.ndarray, axis_unit: str, modality: str):
    """Map native spectra (n, p) to the global grid: returns (Xg (n, G), valid (G,))."""
    ax = native_to_modality_axis(axis, axis_unit, modality)
    W, valid = resample_operator(ax, modality)
    Xg = np.asarray(X, float) @ W.T
    Xg[:, ~valid] = 0.0
    return Xg.astype(np.float32), valid
