"""Spectral preprocessing (sklearn-style, fitted on training data only)."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def snv(X):
    X = np.asarray(X, float)
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True)
    return (X - mu) / np.where(sd < 1e-12, 1.0, sd)


def sg(X, deriv, window=None, poly=2):
    p = X.shape[1]
    if window is None:
        window = 11 if deriv == 1 else 15
    window = min(window, p - (1 - p % 2))  # must be odd and <= p
    if window % 2 == 0:
        window -= 1
    window = max(window, poly + 2 + (poly % 2 == 0))
    if window % 2 == 0:
        window += 1
    return savgol_filter(np.asarray(X, float), window, poly, deriv=deriv, axis=1)


class Prep:
    """Named preprocessing pipeline. MSC reference is learned in fit()."""

    NAMES = ["raw", "snv", "msc", "sg1", "sg2", "snv_sg1", "snv_sg2"]

    def __init__(self, name="raw"):
        assert name in self.NAMES, name
        self.name = name
        self.ref_ = None

    def fit(self, X):
        if self.name == "msc":
            self.ref_ = np.asarray(X, float).mean(0)
        return self

    def _msc(self, X):
        out = np.empty_like(X, dtype=float)
        r = self.ref_ - self.ref_.mean()
        denom = (r ** 2).sum()
        for i, x in enumerate(X):
            b = ((x - x.mean()) * r).sum() / denom
            a = x.mean() - b * self.ref_.mean()
            out[i] = (x - a) / (b if abs(b) > 1e-12 else 1.0)
        return out

    def transform(self, X):
        X = np.asarray(X, float)
        n = self.name
        if n == "raw":
            return X
        if n == "snv":
            return snv(X)
        if n == "msc":
            return self._msc(X)
        if n == "sg1":
            return sg(X, 1)
        if n == "sg2":
            return sg(X, 2)
        if n == "snv_sg1":
            return sg(snv(X), 1)
        if n == "snv_sg2":
            return sg(snv(X), 2)
        raise ValueError(n)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
