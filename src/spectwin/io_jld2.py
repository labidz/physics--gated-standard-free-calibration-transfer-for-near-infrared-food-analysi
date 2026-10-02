"""Minimal reader for JLD2 files (HDF5-compatible) written by Julia (e.g. JchemoData.jl).

Converts Julia DataFrames to pandas DataFrames, CategoricalArrays to object arrays,
numeric arrays to numpy and strings to str. Only what is needed to read spectral
datasets is supported; anything unrecognised is returned as a dict of its fields.
"""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd


def _decode(x):
    if isinstance(x, (bytes, np.bytes_)):
        return bytes(x).decode("utf-8", errors="replace")
    return x


def _as_list(x):
    """Flatten (possibly nested) object arrays coming from HDF5 vlen fields."""
    if not isinstance(x, np.ndarray):
        return [x]
    out = []
    for e in x.ravel():
        if isinstance(e, np.ndarray) and e.dtype == object:
            out.extend(_as_list(e))
        elif isinstance(e, np.ndarray) and e.ndim > 0 and e.dtype.kind in "SU":
            out.extend(list(e.ravel()))
        else:
            out.append(e)
    return out


def _deref(f: h5py.File, x):
    if isinstance(x, h5py.Reference):
        if not x:
            return None
        tgt = f[x]
        if isinstance(tgt, h5py.Group):
            return tgt
        if tgt.shape is None:  # empty dataspace
            return None
        return tgt[()]
    return x


def _struct(f: h5py.File, rec):
    names = rec.dtype.names
    fields = set(names)
    if {"columns", "colindex"} <= fields:  # Julia DataFrame
        colindex = _deref(f, rec["colindex"])
        cols = [_convert(f, r) for r in _as_list(_deref(f, rec["columns"]))]
        colnames = [str(_convert(f, n)) for n in _as_list(_deref(f, colindex["names"]))]
        data = {}
        for n, c in zip(colnames, cols):
            if isinstance(c, np.ndarray) and c.ndim == 1:
                data[n] = c
            elif isinstance(c, list):
                data[n] = np.array(c, dtype=object)
            else:
                data[n] = np.atleast_1d(np.asarray(c, dtype=object))
        return pd.DataFrame(data)
    if {"refs", "pool"} <= fields:  # CategoricalArray
        refs = np.asarray(_convert(f, rec["refs"])).astype(int).ravel()
        pool = _convert(f, rec["pool"])
        levels = None
        if isinstance(pool, dict):
            levels = pool.get("levels")
        if levels is None:
            return refs
        levels = list(np.atleast_1d(levels))
        return np.array([levels[r - 1] if r > 0 else None for r in refs], dtype=object)
    out = {}
    for n in names:
        try:
            out[n] = _convert(f, rec[n])
        except Exception as e:  # keep going; record the problem
            out[n] = f"<unreadable: {e!r}>"
    return out


def _convert(f: h5py.File, obj):
    """Recursively convert an HDF5 value (possibly a reference) to Python."""
    if isinstance(obj, h5py.Reference):
        tgt = _deref(f, obj)
        if tgt is None:
            return None
        if isinstance(tgt, h5py.Group):
            return {k: _convert(f, tgt[k][()]) for k in tgt.keys()}
        return _convert(f, tgt)
    if isinstance(obj, (bytes, np.bytes_)):
        return _decode(obj)
    if isinstance(obj, np.void) or (isinstance(obj, np.ndarray) and obj.dtype.names and obj.shape == ()):
        return _struct(f, obj[()] if isinstance(obj, np.ndarray) else obj)
    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            return [_convert(f, o) for o in obj.ravel()]
        if obj.dtype == object:
            out = [_convert(f, o) for o in obj.ravel()]
            try:
                arr = np.array(out)
                return arr.reshape(obj.shape) if arr.size == obj.size else out
            except Exception:
                return out
        if obj.dtype.kind == "S":
            return np.char.decode(obj, "utf-8")
        return obj
    return obj


def read_jld2(path: str) -> dict:
    """Read all top-level (non-internal) entries of a JLD2 file."""
    with h5py.File(path, "r") as f:
        return {k: _convert(f, f[k][()]) for k in f.keys() if not k.startswith("_")}
