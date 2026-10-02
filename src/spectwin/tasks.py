"""Task registry and outer evaluation splits (as frozen in PROTOCOL.md)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bench import load
from .chemo import group_kfold


@dataclass
class Task:
    dataset: str
    target: str
    kind: str                 # 'reg' | 'cls'
    subset: dict | None = None  # domain filter, e.g. {'instrument': 'm5'}

    @property
    def name(self):
        s = f"{self.dataset}:{self.target}"
        if self.subset:
            s += "[" + ",".join(f"{k}={v}" for k, v in self.subset.items()) + "]"
        return s


E1_TASKS = [
    Task("mango", "DM", "reg"),
    *[Task("plum", t, "reg") for t in ["SSC", "pH", "SRF", "MFF", "LS"]],
    Task("ch2021", "y", "reg"),
    Task("ch2018", "protein", "reg"),
    *[Task("corn", t, "reg", {"instrument": "m5"}) for t in ["moisture", "oil", "protein", "starch"]],
    Task("wheat", "protein", "reg"),
    Task("cassava", "TBC", "reg"),
    *[Task("sorghum", t, "reg") for t in ["CP", "NDF", "ADF"]],
    *[Task("forages", t, "reg") for t in ["DM", "NDF"]],
    Task("pears", "Brix", "reg"),
    *[Task("tecator", t, "reg") for t in ["moisture", "fat", "protein"]],
    *[Task("cookie", t, "reg") for t in ["fat", "sucrose", "dry_flour", "water"]],
    *[Task("bioethanol", t, "reg") for t in ["glucose", "ethanol"]],
    Task("fishoil", "iodine", "reg"),
    *[Task("emulsions_nir", t, "reg") for t in ["PUFAtot", "PUFAfat"]],
    *[Task("emulsions_raman", t, "reg") for t in ["PUFAtot", "PUFAfat"]],
    Task("ch2018", "product", "cls"),
    Task("grapes", "variety", "cls"),
    Task("mayonnaise", "oil", "cls"),
]


def task_data(task: Task):
    """Return (ds, idx) where idx are rows usable for this task (finite target, subset filter)."""
    ds = load(task.dataset)
    y = ds.targets[task.target]
    ok = np.ones(len(y), bool)
    if task.kind == "reg":
        ok &= np.isfinite(y.astype(float))
    else:
        ok &= (y.astype(str) != "") & (y.astype(str) != "nan")
    if task.subset:
        for k, v in task.subset.items():
            ok &= ds.domains[k] == v
    return ds, np.where(ok)[0]


def outer_folds(task: Task, ds=None, idx=None):
    """List of (train_rows, test_rows) as absolute row indices into the dataset."""
    if ds is None:
        ds, idx = task_data(task)
    split = ds.split[idx]
    if task.dataset == "plum":
        f = ds.domains["fold"][idx]
        return [(idx[f != k], idx[f == k]) for k in sorted(np.unique(f))]
    if (split == "train").any() and (split == "test").any():
        return [(idx[split == "train"], idx[split == "test"])]
    folds = []
    for rep in range(2):
        for tr, te in group_kfold(ds.groups[idx], 5, seed=100 + rep):
            folds.append((idx[tr], idx[te]))
    return folds
