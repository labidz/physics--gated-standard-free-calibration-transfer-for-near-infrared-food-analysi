"""Pretrain SpecTwin on the inductive corpus (training partitions only) + optional SimFood-RT.

usage: python pretrain_twin.py <variant> [steps] [key=value ...]
  variants share the dataset-id map in results/models/ds_ids.json
  options: modality=vnir|raman|mir  sim=0|1  frac=1.0  use_physics=1 use_instrument=1 use_scatter=1
           use_prep_op=1 use_invariance=1 food_only=1 tokens=wavenumber
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spectwin.bench import load
from spectwin.grid import to_global
from spectwin.twin import SpecTwin, TwinConfig, Corpus, train_ssl

torch.set_num_threads(int(__import__("os").environ.get("TWIN_THREADS", "1")))
MODELS = ROOT / "results" / "models"; MODELS.mkdir(parents=True, exist_ok=True)

ALL = [c["name"] for c in json.loads((ROOT / "data/bench/registry.json").read_text())]
IDS_FILE = MODELS / "ds_ids.json"
if IDS_FILE.exists():
    DS_IDS = json.loads(IDS_FILE.read_text())
else:
    DS_IDS = {n: i for i, n in enumerate(ALL)}
    DS_IDS.update({f"sim_{k}": 40 + k for k in range(16)})
    IDS_FILE.write_text(json.dumps(DS_IDS, indent=1))

# Inductive corpus: datasets with an official split contribute their TRAINING partition only.
# Datasets evaluated by cross-validation (plum, corn, bioethanol, emulsions, fishoil) are excluded
# entirely (they are "unseen" and adapted per fold on training folds only).
PRETRAIN = {"vnir": ["mango", "ch2021", "ch2018", "wheat", "cassava", "sorghum", "forages", "pears", "grapes",
                     "tecator", "cookie", "mayonnaise", "oregano"],
            "raman": ["carbs", "minerals"],
            "mir": ["ligno"]}
NONFOOD = {"vnir": ["tablets"]}


def corpus_rows(ds, name):
    if name == "ch2021":
        return np.where((ds.split == "train") | (ds.domains["role"] == "std"))[0]
    if name == "ligno":   # off-line standards + first 10% of the on-line run (adaptation window)
        on = np.where(ds.domains["role"] == "online")[0]
        return np.concatenate([np.where(ds.domains["role"] == "offline")[0], on[: int(0.1 * len(on))]])
    if name in ("carbs", "minerals"):
        return np.arange(len(ds.X))
    if name == "tablets":
        return np.where(ds.split == "train")[0]
    return np.where(ds.split == "train")[0]


def build_corpus(modality, frac=1.0, sim=False, food_only=True, seed=0):
    rng = np.random.default_rng(seed)
    c = Corpus(); stats = {}
    names = PRETRAIN[modality] + ([] if food_only else NONFOOD.get(modality, []))
    for n in names:
        ds = load(n); rows = corpus_rows(ds, n)
        if frac < 1.0:
            rows = np.sort(rng.choice(rows, max(8, int(frac * len(rows))), replace=False))
        Xg, valid = to_global(ds.X[rows], ds.axis, ds.axis_unit, ds.modality)
        c.add(n, Xg, valid, ds.modality, ds.signal, DS_IDS[n]); stats[n] = int(len(rows))
    if sim:
        simdir = ROOT / "data" / "sim"
        for f in sorted(simdir.glob(f"simfood_{modality}_unit*.npz")):
            z = np.load(f)
            k = int(z["unit_id"]); rows = np.where(z["split"] == "train")[0]
            if frac < 1.0:
                rows = np.sort(rng.choice(rows, max(8, int(frac * len(rows))), replace=False))
            Xg, valid = to_global(z["X"][rows], z["axis"], str(z["axis_unit"]), modality)
            c.add(f"sim_{k}", Xg, valid, modality, "absorbance", DS_IDS[f"sim_{k}"], weight=0.5)
            stats[f"sim_{k}"] = int(len(rows))
    return c, stats


if __name__ == "__main__":
    variant = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    opts = dict(a.split("=") for a in sys.argv[3:])
    modality = opts.get("modality", "vnir")
    cfg = TwinConfig(**{k: bool(int(opts[k])) for k in ["use_physics", "use_instrument", "use_scatter", "use_prep_op", "use_invariance"] if k in opts})
    corpus, stats = build_corpus(modality, float(opts.get("frac", 1.0)), bool(int(opts.get("sim", 0))), bool(int(opts.get("food_only", 1))))
    print(f"variant={variant} modality={modality} steps={steps} corpus={stats} total={sum(stats.values())}", flush=True)
    model = SpecTwin(cfg)
    t0 = time.time()
    hist = train_ssl(model, corpus, steps=steps, B=64, lr=1e-3, seed=0, log_every=250, log=lambda s: print(s, flush=True))
    torch.save(dict(state=model.state_dict(), cfg=cfg.__dict__, corpus=stats, modality=modality, seconds=time.time() - t0,
                    hist=hist[-500:]), MODELS / f"twin_{modality}_{variant}.pt")
    print(f"saved twin_{modality}_{variant}.pt in {(time.time()-t0)/60:.1f} min", flush=True)
