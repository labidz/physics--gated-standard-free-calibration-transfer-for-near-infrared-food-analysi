"""E6: grid / resolution invariance and E10: engineering (size, latency, cost model)."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
torch.set_num_threads(1)
from spectwin.bench import load
from spectwin.baselines import pls_tune
from spectwin.chemo import PLS, reg_metrics
from spectwin.prep import Prep
from spectwin.grid import to_global
from spectwin.twin import encode_np
from spectwin.twin_run import load_twin, ds_ids, scale_of, corrected_global
from spectwin.deep import train_net

OUT = ROOT / "results" / "e6e10"; OUT.mkdir(parents=True, exist_ok=True)


def degrade(X, axis, mode):
    if mode == "full":
        return X, axis
    if mode.startswith("x"):
        k = int(mode[1:])
        return X[:, ::k], axis[::k]
    if mode == "trim20":   # instrument with a 20% narrower range (10% cut at each end)
        p = len(axis); a, b = int(0.1 * p), int(0.9 * p)
        return X[:, a:b], axis[a:b]
    raise ValueError(mode)


def to_native(Xd, axd, axis):
    return np.stack([np.interp(axis, axd, x) for x in Xd])


def e6():
    tasks = [("mango", "DM"), ("ch2018", "protein"), ("pears", "Brix"), ("cookie", "fat"), ("sorghum", "CP"), ("tecator", "fat")]
    modes = ["full", "x2", "x4", "x8", "trim20"]
    res = []
    for name, target in tasks:
        d = load(name); y = d.targets[target].astype(float); ok = np.isfinite(y)
        tr = np.where(ok & (d.split == "train"))[0]; te = np.where(ok & (d.split == "test"))[0]
        X = d.X.astype(float)
        hp = pls_tune(X[tr], y[tr], d.groups[tr], "reg")
        P = Prep(hp["prep"]).fit(X[tr]); pls = PLS(hp["k"]).fit(P.transform(X[tr]), y[tr])
        cnn, _ = train_net("cnn", X[tr], y[tr], d.groups[tr], "reg", seed=0)
        m = load_twin(d.modality); did = ds_ids()[name]
        Xg, valid = to_global(X[tr], d.axis, d.axis_unit, d.modality); sc = scale_of(Xg, valid)
        Ctr = corrected_global(Xg, encode_np(m, Xg, valid, d.modality, d.signal, did, sc), valid, d.modality, d.signal)
        hp2 = pls_tune(Ctr, y[tr], d.groups[tr], "reg"); P2 = Prep(hp2["prep"]).fit(Ctr); tw = PLS(hp2["k"]).fit(P2.transform(Ctr), y[tr])
        base_pred = {}
        for mode in modes:
            Xd, axd = degrade(X[te], d.axis, mode)
            Xn = to_native(Xd, axd, d.axis)                       # baselines need the native grid back
            p_pls = pls.predict(P.transform(Xn), hp["k"])
            p_cnn = cnn(Xn)
            Xgd, vd = to_global(Xd, axd, d.axis_unit, d.modality)   # SpecTwin reads any grid directly
            lat = encode_np(m, Xgd, vd, d.modality, d.signal, did, sc)
            Cv = corrected_global(Xgd, lat, vd, d.modality, d.signal)        # corrected values on the degraded grid's points
            gi_tr, gi_d = np.where(valid)[0], np.where(vd)[0]
            Cte = np.stack([np.interp(gi_tr, gi_d, row) for row in Cv])     # read onto the training global points
            p_tw = tw.predict(P2.transform(Cte), hp2["k"])
            for meth, p in [("PLS (interpolated to native grid)", p_pls), ("CNN1D (interpolated)", p_cnn), ("SpecTwin-PLS (native read)", p_tw)]:
                if mode == "full":
                    base_pred[meth] = p
                r = reg_metrics(y[te], p)
                res.append(dict(task=f"{name}:{target}", mode=mode, method=meth, RMSEP=r["RMSEP"],
                                pred_rmsd_vs_full=float(np.sqrt(np.mean((p - base_pred[meth]) ** 2))), n_points=int(len(axd))))
        print("E6", name, [(r["mode"], r["method"][:8], round(r["RMSEP"], 4)) for r in res if r["task"] == f"{name}:{target}"], flush=True)
        (OUT / "e6.json").write_text(json.dumps(res, indent=1))


def e10():
    out = {}
    m = load_twin("vnir")
    ck = torch.load(ROOT / "results/models/twin_vnir_real.pt", weights_only=False)
    out["twin_params"] = int(sum(p.numel() for p in m.parameters()))
    out["twin_pretrain_minutes_2core_shared"] = float(ck["seconds"] / 60)
    d = load("ch2021"); Xg, valid = to_global(d.X[:512], d.axis, d.axis_unit, "vnir"); sc = scale_of(Xg, valid)
    for bs in [1, 256]:
        t = time.time(); n = 0
        while time.time() - t < 5:
            encode_np(m, Xg[:bs], valid, "vnir", "absorbance", ds_ids()["ch2021"], sc, bs=bs); n += bs
        out[f"twin_ms_per_spectrum_batch{bs}"] = 1000 * (time.time() - t) / n
    X = d.X.astype(float); y = d.targets["y"]; tr = np.where(d.split == "train")[0]
    hp = pls_tune(X[tr], y[tr], d.groups[tr], "reg"); P = Prep(hp["prep"]).fit(X[tr]); pls = PLS(hp["k"]).fit(P.transform(X[tr]), y[tr])
    t = time.time(); [pls.predict(P.transform(X[:1]), hp["k"]) for _ in range(2000)]; out["pls_ms_per_spectrum"] = 1000 * (time.time() - t) / 2000
    from spectwin.transfer import TTPA
    role, inst = d.domains["role"], d.domains["instrument"]
    A = np.where((role == "std") & (inst == "4"))[0]
    t = time.time(); tt = TTPA("fleet").fit_source(X[tr], d.axis, k_min=hp["k"] + 5).fit_target(X[A]); out["ttpa_fit_seconds_ch2021"] = time.time() - t
    # parametric recalibration cost avoided for a fleet (no claim about actual prices)
    out["cost_model"] = {f"units={M}, refs/unit={N}, cost/ref={c}": M * N * c for M in [7] for N in [20, 50, 100] for c in [10, 30, 100]}
    (OUT / "e10.json").write_text(json.dumps(out, indent=1))
    print("E10", {k: v for k, v in out.items() if k != "cost_model"}, flush=True)


if __name__ == "__main__":
    which = sys.argv[1].split(",") if len(sys.argv) > 1 else ["e6", "e10"]
    if "e6" in which: e6()
    if "e10" in which: e10()
