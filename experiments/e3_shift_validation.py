"""Physical validity of TTPA (added after the main analysis): unpaired TTPA wavelength-shift estimates on corn
vs the shift fitted from the 80 paired spectra under the same instrument operator (shift + warp + per-sample cubic baseline).
Caveat: corn was the TTPA development pilot (protocol deviation 6)."""
import sys, json, collections
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from spectwin.bench import load
from spectwin.transfer import instrument_op
d = load("corn"); inst = np.asarray(d.domains["instrument"]).astype(str); ax = d.axis.astype(float)
X = {i: d.X[inst == i].astype(float) for i in np.unique(inst)}
t = np.linspace(-1, 1, len(ax)); B = np.stack([t ** k for k in range(4)], 1); Pi = np.linalg.pinv(B).T


def paired_fit(Xs, Xt):
    def loss(th):
        R = instrument_op(Xs, ax, th[0], th[1], 0.0) - Xt; R = R - (R @ Pi) @ B.T
        return float(np.mean(R ** 2))
    return min((minimize(loss, [s, 0.0], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-14, maxiter=600))
                for s in [-2, -1, 0, 1, 2]), key=lambda r: r.fun).x


E = json.loads((ROOT / "results/e3/corn.json").read_text()); tt = collections.defaultdict(list)
for r in E:
    tt[r["pair"]].append(r["info"]["fleet"]["theta"][0])
rows, P, T = [], [], []
for pair in sorted(tt):
    a, b = pair.split("->"); th = paired_fit(X[a], X[b]); P.append(th[0]); T.append(np.mean(tt[pair]))
    rows.append(f"| {pair} | {th[0]:+.3f} | {np.mean(tt[pair]):+.3f} ± {np.std(tt[pair]):.3f} | {len(tt[pair])} |")
P, T = np.array(P), np.array(T); slope = float(np.polyfit(P, T, 1)[0])
txt = "\n".join(["| Corn pair | shift from paired spectra (nm) | TTPA unpaired estimate (nm, mean ± sd) | runs |", "|---|---|---|---|"] + rows +
                ["", f"Pearson r = {np.corrcoef(P, T)[0, 1]:.3f}; MAE = {np.abs(P - T).mean():.3f} nm (sampling step 2 nm); OLS slope = {slope:.2f}."])
(ROOT / "results/tables/e3_corn_shift_validation.md").write_text(txt + "\n"); print(txt)
