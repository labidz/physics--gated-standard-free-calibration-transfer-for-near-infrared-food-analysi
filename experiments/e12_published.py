"""Published-protocol comparison: our runs under the Kennard-Stone protocols of CTAI (Zhao et al., Molecules
2019, 24:1802) next to the RMSEPs reported in that paper (its Tables for corn m5->mp6 and IDRC-2016 wheat)."""
import json, os
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]

PUBLISHED = {  # RMSEP as reported by Zhao et al. 2019 (CTAI paper)
    "corn KS m5->mp6 moisture": dict(CTAI=0.211, PDS=0.242, SBC=0.426, TCR=0.391, CCA=0.233, MSC=1.669),
    "corn KS m5->mp6 oil": dict(CTAI=0.082, PDS=0.095, SBC=0.084, TCR=0.142, CCA=0.114, MSC=1.232),
    "corn KS m5->mp6 protein": dict(CTAI=0.166, PDS=0.282, SBC=0.191, TCR=0.352, CCA=0.395, MSC=1.801),
    "corn KS m5->mp6 starch": dict(CTAI=0.347, PDS=0.790, SBC=0.370, TCR=0.799, CCA=0.857, MSC=1.931),
    "IDRC KS B1->B2": dict(CTAI=0.414, TCR=0.869, SBC=0.468, PDS=4.090),
    "IDRC KS B1->B3": dict(CTAI=0.682, TCR=0.730, SBC=0.793, PDS=1.335),
    "IDRC KS B3->B2": dict(CTAI=0.384, TCR=0.637, SBC=0.535, PDS=1.836),
}
PRIMARY = os.environ.get("E12_PRIMARY", "TTPA4-safe-abstain")
OURS = ["NC", "DMC", "diPLS", "uDOP", "TCR", "CT-VPdtw", PRIMARY, "PDS (std)", "SBC (std)"]

if __name__ == "__main__":
    lines = ["| Scenario | " + " | ".join(OURS) + " | CTAI (published) | best other published |", "|---|" + "---|" * (len(OURS) + 2)]
    for f in ["Published protocol corn m5->mp6 (KS 64-16)", "Published protocol IDRC-2016 (KS 198-50)"]:
        p = ROOT / "results" / "e12" / "conf" / f"{f}.json"
        if not p.exists():
            p = next((ROOT / "results" / "e12" / "conf").glob(f[:30] + "*"), None)
        if p is None or not p.exists():
            continue
        for r in json.loads(p.read_text()):
            m = {k: v["RMSEP"] for k, v in r["metrics"].items() if v}
            pub = PUBLISHED.get(r["scenario"], {})
            other = {k: v for k, v in pub.items() if k != "CTAI"}
            bo = min(other, key=other.get) if other else None
            lines.append(f"| {r['scenario']} | " + " | ".join(f"{m[k]:.3f}" if k in m else "–" for k in OURS) +
                         f" | {pub.get('CTAI', float('nan')):.3f} | {bo} {other.get(bo, float('nan')):.3f} |")
    txt = "\n".join(lines)
    (ROOT / "results" / "tables" / "e12_published_protocols.md").write_text(txt + "\n")
    print(txt)
