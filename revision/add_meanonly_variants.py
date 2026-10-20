"""Adds two post-hoc TTPA variants (reviewer #2 / Run A). Run from the folder that contains `spectwin/`.
   python add_meanonly_variants.py [path/to/spectwin]
Variants (names used in the result JSONs):
  TTPA4-meanonly : operator OFF, out-of-span mean OFF, in-span mean ON in fleet context (always, as in the frozen config),
                   source calibration returned unchanged in chem context  ->  'MeanOnly_refit'
  TTPA4-noop     : exactly the frozen TTPA but with the operator forced OFF (out-of-span mean still gated) -> 'NoOperator_refit'
Both are post hoc: log them as a deviation in PROTOCOL.md."""
import sys, pathlib
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "spectwin")
t2 = root / "src/spectwin/transfer2.py"; r12 = root / "experiments/run_e12.py"
s = t2.read_text()
if "theta_off" in s: sys.exit("transfer2.py already patched")
old = 'def configure(self, gate=None, use_par=None, gate_par=None, broaden=None, gof="auto", chem_theta="test", chem_mean="perp"):'
assert old in s, "configure() signature not found"
s = s.replace(old, old[:-2] + ', theta_off=False, perp_off=False):')
a = "        theta_ok = theta_ok and gof_ok\n"; assert s.count(a) == 1
s = s.replace(a, a + "        if theta_off:\n            theta_ok = False\n")
lines = s.split("\n"); i = next(k for k, l in enumerate(lines) if l.strip().startswith("d_ok = par_ok = False"))
lines.insert(i + 1, "        if perp_off:\n            d_ok = False"); t2.write_text("\n".join(lines))
r = r12.read_text(); b = 'chem_mean="none", chem_theta="abstain")}'; assert r.count(b) == 1, "VARIANTS end not found"
new = ('chem_mean="none", chem_theta="abstain"),\n'
       '            # post hoc (reviewer #2): operator removed\n'
       '            "TTPA4-meanonly": dict(broaden="auto", gate_par=False, gof=True, chem_mean="none", chem_theta="abstain", theta_off=True, perp_off=True),\n'
       '            "TTPA4-noop": dict(broaden="auto", gate_par=False, gof=True, chem_mean="none", chem_theta="abstain", theta_off=True)}')
r12.write_text(r.replace(b, new)); print("patched OK")
