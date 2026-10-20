"""Builds T1_master.csv and T2_diagnostics.csv from the per-run result JSONs.
   python build_master.py spectwin/results/e12 .   (re-run after the Run A add-on to get MeanOnly_refit / NoOperator_refit rows)"""
import json, glob, os, re, sys
import numpy as np, pandas as pd
ROOT=sys.argv[1] if len(sys.argv)>1 else "spectwin/results/e12"
OUTD=sys.argv[2] if len(sys.argv)>2 else "."
LABEL={"NC":"NC","DMC":"DMC","CORAL":"CORAL","diPLS":"diPLS","uDOP":"uDOP","TCR":"TCA","OT-DA":"OTDA","DANN":"DANN",
"CT-VPdtw":"CTVPdtw","TTPA3":"TTPA_ungated","TTPA4-safe-abstain":"TTPA",
"TTPA4-safe":"abl_b","TTPA4-pargate-off":"abl_c","TTPA4":"abl_d","TTPA4-nobroad":"abl_e","TTPA4-perp":"abl_f","TTPA4-nogate":"abl_g",
"TTPA4-meanonly":"MeanOnly_refit","TTPA4-noop":"NoOperator_refit","TTPA4-nogof":"x_nogof","TTPA4-abstain":"x_abstain","TTPA4-broadtest":"x_broadtest","TTPA3-fleet":"x_ttpa3_fleet","TTPA3-chem":"x_ttpa3_chem",
"DS (std)":"DS_std","PDS (std)":"PDS_std","SBC (std)":"SBC_std","Oracle (target labels)":"Oracle"}
FAMNAME={"Cereals new species (population shift)":"cereals_new_species","Grain new harvest year (population shift)":"grain_new_harvest",
"ICRISAT cereals benchtop -> handheld":"cereals_handheld","IDRC-2016 wheat fleet A (same samples)":"wheat_fleet_A",
"IDRC-2016 wheat fleet B (same samples)":"wheat_fleet_B","IDRC-2016 wheat new units + new samples":"wheat_new_units",
"Sugarcane lab -> portable spectrometers":"sugarcane_portables","cassava new year":"cassava","ch2018 leave-one-product-out protein":"new_product",
"ch2021":"fruit_7_units","corn":"corn","grapes new harvest dates":"grapes","ligno":"bioprocess","mango fruit temperature low-mid -> high":"mango_temperature",
"mango season 1-3 -> 4":"mango_season","plum leave-one-cultivar-out SSC":"plum_cultivar","simfood":"simfood_rt","tablets":"tablets","wheat drift (2 months)":"wheat_drift"}
def pdiff(r):
    try:
        a=np.array(r["pred"]["TTPA4-safe-abstain"],float).ravel(); b=np.array(r["pred"]["NC"],float).ravel()
        return float(np.sqrt(np.mean((a-b)**2)))
    except Exception:
        return float('nan')
rows=[];diag=[];rid=0
for ph in ["dev","conf"]:
    for f in sorted(glob.glob(f"{ROOT}/{ph}/*.json")):
        stem=os.path.basename(f)[:-5]
        if stem.startswith("Published protocol"): continue
        fam=FAMNAME[stem]
        for r in json.load(open(f)):
            rid+=1; run_id=f"{rid:05d}"
            nS,nA,nT=r["n"][:3]
            base=dict(run_id=run_id,family_id=fam,family_type=r["context"],phase="confirm" if ph=="conf" else "dev",
                      scenario_id=r["scenario"],instrument_pair=str(r.get("pair","")),split_seed=r["rep"],n_S=nS,n_A=nA,n_T=nT,LV=r["hp"].get("k"),kind=r["kind"])
            for m,v in r["metrics"].items():
                if v is None: 
                    rows.append({**base,"method":LABEL.get(m,m),"rmsep":np.nan,"status":"failed","fail_reason":"metrics null"}); continue
                err=v["RMSEP"] if r["kind"]=="reg" else 1-v["accuracy"]
                rows.append({**base,"method":LABEL.get(m,m),"rmsep":err,"status":"ok","fail_reason":""})
            # methods that never produced a metrics key but are expected
            exp=["NC","DMC","CORAL","diPLS","uDOP","TCR","OT-DA","DANN","CT-VPdtw"]
            for m in exp:
                if m not in r["metrics"]:
                    rows.append({**base,"method":LABEL[m],"rmsep":np.nan,"status":"failed_or_na","fail_reason":"not applicable (classification) or failed; key absent"})
            t=r["info"]["TTPA4-safe-abstain"]
            diag.append(dict(run_id=run_id,family_id=fam,phase=base["phase"],context=r["context"],theta_gate=int(t["theta_gate"]),gof_ok=int(t.get("gof_ok",-1)),
                broad_gate=int(t["broad_gate"]),dperp_gate=int(t["dperp_gate"]),dpar_used=int(bool(t["dpar_used"])),
                s_hat=t["theta_used"][0] if t.get("theta_used") else np.nan,w_hat=t["theta_used"][1] if t.get("theta_used") else np.nan,sigma_hat=t["theta_used"][2] if t.get("theta_used") else np.nan,
                norm_dpar=t["norm_dpar"],norm_dperp=t["norm_dperp"],k=t["k"],
                pred_diff_vs_NC=pdiff(r)))
M=pd.DataFrame(rows); D=pd.DataFrame(diag)
M.to_csv(os.path.join(OUTD,"T1_master.csv"),index=False); D.to_csv(os.path.join(OUTD,"T2_diagnostics.csv"),index=False)
print(M.shape,D.shape); print(M.method.value_counts().to_dict()); print(M.status.value_counts().to_dict())
print(M[M.status!="ok"].groupby(["phase","family_id","method"]).size().to_string())
