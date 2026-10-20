import numpy as np, pandas as pd, warnings, json, sys
warnings.filterwarnings("ignore")
import statsmodels.formula.api as smf
from scipy import stats
W="./"
M=pd.read_csv(W+"out2/T1_master.csv",dtype={"run_id":str}); D=pd.read_csv(W+"out2/T2_diagnostics.csv",dtype={"run_id":str})
CF=M[M.phase=="confirm"]
FAMS=["cereals_handheld","sugarcane_portables","wheat_fleet_A","wheat_fleet_B","wheat_new_units","grain_new_harvest","cereals_new_species"]
FN={"cereals_handheld":"Cereals handheld","sugarcane_portables":"Sugarcane portables","wheat_fleet_A":"Wheat fleet A","wheat_fleet_B":"Wheat fleet B","wheat_new_units":"Wheat new units","grain_new_harvest":"Grain new harvest","cereals_new_species":"Cereals new species"}
pc=lambda x:100*(np.exp(x)-1)
def P(x,d=1,sign=True):
    if x is None or (isinstance(x,float) and np.isnan(x)): return "--"
    x=round(float(x),d)
    if x==0: x=0.0
    s=("$+$" if x>0 else ("$-$" if x<0 else "")) if sign else ("$-$" if x<0 else "")
    return f"{s}{abs(x):.{d}f}\\%"
def CI(lo,hi,d=1): return f"[{P(lo,d)}, {P(hi,d)}]".replace("[","(").replace("]",")")
w=CF.pivot_table(index="run_id",columns="method",values="rmsep"); meta=M.drop_duplicates("run_id").set_index("run_id")[["family_id","scenario_id","family_type","phase"]]
J=w.join(meta); J["ctxDMC"]=np.where(J.family_type=="fleet",J.DMC,J.NC)
out=[]; NUM={}
def tab(caption,label,body,cols,note=None,small=True):
    s=["\\begin{table}[pos=!htbp]",f"\\caption{{{caption}}}\\label{{{label}}}","\\centering\\scriptsize\\revcolor","\\begin{adjustbox}{max width=\\linewidth}",f"\\begin{{tabular}}{{{cols}}}","\\toprule"]+body+["\\bottomrule","\\end{tabular}","\\end{adjustbox}"]
    if note: s+=["\\begin{flushleft}\\scriptsize",note,"\\end{flushleft}"]
    s.append("\\end{table}\n"); out.append("\n".join(s))
# ---------- D.1 family-level
body=["Family & NC & DMC & di-PLS & TCA & OT-DA & ctx-DMC\\\\","\\midrule"]
fam_tab={}
for f in FAMS:
    d=J[J.family_id==f]; row=[]
    for ref in ["NC","DMC","diPLS","TCA","OTDA","ctxDMC"]:
        v=pc(np.log(d.TTPA/d[ref]).mean()); row.append(v); fam_tab[(f,ref)]=v
    body.append(FN[f]+" & "+" & ".join(P(v) for v in row)+"\\\\")
body.append("\\midrule")
wl=[]
for ref in ["NC","DMC","diPLS","TCA","OTDA","ctxDMC"]:
    v=np.array([fam_tab[(f,ref)] for f in FAMS]); wl.append(f"{int((v<-0.1).sum())}/{int((v>0.1).sum())}/{int((abs(v)<=0.1).sum())}"); NUM[f"wlt_{ref}"]=wl[-1]
body.append("Better/worse/tied (${\\pm}0.1\\%$) & "+" & ".join(wl)+"\\\\")
tab("Family-level geometric-mean change in $\\rmsep$ of TTPA relative to each comparator (negative: TTPA better). ctx-DMC: context-conditioned DMC (DMC in the fleet context, no correction in the chem context).","tab:famlevel",body,"@{}lrrrrrr@{}")
# ---------- D.2 mixed models
def fit(d,re=False):
    d=d.rename(columns={"family_id":"family","scenario_id":"scenario"}); kw=dict(re_formula="1") if re else {}
    m=smf.mixedlm("lr ~ 1",d,groups="family",vc_formula={"scenario":"0 + C(scenario)"},**kw).fit(reml=True,method="lbfgs")
    e,se=m.params["Intercept"],m.bse["Intercept"]; vc={k:float(v*m.scale) for k,v in m.params.drop("Intercept").items()}
    return pc(e),pc(e-1.96*se),pc(e+1.96*se),vc,float(m.scale)
body=["Comparator & Runs & \\multicolumn{2}{c}{Primary model (scenario within family)} & \\multicolumn{2}{c}{With family intercept}\\\\","\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}"," & & Estimate & 95\\% CI & Estimate & 95\\% CI\\\\","\\midrule"]
LAB={"NC":"No correction","DMC":"DMC","CORAL":"CORAL","diPLS":"di-PLS","uDOP":"uDOP","TCA":"TCA","OTDA":"OT-DA","DANN":"DANN","CTVPdtw":"CT-VPdtw","ctxDMC":"ctx-DMC"}
VC={}
for ref in ["NC","DMC","ctxDMC","CORAL","diPLS","uDOP","TCA","OTDA","DANN","CTVPdtw"]:
    d=J[["TTPA",ref,"family_id","scenario_id"]].dropna().clip(lower=1e-12) if False else J[["TTPA",ref,"family_id","scenario_id"]].dropna()
    d=d.assign(lr=np.log(d.TTPA.clip(lower=1e-12)/d[ref].clip(lower=1e-12)))
    a=fit(d,False); b=fit(d,True); VC[ref]=(a[3],a[4],b[3],b[4]); NUM[f"mm_{ref}"]=(a[:3],b[:3],len(d))
    body.append(f"{LAB[ref]} & {len(d)} & {P(a[0])} & {CI(a[1],a[2])} & {P(b[0])} & {CI(b[1],b[2])}\\\\")
vcs="; ".join(f"{LAB[r]}: scenario {list(v[0].values())[0]:.2f}, residual {v[1]:.2f} (primary); family {v[2].get('family Var',float('nan')):.2f}, scenario {v[2].get('scenario Var',float('nan')):.2f}, residual {v[3]:.2f} (family intercept)" for r,v in VC.items() if r in ("NC","DMC","diPLS"))
tab("Mixed-model comparison of TTPA with each comparator (change in $\\rmsep$), confirmatory families. Primary model: variance component for scenario within family, no separate family intercept. Right: the same model with an explicit family random intercept.","tab:mm",body,"@{}lrrrrr@{}",
    note="Variance components of the log ratio ("+vcs.replace('_','\\_')+"). The CI of the primary model for di-PLS is unreliable (the scenario variance estimate sat at the optimiser's starting scale with a convergence warning).")
# ---------- D.3 leave-one-family-out
body=["Family left out & No correction & DMC & di-PLS\\\\","\\midrule"]
for ex in ["none"]+FAMS:
    cells=[]
    for ref in ["NC","DMC","diPLS"]:
        sub=J[J.family_id!=ex] if ex!="none" else J
        d=sub[["TTPA",ref,"family_id","scenario_id"]].dropna(); d=d.assign(lr=np.log(d.TTPA.clip(lower=1e-12)/d[ref].clip(lower=1e-12)))
        r=fit(d,False); cells.append(f"{P(r[0])} {CI(r[1],r[2])}"); NUM[f"loo_{ex}_{ref}"]=r[:3]
    body.append(("None (all seven)" if ex=="none" else FN[ex])+" & "+" & ".join(cells)+"\\\\")
tab("Leave-one-family-out: mixed-model change in $\\rmsep$ of TTPA relative to no correction, DMC and di-PLS (primary model).","tab:loo",body,"@{}lccc@{}")
# ---------- D.4 Run A
body=["Family & Runs & Operator applied & TTPA & NoOperator & MeanOnly & DMC\\\\","\\midrule"]
Dg=D[D.phase=="confirm"].groupby("family_id").theta_gate.mean()
RA={}
for f in ["cereals_handheld","wheat_fleet_A","wheat_fleet_B","wheat_new_units","grain_new_harvest","cereals_new_species"]:
    d=J[J.family_id==f]; v=[pc(np.log(d[m]/d.NC).mean()) for m in ["TTPA","NoOperator_refit","MeanOnly_refit","DMC"]]; RA[f]=v
    body.append(f"{FN[f]} & {len(d)} & {100*Dg[f]:.0f}\\% & "+" & ".join(P(x) for x in v)+"\\\\")
g_sc=J[J.family_id=='sugarcane_portables']
body.append(f"Sugarcane portables (not repeated) & 140 & {100*Dg['sugarcane_portables']:.0f}\\% & {P(pc(np.log(g_sc.TTPA/g_sc.NC).mean()))} & = TTPA in 137 of 140 runs & not run & {P(pc(np.log(g_sc.DMC/g_sc.NC).mean()))}\\\\")
sub=J[(J.family_id.isin(FAMS))&J.NoOperator_refit.notna()]
app=sub.join(D.set_index("run_id")[["theta_gate"]],rsuffix="_d") if False else sub.join(D.set_index("run_id")["theta_gate"])
a=app[app.theta_gate==1]; NUM["runA_applied"]=(len(a),pc(np.log(a.TTPA/a.NoOperator_refit).mean()),pc(np.log(a.TTPA/a.NC).mean()))
for f in ["wheat_fleet_A","wheat_fleet_B","wheat_new_units"]:
    x=a[a.family_id==f]; NUM[f"runA_{f}"]=(len(x),pc(np.log(x.TTPA/x.NoOperator_refit).mean()),int((x.TTPA<x.NoOperator_refit).sum()))
body.append("\\midrule")
body.append(f"Runs with operator applied ({len(a)}), TTPA vs NoOperator & & & {P(NUM['runA_applied'][1])} & & & \\\\")
mo=pc(np.log(J[J.family_id=="cereals_handheld"].MeanOnly_refit/J[J.family_id=="cereals_handheld"].TTPA).mean()); NUM["meanonly_hh_vs_ttpa"]=mo
tab("Post hoc operator ablation on the stored splits: change in $\\rmsep$ relative to no correction (geometric mean). NoOperator: frozen TTPA with the operator forced off. MeanOnly: additionally without the out-of-span mean correction. The frozen TTPA predictions were reproduced exactly. In the last row, TTPA is compared with NoOperator in the runs in which the operator was applied.","tab:runA",body,"@{}lrrrrrr@{}")
# ---------- D.5 null draws
N=pd.read_csv(W+"out3/null_draws.csv"); N["anyc"]=((N.operator==1)|(N.outspan==1)).astype(int)
def wil(k,n,z=1.96):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return c-h,c+h
body=["& \\multicolumn{2}{c}{Full size} & \\multicolumn{2}{c}{Half size}\\\\","\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}","Quantity & Rate & 95\\% CI & Rate & 95\\% CI\\\\","\\midrule"]
for col,lab in [("operator","Operator gate"),("broadening","Broadening gate"),("outspan","Out-of-span mean gate"),("inspan_gate_would_fire","In-span gate (not used)"),("anyc","Operator or out-of-span applied")]:
    r=[]
    for fr in (1.0,0.5):
        g=N[N.size_fraction==fr]; k=int(g[col].sum()); n=len(g); lo,hi=wil(k,n); r+= [f"{k}/{n} = {k/n:.3f}",f"{lo:.3f}--{hi:.3f}"]
        NUM[f"null_{col}_{fr}"]=(k,n,lo,hi)
    body.append(f"{lab} & "+" & ".join(r)+"\\\\")
body.append("\\midrule"); body.append("Change in $\\rmsep$ & \\multicolumn{2}{c}{Full size} & \\multicolumn{2}{c}{Half size}\\\\")
for ref,lab in [("NC","TTPA vs no correction"),("DMC","TTPA vs DMC")]:
    r=[]
    for fr in (1.0,0.5):
        g=N[N.size_fraction==fr]; v=pc(np.log(g.rmsep_TTPA/g[f"rmsep_{ref}"]).mean()); r+=[f"\\multicolumn{{2}}{{c}}{{{P(v)}}}"]; NUM[f"nullrmsep_{ref}_{fr}"]=v
    body.append(f"{lab} & {r[0]} & {r[1]}\\\\")
g=N[(N.anyc==0)]; v=pc(np.log(g.rmsep_TTPA/g.rmsep_NC).mean()); NUM["null_nogate_vsNC"]=v
body.append(f"TTPA vs no correction, draws without operator or out-of-span correction ({len(g)} draws) & \\multicolumn{{4}}{{c}}{{{P(v)}}}\\\\")
tab("Post hoc same-instrument test of the complete algorithm (fleet context): four datasets (IDRC wheat A1 and B1, ICRISAT benchtop cereals, grain 2022), 20 random draws each; full size: half of the spectra as pseudo-source, a quarter as adaptation set, a quarter as test set; half size: the source and adaptation sets halved. Wilson intervals assume independent draws. The in-span mean correction was applied in every draw.","tab:null",body,"@{}lcccc@{}")
# ---------- D.6 recovery
R=pd.read_csv(W+"out3/recovery.csv")
body=["True (shift, warp, $\\sigma$) & Noise & Shift RMSE & Warp RMSE & $\\hat\\sigma$ mean & $\\hat\\sigma$ RMSE\\\\","\\midrule"]
for (s,wp,sg),g0 in R.groupby(["true_shift","true_warp","true_sigma"]):
    for nz,g in g0.groupby("noise_rel"):
        rm=lambda x:float(np.sqrt((x**2).mean()))
        body.append(f"({s:g}, {wp:g}, {sg:g}) & {nz*100:.1f}\\% & {rm(g.shift_b-s):.2f} & {rm(g.warp_b-wp):.2f} & {g.sigma_b.mean():.2f} & {rm(g.sigma_b-sg):.2f}\\\\")
tab("Post hoc parameter-recovery simulation: IDRC spectra transformed by a known operator (shift and warp in channels, $\\sigma$ in channels) plus a smooth additive response and noise (percent of the spectral standard deviation); 16 fits per row (adaptation sets of 20 and 60 spectra, 8 repeats each).","tab:recov",body,"@{}lrrrrr@{}")
# ---------- D.7 sensitivity
S=pd.read_csv(W+"out3/sensitivity_k_baseline.csv"); FAMLONG={"ICRISAT cereals benchtop -> handheld":"cereals_handheld","IDRC-2016 wheat fleet A (same samples)":"wheat_fleet_A","IDRC-2016 wheat fleet B (same samples)":"wheat_fleet_B","IDRC-2016 wheat new units + new samples":"wheat_new_units"}
S["family_id"]=S.family.map(FAMLONG)
key=CF.pivot_table(index=["family_id","scenario_id","split_seed"],columns="method",values="rmsep")[["NC","TTPA"]].reset_index(); S=S.merge(key,left_on=["family_id","scenario","rep"],right_on=["family_id","scenario_id","split_seed"])
cfg=[("TTPA","Frozen"),("k_var99.9","$k$ at 99.9\\% variance"),("k_LV+5_only","$k=$ LV$+5$"),("baseline_none","No baseline"),("baseline_linear","Linear baseline")]
body=["Configuration & "+" & ".join(FN[f] for f in FAMS[:1]+FAMS[2:5])+"\\\\","\\midrule"]
sens_max=0
for c,lab in cfg:
    r=[]
    for f in ["cereals_handheld","wheat_fleet_A","wheat_fleet_B","wheat_new_units"]:
        d=S[S.family_id==f]; v=pc(np.log(d[c]/d.NC).mean()); r.append(P(v))
        if c!="TTPA": sens_max=max(sens_max,abs(v-pc(np.log(d.TTPA/d.NC).mean())))
    body.append(lab+" & "+" & ".join(r)+"\\\\")
NUM["sens_max_diff"]=sens_max; NUM["sens_runs"]=len(S)
DP=pd.read_csv(W+"out3/dipls_penalty.csv"); INV={v:k for k,v in {"cereals_handheld":"ICRISAT cereals benchtop -> handheld","wheat_fleet_A":"IDRC-2016 wheat fleet A (same samples)","wheat_fleet_B":"IDRC-2016 wheat fleet B (same samples)","wheat_new_units":"IDRC-2016 wheat new units + new samples","grain_new_harvest":"Grain new harvest year (population shift)","cereals_new_species":"Cereals new species (population shift)"}.items()}
DP["family_id"]=DP.family.map(INV)
key2=CF.pivot_table(index=["family_id","scenario_id","split_seed"],columns="method",values="rmsep")[["TTPA","diPLS"]].reset_index(); DP=DP.merge(key2,left_on=["family_id","scenario","rep"],right_on=["family_id","scenario_id","split_seed"])
body.append("\\midrule"); body.append("TTPA vs di-PLS (family-level change) & \\multicolumn{4}{c}{Family-balanced mean (six families)}\\\\")
for c,lab in [("diPLS","di-PLS, heuristic penalty (as reported)"),("diPLS_l10","di-PLS, $\\lambda=10$"),("diPLS_l100","di-PLS, $\\lambda=100$"),("diPLS_l1000","di-PLS, $\\lambda=10^3$"),("diPLS_l10000","di-PLS, $\\lambda=10^4$")]:
    v=pc(np.log(DP.TTPA/DP[c]).groupby(DP.family_id).mean().mean()); hh=pc(np.log(DP.TTPA/DP[c])[DP.family_id=="cereals_handheld"].mean()); NUM[f"dipls_{c}"]=(v,hh)
    body.append(f"{lab} & \\multicolumn{{2}}{{c}}{{{P(v)}}} & \\multicolumn{{2}}{{c}}{{handheld {P(hh)}}}\\\\")
tab("Post hoc sensitivity analyses. Upper block: family-level change in $\\rmsep$ relative to no correction for TTPA with other subspace rules and baseline bases (fleet families only; in the chem context TTPA returns the source calibration whatever these settings are). Lower block: TTPA relative to di-PLS with its reported heuristic penalty and with fixed penalties.","tab:sens",body,"@{}lrrrr@{}",
    note="The fixed-penalty runs were nearly identical across $\\lambda$ (and equal to DMC in the wheat families), so this grid is probably saturated; smaller $\\lambda$ and DANN settings were not tested.")
(__import__("os").makedirs(W+"tex",exist_ok=True), open(W+"tex/tables_appendixD.tex","w"))[1].write("\n".join(out))
json.dump({k:(v if not isinstance(v,tuple) else list(v)) for k,v in NUM.items()},open(W+"out3/NUM.json","w"),default=float,indent=1)
for k in ["mm_NC","mm_DMC","mm_ctxDMC","mm_diPLS","runA_applied","runA_wheat_fleet_A","runA_wheat_fleet_B","runA_wheat_new_units","meanonly_hh_vs_ttpa","null_operator_1.0","null_anyc_1.0","null_anyc_0.5","nullrmsep_NC_1.0","nullrmsep_NC_0.5","nullrmsep_DMC_1.0","null_nogate_vsNC","sens_max_diff","sens_runs","dipls_diPLS","dipls_diPLS_l100","loo_sugarcane_portables_NC","loo_none_NC","wlt_NC","wlt_DMC","wlt_diPLS","wlt_TCA"]:
    print(k,NUM.get(k))
