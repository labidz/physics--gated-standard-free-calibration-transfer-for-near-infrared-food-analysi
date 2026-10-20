"""Spectrum-level paired validation on corn (post hoc). python paired_validation.py  (needs the repository, data/raw/jchemodata/corn.jld2)"""
import sys, json
REPO=sys.argv[1] if len(sys.argv)>1 else "spectwin"; sys.path.insert(0,REPO+"/src")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from spectwin.io_jld2 import read_jld2
from spectwin.transfer2 import TTPA4, instrument_op
d=read_jld2(REPO+"/data/raw/jchemodata/corn.jld2")["dat"]
axis=np.array([float(c) for c in d["Xm5"].columns]); X={k:d[f"X{k}"].values.astype(float) for k in ["m5","mp5","mp6"]}
FROZEN=dict(broaden="auto",gate_par=False,gof=True,chem_mean="none",chem_theta="abstain")
rms=lambda a:float(np.sqrt(np.mean(np.asarray(a)**2)))
rows=[]; keep=None
for src in X:
    for tgt in X:
        if src==tgt: continue
        for rep in range(5):
            perm=np.random.default_rng(rep).permutation(80); S,A,T=perm[:40],perm[40:60],perm[60:]
            tt=TTPA4("fleet",seed=rep,broaden="test").fit_source(X[src][S],axis,k_min=8).fit_target(X[tgt][A]); tt.configure(**FROZEN)
            th=np.array(tt.stats["theta_used"],float); dd=np.asarray(tt.d,float).ravel()
            xt=X[tgt][T]; xs=X[src][T]
            xtilde=instrument_op(xs,axis,*th)+dd
            xmean=xs+(X[tgt][A].mean(0)-X[src][S].mean(0))          # mean shift only
            r0,r1,r2=rms(xt-xs),rms(xt-xmean),rms(xt-xtilde)
            rows.append(dict(src=src,tgt=tgt,rep=rep,operator=int(tt.stats["theta_gate"]),shift=th[0],warp=th[1],sigma=th[2],rms_none=r0,rms_mean=r1,rms_ttpa=r2))
            if (src,tgt,rep)==("m5","mp6",0): keep=dict(axis=axis,xs=xs.mean(0),xt=xt.mean(0),xmean=xmean.mean(0),xtilde=xtilde.mean(0),res_none=(xt-xs),res_mean=(xt-xmean),res_ttpa=(xt-xtilde),theta=th)
D=pd.DataFrame(rows); D.to_csv("out3/paired_validation_corn.csv",index=False)
print("pairs x seeds:",len(D),"operator accepted:",D.operator.sum())
print("median RMS none/mean/ttpa:",D[["rms_none","rms_mean","rms_ttpa"]].median().round(4).to_dict())
print("median ratio ttpa/none %.3f  ttpa/mean %.3f"%((D.rms_ttpa/D.rms_none).median(),(D.rms_ttpa/D.rms_mean).median()))
print("share ttpa<mean %.2f  ttpa<none %.2f"%((D.rms_ttpa<D.rms_mean).mean(),(D.rms_ttpa<D.rms_none).mean()))
acc=D[D.operator==1]; print("when operator accepted (n=%d): median ratio ttpa/mean %.3f; ttpa<mean in %.2f"%(len(acc),(acc.rms_ttpa/acc.rms_mean).median(),(acc.rms_ttpa<acc.rms_mean).mean()))
rej=D[D.operator==0]; print("operator rejected n=%d: ratio ttpa/mean %.3f"%(len(rej),(rej.rms_ttpa/rej.rms_mean).median() if len(rej) else float('nan')))
print("per-pair median ratio ttpa/mean:",D.assign(r=D.rms_ttpa/D.rms_mean).groupby(["src","tgt"]).r.median().round(3).to_dict())
print("theta example (m5->mp6 rep0):",keep["theta"])
np.savez("out3/paired_example.npz",**{k:v for k,v in keep.items()})
