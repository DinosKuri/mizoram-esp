"""Disturbance history and time since last disturbance (fallow age) from the 35-season RSEI series (100 m).
Event in season t: anomaly from the state-wide season median falls by >= THR below the median anomaly of the
previous three observed seasons. Removing the state median avoids flagging region-wide signals (e.g., mautam)."""
import numpy as np, json, sys
s=np.load('rsei_stack100.npz'); yrs=np.array([int(k) for k in s.files])
R=np.stack([s[k] for k in s.files]).astype('float32')
med=np.nanmedian(R.reshape(len(yrs),-1),1)
A=R-med[:,None,None]
v=np.load('validation100.npz'); lf,ly,tc=v['loss_frac'],v['loss_year'],v['tc2000']
def detect(thr):
    ev=np.zeros(R.shape,bool)
    for i in range(3,len(yrs)):
        base=np.nanmedian(A[i-3:i],0)
        d0=A[i]-base
        if CONF and i+1<len(yrs):
            d1=A[i+1]-base
            ev[i]=(d0<=-thr)&((d1<=-thr/2)|np.isnan(d1))
        else: ev[i]=d0<=-thr
    return ev
CONF=False
res={}
for thr in [0.10,0.15,0.20,0.25,0.30]:
    ev=detect(thr)
    # Hansen agreement: loss year L detected in season L or L+1
    m=(lf>=0.5)&(tc>=50)&(ly>=4)&(ly<=20)
    iy,ix=np.where(m); L=2000+ly[iy,ix].astype(int); yi={y:i for i,y in enumerate(yrs)}
    hit=np.zeros(len(L),bool)
    for y in np.unique(L):
        k=L==y
        for yy in (y,y+1):
            if yy in yi: hit[k]|=ev[yi[yy],iy[k],ix[k]]
    ctrl=(lf==0)&(tc>=70)
    sel=(yrs>=2004)&(yrs<=2021)
    fp=ev[sel][:,ctrl].mean()   # events per cell-season in undisturbed forest
    res[thr]=dict(detect_rate=float(hit.mean()),false_rate_per_season=float(fp))
    print(thr,res[thr],flush=True)
json.dump(res,open('fallow_threshold_test.json','w'),indent=1)
# ---- final product: THR from argv (default 0.25), no confirmation
THR=float(sys.argv[1]) if len(sys.argv)>1 else 0.25
ev=detect(THR)
ev[:3]=False
last=np.full(R.shape[1:],np.nan,'float32')
for i in range(len(yrs)):
    last[ev[i]]=yrs[i]
valid=np.isfinite(R[-1])
age=np.where(np.isfinite(last),2025-last,2025-yrs[3]).astype('float32')   # never detected -> >= 32 (censored)
age[~valid]=np.nan
nev=ev.sum(0).astype('float32'); nev[~valid]=np.nan
np.savez_compressed('fallow_age100.npz',age=age,n_events=nev,censored=(~np.isfinite(last))&valid)
st={'threshold':THR,'detect_rate_hansen':res[THR]['detect_rate'],'false_rate_per_season':res[THR]['false_rate_per_season'],
    'share_never_disturbed':float(((~np.isfinite(last))&valid).sum()/valid.sum()),
    'share_age_lt5':float(((age<5)&valid).sum()/valid.sum()),'share_age_5_15':float(((age>=5)&(age<15)).sum()/valid.sum()),
    'share_age_ge15':float(((age>=15)).sum()/valid.sum()),'median_events':float(np.nanmedian(nev))}
json.dump(st,open('fallow_age_stats.json','w'),indent=1); print(st)
