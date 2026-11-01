"""Per-pixel Theil-Sen + Mann-Kendall, modified MK (Yue & Wang 2004 effective sample size, AR(1) of
Sen-detrended series) and Benjamini-Hochberg FDR across pixels. 100 m grid. Writes trend_mmk100.npy and trend_sensitivity.csv"""
import numpy as np, pandas as pd
from scipy.stats import norm
S=np.load('rsei_stack100_obs.npz'); YRS=np.array(sorted(int(k) for k in S.files),float)
V=~np.isnan(S['2025']); H,W=V.shape
A_all=np.stack([S[str(int(y))][V] for y in YRS],1).astype(np.float32)
def run(sel):
    A=A_all[:,sel]; yrs=YRS[sel]; P,n_t=A.shape
    I,J=np.triu_indices(n_t,1); dt=(yrs[J]-yrs[I]).astype(np.float32)
    sen=np.empty(P,np.float32); s=np.empty(P,np.float32); n=np.empty(P,np.float32); r1=np.empty(P,np.float32)
    for a in range(0,P,40000):
        X=A[a:a+40000]; d=X[:,J]-X[:,I]; sl=np.nanmedian(d/dt,1)
        sen[a:a+40000]=sl; s[a:a+40000]=np.nansum(np.sign(d),1); n[a:a+40000]=(~np.isnan(X)).sum(1)
        ic=np.nanmedian(X-sl[:,None]*yrs,1); R=X-(sl[:,None]*yrs+ic[:,None]); R=R-np.nanmean(R,1,keepdims=True)
        r1[a:a+40000]=np.nansum(R[:,:-1]*R[:,1:],1)/np.maximum(np.nansum(R**2,1),1e-12)
    var0=n*(n-1)*(2*n+5)/18; k=np.arange(1,n_t)[None,:].astype(np.float32); rr=np.clip(r1,0,0.95)[:,None]
    corr=1+2*np.sum((1-k/n[:,None])*rr**k,1)
    def pv(var): z=np.where(s>0,(s-1)/np.sqrt(var),np.where(s<0,(s+1)/np.sqrt(var),0)); return 2*norm.sf(np.abs(z))
    return sen*10,pv(var0),pv(var0*corr),r1,n,corr
def bh(p,q=0.05):
    o=np.sort(p); m=len(p); ok=o<=q*np.arange(1,m+1)/m; return o[ok].max() if ok.any() else 0.0
rows=[]; tot=V.sum()*0.01; CLS={}
for tag,sel in [('1990-2025 (35 seasons)',np.ones(len(YRS),bool)),('excluding 2006-2008',~np.isin(YRS,[2006,2007,2008])),('2000-2025 (26 seasons)',YRS>=2000)]:
    sen,p0,p1,r1,n,corr=run(sel)
    if tag.startswith('1990'):
        out=np.full((4,H,W),np.nan,np.float32)
        for i,v in enumerate([sen,p1,r1,n]): out[i][V]=v
        np.save('trend_mmk100.npy',out); np.save('fdr_cut.npy',np.array([bh(p1)]))
        print('median r1',np.median(r1),'median n/n*',np.median(corr),'sen pct',np.percentile(sen,[5,50,95]))
    CLS[tag]=np.where(p1<=bh(p1),np.sign(sen),0)
    for test,p,thr in [('MK, p<0.05',p0,0.05),('MK + FDR',p0,bh(p0)),('Modified MK, p<0.05',p1,0.05),('Modified MK + FDR',p1,bh(p1))]:
        inc=((p<=thr)&(sen>0)).sum()*0.01; dec=((p<=thr)&(sen<0)).sum()*0.01
        rows.append(dict(period=tag,test=test,increase_km2=round(inc),increase_pct=round(inc/tot*100,1),decrease_km2=round(dec),decrease_pct=round(dec/tot*100,1)))
a=CLS['1990-2025 (35 seasons)']
import json
conf={k:int(((a*b)<0).sum()) for k,b in CLS.items()}; agree={k:float(((a!=0)&(a==b)).sum()/max((b!=0).sum(),1)*100) for k,b in CLS.items()}
json.dump({'conflicts':conf,'agree_pct':agree},open('trend_consistency.json','w')); print(conf,agree)
T=pd.DataFrame(rows); T.to_csv('trend_sensitivity.csv',index=False); print(T.to_string())
