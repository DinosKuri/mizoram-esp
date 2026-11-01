"""Pinch points (Circuitscape current and Omniscape normalized flow), barriers (improvement scores), Omniscape classes, link table."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
from scipy import ndimage as ndi
from resistance import pressure
import sys
WG=json.load(open('corridor_W.json')); ST8=np.ones((3,3))
GUILDS=sys.argv[1:] or ['birds','arboreal','terrestrial']
def read_asc(fn):
    a=np.loadtxt(fn,skiprows=6); a[a==-9999]=np.nan; return a
def agg(a,f=2,how='min'):
    H,Wd=a.shape; a=a[:H//f*f,:Wd//f*f].reshape(H//f,f,Wd//f,f)
    return {'min':np.nanmin,'max':np.nanmax,'mean':np.nanmean}[how](a,axis=(1,3))
def clusters(m,minsz):
    lab,n=ndi.label(m,structure=ST8); sz=np.bincount(lab.ravel()); ok=np.where(sz>=minsz)[0]; ok=ok[ok>0]
    lab,n=ndi.label(np.isin(lab,ok),structure=ST8); return lab,n
src100=np.load('sources100.npy')>0; src=agg(src100.astype(float),how='max')>0
P=np.load('predictors100.npz'); pa100=P['pa_inside']>0
L3=np.load('step3_layers100.npz'); V=np.load('validation100.npz')
CL=[(0,0.5,'Impeded'),(0.5,0.9,'Reduced'),(0.9,1.3,'Diffuse'),(1.3,1.7,'Intensified'),(1.7,99,'Channelized')]
import os
out=json.load(open('step3_stats.json')) if os.path.exists('step3_stats.json') else {}
for g in GUILDS:
    W=WG[g]
    b=np.load(f'step3_{g}.npz'); valid=np.isfinite(b['R'])
    corr100=(b['mosaic']<=W)&~src100; corr=agg(b['mosaic'])<=W; corr&=~src
    near_src=ndi.distance_transform_edt(~src)<=2.5     # within 500 m of a source (node-injection zone)
    corr_pp=corr&~near_src
    cur=read_asc(f'cs/out/cs_{g}_cum_curmap.asc'); om=read_asc(f'cs/out/omni_{g}/normalized_cum_currmap.asc')
    om=om[:corr.shape[0],:corr.shape[1]]; cur=cur[:corr.shape[0],:corr.shape[1]]
    v200=np.isfinite(om)&(agg(valid.astype(float),how='max')>0)
    # Omniscape classes
    ocls=np.zeros(om.shape,np.int8)
    for k,(lo,hi,_) in enumerate(CL,1): ocls[(om>=lo)&(om<hi)&v200]=k
    ocls[~v200]=0
    oshare={nm:float((ocls==k).sum()/v200.sum()) for k,(_,_,nm) in enumerate(CL,1)}
    oshare_corr={nm:float((ocls[corr]==k).mean()) for k,(_,_,nm) in enumerate(CL,1)}
    # pinch points
    # pinch points: corridor cells that carry the top 5% of pairwise current AND show intensified or channelized
    # omnidirectional flow (normalized current >= 1.3); clusters >= 3 cells (12 ha). Variants reported for sensitivity.
    variants={}
    for qq in (0.90,0.95,0.98):
        for ot in (1.0,1.3,1.7):
            m=corr_pp&(cur>=np.nanquantile(cur[corr_pp],qq))&(om>=ot); lab,n=clusters(m,3)
            variants[f'cs_top{int(round((1-qq)*100))}_om{ot}']=dict(n=int(n),km2=float((lab>0).sum()*0.04))
    thr=np.nanquantile(cur[corr_pp],0.95)
    cs_pp=corr_pp&(cur>=thr); om_pp=corr_pp&(om>=1.3)
    lab_cs,n_cs=clusters(cs_pp,3); lab_om,n_om=clusters(om_pp,3)
    lab_rb,n_rb=clusters(cs_pp&om_pp,3); robust=lab_rb>0
    pa200=agg(pa100.astype(float),how='mean')>0.5
    # barriers (100 m, r = 300 m)
    bar=b['barrier_r300']; bthr=float(np.nanpercentile(bar[np.isfinite(bar)&(bar>0)],95)); bm=(bar>=bthr)&valid
    lab_b,n_b=clusters(bm,9)           # >= 9 ha
    ids=np.arange(1,n_b+1)
    hp=pressure(g)
    bt=pd.DataFrame(dict(barrier=ids,area_km2=ndi.sum(np.ones_like(lab_b),lab_b,ids)/100,max_improvement=ndi.maximum(bar,lab_b,ids),
        mean_pressure=ndi.mean(hp,lab_b,ids),built_share=ndi.mean((np.nan_to_num(L3['built_frac'])>0.05).astype(float),lab_b,ids),
        crop_share=ndi.mean((V['worldcover']==40).astype(float),lab_b,ids),road_within_500m=ndi.mean((np.nan_to_num(P['dist_road'],nan=1e5)<=500).astype(float),lab_b,ids),
        in_PA=ndi.mean(pa100.astype(float),lab_b,ids)))
    cy,cx=np.array(ndi.center_of_mass(np.ones_like(lab_b),lab_b,ids)).T if n_b else ([],[])
    G=json.load(open('grid.json')); bt['x']=G['x0']+(np.array(cx)+0.5)*100; bt['y']=G['y1']-(np.array(cy)+0.5)*100
    bt.sort_values('max_improvement',ascending=False).to_csv(f'barriers_{g}.csv',index=False)
    # links: add effective resistance
    T=pd.read_csv(f'links_{g}.csv'); E=pd.read_csv(f'cs/out/cs_{g}_resistances_3columns.out',sep=r'\s+',header=None,names=['a','b','R_eff'])
    T=T.drop(columns=[c for c in T.columns if c.startswith('R_eff')]).merge(E,on=['a','b'],how='left'); T.to_csv(f'links_{g}.csv',index=False)
    good=T[~T.through_source]
    np.savez_compressed(f'step3_post_{g}.npz',ocls=ocls,pinch_cs=lab_cs,pinch_om=lab_om,pinch_robust=lab_rb,barriers=lab_b,corr200=corr,cur=cur,om=om)
    out[g]=dict(W_ckm=W,barrier_threshold=bthr,n_links=int(len(good)),lcp_km=float(good.path_km.sum()),median_lcd=float(good.lcd_ckm.median()),median_cwd_pl=float(good.cwd_pl.median()),
        median_path_km=float(good.path_km.median()),median_Reff=float(good.R_eff.median()),corridor_km2=float(corr100.sum()/100),
        corridor_in_PA_share=float(pa100[corr100].mean()),
        omni_share_state=oshare,omni_share_corridor=oshare_corr,
        pinch_cs_n=int(n_cs),pinch_cs_km2=float((lab_cs>0).sum()*0.04),pinch_om_n=int(n_om),pinch_om_km2=float((lab_om>0).sum()*0.04),
        pinch_robust_n=int(n_rb),pinch_robust_km2=float(robust.sum()*0.04),pinch_robust_in_PA_share=float(pa200[robust].mean()) if robust.any() else 0,
        cs_area_also_om=float(om_pp[cs_pp].mean()),pinch_variants=variants,
        barrier_n=int(n_b),barrier_km2=float((lab_b>0).sum()/100),barrier_median_pressure=float(bt.mean_pressure.median()) if n_b else None,
        barrier_share_road500=float(bt.road_within_500m.mean()) if n_b else None,barrier_in_PA=int((bt.in_PA>0.5).sum()))
    print(g,json.dumps(out[g],indent=0)[:1500],flush=True)
json.dump(out,open('step3_stats.json','w'),indent=1)
