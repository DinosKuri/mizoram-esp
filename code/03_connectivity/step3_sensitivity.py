"""Sensitivity of least-cost corridors (200 m) for the three guilds: one-at-a-time scenarios and a Monte Carlo ensemble.
Usage: python step3_sensitivity.py oat | mc N"""
import numpy as np, pandas as pd, json, sys, warnings, time; warnings.filterwarnings('ignore')
from scipy import ndimage as ndi
from resistance import resistance, GUILD
from corridors import run
from cs_prep import agg
import sources as SRC
WG=json.load(open('corridor_W.json'))
src0=np.nan_to_num(agg(np.load('sources100.npy').astype('float32'),how='max')).astype(int)
def R2(g,**kw): return agg(resistance(g,**kw))
def go(g,R,src=src0): return run(R,src,res=200.0)
def metrics(o,base,g):
    T=o['table']; good=T[~T.through_source]; W=WG[g]
    corr=o['mosaic']<=W; lcp=o['lcp']>0
    m=dict(n_links=len(good),lcp_km=float(good.path_km.sum()),median_lcd=float(good.lcd_ckm.median()),corridor_km2=float(corr.sum()*0.04))
    if base is not None:
        bc,bl=base
        near=ndi.distance_transform_edt(~lcp)<=2.5            # within 500 m
        m['baseline_lcp_within500m']=float(near[bl].mean())
        mo=o['mosaic']; n=int(bc.sum()); v=mo[np.isfinite(mo)]; thr=np.partition(v,n)[n] if n<v.size else np.inf
        ca=mo<=thr; m['jaccard_area_matched']=float((ca&bc).sum()/max((ca|bc).sum(),1))
    return m
GU=['birds','arboreal','terrestrial']
if sys.argv[1]=='oat':
    rows=[]; bases={}
    for g in GU:
        o=go(g,R2(g)); W=WG[g]; bases[g]=(o['mosaic']<=W,o['lcp']>0)
        np.savez_compressed(f'sens/base200_{g}.npz',mosaic=o['mosaic'].astype('float16'),lcp=o['lcp']>0)
        m=metrics(o,None,g); m.update(guild=g,scenario='baseline (200 m)',group='baseline'); rows.append(m); print(m,flush=True)
        SC={'linear (c≈0)':dict(c=0),'c=4':dict(c=4),'c=8':dict(c=8),'c=16':dict(c=16),
            'no night lights':dict(use_ntl=False),'no disturbance pressure':dict(use_pressure=False),'pressure softened (w=0.5)':dict(hp_weight=0.5),
            'tau=5':dict(tau=5),'tau=15':dict(tau=15),
            'without quality':dict(drop=('quality',)),'without disturbance age':dict(drop=('age',)),'without canopy height':dict(drop=('canopy',)),'without fire':dict(drop=('fire',))}
        if g=='birds': SC.update({'decay x0.5 (150/250 m)':dict(scale=0.5),'decay x2 (600 m/1 km)':dict(scale=2.0)})
        if g=='arboreal': SC.update({'gap < 5 m':dict(gap=5),'gap < 15 m':dict(gap=15),'no hunting':dict(use_hunting=False),'hunting T0=0.5 h':dict(hunt_T0=0.5),'hunting T0=2 h':dict(hunt_T0=2.0)})
        if g=='terrestrial': SC.update({'decay 1 km (x1/3)':dict(scale=1/3),'decay 5 km (x5/3)':dict(scale=5/3),'no hunting':dict(use_hunting=False),'hunting T0=1 h':dict(hunt_T0=1.0),'hunting T0=4 h':dict(hunt_T0=4.0),'hunting weight 1.0':dict(hunt_w=1.0)})
        SC={k:v for k,v in SC.items() if not (k==f"c={GUILD[g]['c']}")}
        for name,kw in SC.items():
            t=time.time(); o=go(g,R2(g,**kw)); m=metrics(o,bases[g],g); m.update(guild=g,scenario=name,group='resistance'); rows.append(m)
            np.savez_compressed(f"sens/{g}_{name.replace(' ','_').replace('/','-')}.npz",lcp=o['lcp']>0); print(g,name,round(m['baseline_lcp_within500m'],2),round(time.time()-t),flush=True)
        if g=='terrestrial':
            for name,kw in {'min area 2 km2':dict(AMIN=2),'min area 10 km2':dict(AMIN=10),'edge 200 m':dict(EDGE=200),'persistent+recurrent hotspots':dict(HH=(1,2))}.items():
                _,_,rl,k=SRC.make(EDGE=kw.get('EDGE',100),AMIN=kw.get('AMIN',5),HH=kw.get('HH',(1,)))
                s2=np.nan_to_num(agg(rl.astype('float32'),how='max')).astype(int)
                o=go(g,R2(g),s2); m=metrics(o,bases[g],g); m.update(guild=g,scenario=name,group='sources',n_sources=int(k)); rows.append(m); print(g,name,flush=True)
        # pressure-only surrogate (habitat set to 1): how much of the network is a remoteness map?
        import resistance as RS
        hp=RS.pressure(g); A=RS.hunting(g) if GUILD[g]['hunt_w'] else 0
        h=(1-hp)*(1-GUILD[g]['hunt_w']*A); c=GUILD[g]['c']
        Rp=100-99*(1-np.exp(-c*h))/(1-np.exp(-c)) if c>0.01 else 100-99*h
        Rp=np.where(RS.BF>0.2,100,Rp).astype('float32'); Rp[~RS.VALID]=np.nan
        o=go(g,agg(Rp)); m=metrics(o,bases[g],g); m.update(guild=g,scenario='pressure only (no habitat)',group='surrogacy'); rows.append(m); print(g,'pressure only',round(m['baseline_lcp_within500m'],2),flush=True)
        pd.DataFrame(rows).to_csv('step3_sensitivity_oat.csv',index=False)
    # cross-guild overlap of baseline networks
    X=[]
    for a in GU:
        for b2 in GU:
            if a==b2: continue
            near=ndi.distance_transform_edt(~bases[b2][1])<=2.5
            X.append(dict(path_of=a,within500m_of=b2,share=float(near[bases[a][1]].mean()),corridor_overlap=float((bases[a][0]&bases[b2][0]).sum()/max(bases[a][0].sum(),1))))
    pd.DataFrame(X).to_csv('guild_overlap.csv',index=False); print(pd.DataFrame(X).round(2))
if sys.argv[1]=='mc':
    N=int(sys.argv[2]); rng=np.random.default_rng(42)
    for g in GU:
        freq=None; rows=[]
        for i in range(N):
            w=dict(zip(['quality','age','canopy','fire'],rng.dirichlet([4,4,4,4])))
            c=float(np.exp(rng.uniform(np.log(2),np.log(16)))); sc=float(np.exp(rng.uniform(np.log(0.5),np.log(2))))
            kw=dict(c=c,w=w,scale=sc)
            if GUILD[g]['hunt_w']: kw['hunt_T0']=GUILD[g]['hunt_T0']*float(np.exp(rng.uniform(np.log(0.5),np.log(2))))
            o=go(g,R2(g,**kw)); T=o['table']; good=T[~T.through_source]
            band=ndi.binary_dilation(o['lcp']>0,iterations=1)
            freq=band.astype('float32') if freq is None else freq+band
            rows.append(dict(i=i,c=c,scale=sc,hunt_T0=kw.get('hunt_T0'),**{'w_'+k:v for k,v in w.items()},n_links=len(good),lcp_km=float(good.path_km.sum())))
            print(g,i,flush=True)
        np.save(f'mc_lcp_freq_{g}.npy',freq/N); pd.DataFrame(rows).to_csv(f'mc_runs_{g}.csv',index=False)
