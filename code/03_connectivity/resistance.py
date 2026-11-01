"""Guild-specific habitat suitability and resistance (100 m), version 2 (three guilds + hornbill stress test).
H   = weighted geometric mean of quality (RSEI 2021-2025), time since disturbance 1-exp(-age/tau), canopy height (/25 m), fire.
HPd = disturbance/edge pressure 1-(1-p_road)(1-p_settlement)(1-p_light), exponential decay with guild scales.
A   = hunting accessibility exp(-t/T0), t = walking time (h) from villages and roads (Tobler), weight w.
h   = H (1-HPd) (1 - w A);  R = 100 - 99(1-exp(-c h))/(1-exp(-c)) (c<=0.01: linear);
built-up (>20%) R = 100; arboreal guild: canopy gaps (canopy < GAP m or disturbed < 5 seasons ago) R = 100."""
import numpy as np
S=np.load('rsei_stack100.npz')
CUR=np.nanmean(np.stack([S[str(y)] for y in range(2021,2026)]),0)
AGE=np.load('fallow_age100.npz')['age']
L3=np.load('step3_layers100.npz'); P=np.load('predictors100.npz')
VALID=np.isfinite(CUR)
FIRE=np.nan_to_num(P['fire_count']); F95=np.nanpercentile(FIRE[VALID],95)
NTL=np.nan_to_num(L3['ntl']); NBG=np.nanmedian(NTL[VALID]); N99=np.nanpercentile(NTL[VALID],99.5)
CH=np.nan_to_num(L3['canopy_h']); BF=np.nan_to_num(L3['built_frac'])
DR=np.nan_to_num(P['dist_road'],nan=60000); DB=np.nan_to_num(P['dist_built'],nan=60000)
WT=np.nan_to_num(np.load('walk_time_h.npy'),nan=99.0)
GUILD={
 'birds':      dict(label='Forest-interior birds',  D_road=300,  D_set=500,  tau=10, c=4, hunt_T0=None, hunt_w=0,   gap=None, radius_km=5,  block=5),
 'arboreal':   dict(label='Arboreal mammals',       D_road=500,  D_set=1000, tau=10, c=4, hunt_T0=1.0,  hunt_w=0.5, gap=10,   radius_km=5,  block=5),
 'terrestrial':dict(label='Terrestrial mammals',    D_road=3000, D_set=3000, tau=8,  c=8, hunt_T0=2.0,  hunt_w=0.5, gap=None, radius_km=20, block=11),
 'hornbills':  dict(label='Hornbills (stress test)',D_road=300,  D_set=500,  tau=10, c=4, hunt_T0=2.0,  hunt_w=0.5, gap=None, radius_km=15, block=9),
}
def factors(tau=8.0):
    q=np.clip((CUR-0.2)/0.6,0,1)
    a=1-np.exp(-np.nan_to_num(AGE)/tau)
    h=np.clip(CH/25.0,0,1)
    f=1-np.clip(FIRE/F95,0,1)
    return dict(quality=q,age=a,canopy=h,fire=f)
def pressure(guild,scale=1.0,use_ntl=True,use_roads=True,use_settle=True,**_):
    g=GUILD[guild]
    pr=np.exp(-DR/(g['D_road']*scale)) if use_roads else 0
    ps=np.exp(-DB/(g['D_set']*scale)) if use_settle else 0
    pl=np.clip((NTL-NBG)/(N99-NBG),0,1) if use_ntl else 0
    return 1-(1-pr)*(1-ps)*(1-pl)
def hunting(guild,T0=None,**_):
    g=GUILD[guild]; T0=T0 or g['hunt_T0']
    return np.exp(-WT/T0) if T0 else np.zeros_like(WT)
def suitability(guild,w=None,drop=(),use_pressure=True,use_hunting=True,tau=None,hp_weight=1.0,hunt_w=None,hunt_T0=None,**kw):
    g=GUILD[guild]; F=factors(tau or g['tau']); w=dict(w or {k:0.25 for k in F})
    w={k:(0 if k in drop else v) for k,v in w.items()}; s=sum(w.values())
    H=np.exp(sum(w[k]/s*np.log(np.clip(F[k],1e-3,1)) for k in F))
    hp=pressure(guild,**kw) if use_pressure else 0
    hw=g['hunt_w'] if hunt_w is None else hunt_w
    A=hunting(guild,T0=hunt_T0) if (use_hunting and hw>0) else 0
    return H*(1-hp_weight*hp)*(1-hw*A)
def resistance(guild,c=None,gap=None,**kw):
    g=GUILD[guild]; c=g['c'] if c is None else c; gap=g['gap'] if gap is None else gap
    h=suitability(guild,**kw)
    R=100-99*(1-np.exp(-c*h))/(1-np.exp(-c)) if c>0.01 else 100-99*h
    R=np.where(BF>0.2,100,R)
    if gap: R=np.where((CH<gap)|(np.nan_to_num(AGE,nan=99)<5),100,R)
    R=R.astype('float32'); R[~VALID]=np.nan
    return R
if __name__=='__main__':
    import json
    out={}
    for gname in GUILD:
        h=suitability(gname); R=resistance(gname)
        out[gname]=dict(h_median=float(np.nanmedian(h[VALID])),R_median=float(np.nanmedian(R)),share_R_gt80=float((R[VALID]>80).mean()),
                        hp_median=float(np.nanmedian(pressure(gname)[VALID])),hunt_median=float(np.nanmedian(hunting(gname)[VALID])) if GUILD[gname]['hunt_T0'] else 0.0,
                        gap_share=float(((CH<GUILD[gname]['gap'])|(np.nan_to_num(AGE,nan=99)<5))[VALID].mean()) if GUILD[gname]['gap'] else 0.0)
    json.dump(out,open('resistance_stats.json','w'),indent=1); print(json.dumps(out,indent=1))
