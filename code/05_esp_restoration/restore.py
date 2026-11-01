"""Restoration scenarios: assisted natural regeneration (ANR) in restoration-priority cells and pressure mitigation at barriers,
vs equal-area random ANR in the corridor buffer. Least-cost link costs recomputed per guild at 200 m."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
import resistance as RS
from corridors import run
from cs_prep import agg
Z=np.load('esp_zones.npy'); GU=['birds','arboreal','terrestrial']
src2=np.nan_to_num(agg(np.load('sources100.npy').astype('float32'),how='max')).astype(int)
barrier=np.zeros(Z.shape,bool)
for g in GU: barrier|=np.load(f'step3_post_{g}.npz')['barriers']>0
anr=(Z==3)&~barrier            # degraded priority-corridor cells
mit=(Z==3)&barrier             # barrier zones
COST_ANR=23940; COST_CA=215385  # Rs/ha, Mizoram State CAMPA APO 2024-25
def R_of(g,anr_mask=None,mit_mask=None):
    G=RS.GUILD[g]; F=RS.factors(G['tau'])
    if anr_mask is not None:
        F['quality']=np.where(anr_mask,1.0,F['quality']); F['age']=np.where(anr_mask,1-np.exp(-30/G['tau']),F['age'])
        F['canopy']=np.where(anr_mask,1.0,F['canopy']); F['fire']=np.where(anr_mask,1.0,F['fire'])
    H=np.exp(sum(0.25*np.log(np.clip(F[k],1e-3,1)) for k in F))
    hp=RS.pressure(g); A=RS.hunting(g) if G['hunt_w'] else 0*hp
    if mit_mask is not None: hp=np.where(mit_mask,hp*0.5,hp); A=np.where(mit_mask,A*0.5,A)
    h=H*(1-hp)*(1-G['hunt_w']*A); c=G['c']
    R=100-99*(1-np.exp(-c*h))/(1-np.exp(-c)); R=np.where(RS.BF>0.2,100,R)
    if G['gap']:
        gapm=(RS.CH<G['gap'])|(np.nan_to_num(RS.AGE,nan=99)<5)
        if anr_mask is not None: gapm&=~anr_mask
        R=np.where(gapm,100,R)
    R=R.astype('float32'); R[~RS.VALID]=np.nan; return agg(R)
rng=np.random.default_rng(7); pool=np.argwhere((Z==4)&RS.VALID); n=int(anr.sum())
rows=[]
for g in GU:
    base=run(R_of(g),src2,res=200.0)['table']; base=base[~base.through_source].set_index(['a','b'])
    sc={'ANR in degraded priority-corridor cells':dict(anr_mask=anr),'Pressure mitigation at barriers':dict(mit_mask=mit),'Both':dict(anr_mask=anr,mit_mask=mit)}
    for i in range(5):
        m=np.zeros(Z.shape,bool); buf=(Z==4)&RS.VALID
        while m.sum()<n:                       # random contiguous 700 m blocks inside the corridor buffer
            y,x=pool[rng.integers(len(pool))]; m[max(0,y-3):y+4,max(0,x-3):x+4]|=buf[max(0,y-3):y+4,max(0,x-3):x+4]
        sc[f'Random ANR (equal area) #{i+1}']=dict(anr_mask=m)
    for name,kw in sc.items():
        T=run(R_of(g,**kw),src2,res=200.0)['table'].set_index(['a','b']); j=base.join(T[['lcd_ckm']],rsuffix='_s',how='inner')
        red=1-j.lcd_ckm_s/j.lcd_ckm
        rows.append(dict(guild=g,scenario=name,links=len(j),total_lcd_reduction=float(1-j.lcd_ckm_s.sum()/j.lcd_ckm.sum()),
                         median_link_reduction=float(red.median()),links_improved_10pct=int((red>=0.10).sum())))
        print(rows[-1],flush=True)
R=pd.DataFrame(rows); R.to_csv('restoration_scenarios.csv',index=False)
st=dict(anr_ha=float(anr.sum()), mitigation_ha=float(mit.sum()), anr_cost_low_crore=float(anr.sum()*COST_ANR/1e7), anr_cost_high_crore=float(anr.sum()*COST_CA/1e7),
        n_barrier_zones=int(sum(int(np.load(f'step3_post_{g}.npz')['barriers'].max()) for g in GU)))
json.dump(st,open('restoration_stats.json','w'),indent=1); print(st)
