"""Bird validation: suitability, sources, corridors and Omniscape vs GBIF/eBird forest-guild share (1 km sites)."""
import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score
from pyproj import Transformer
G=json.load(open('grid.json')); WG=json.load(open('corridor_W.json'))
b=pd.read_csv('val_bird_records.csv',low_memory=False)
import resistance as RS
src=np.load('sources100.npy')>0; Bm=np.load('step3_birds.npz'); Hm=np.load('step3_hornbills.npz')
hb=Bm['h']; corr=(Bm['mosaic']<=WG['birds'])&~src; press_only=1-RS.pressure('birds')
dsrc=ndi.distance_transform_edt(~src)*0.1   # km
om=np.loadtxt('cs/out/omni_birds/normalized_cum_currmap.asc',skiprows=6); om[om==-9999]=np.nan
om100=np.kron(om,np.ones((2,2)))[:hb.shape[0],:hb.shape[1]]
if om100.shape!=hb.shape: om100=np.pad(om100,((0,hb.shape[0]-om100.shape[0]),(0,hb.shape[1]-om100.shape[1])),constant_values=np.nan)
hh=Hm['h']
def cell(a,r,c,f=np.nanmean):
    return f(a[r*10:(r+1)*10,c*10:(c+1)*10])
T=Transformer.from_crs(4326,32646,always_xy=True); ax,ay=T.transform(92.7176,23.7271)
rows=[]
for s,g in b.groupby('site'):
    r,c=map(int,s.split('_'))
    sp=g.groupby('guild').species.nunique()
    cx=G['x0']+(c*10+5)*100; cy=G['y1']-(r*10+5)*100
    rows.append(dict(site=s,r=r,c=c,forest=int(sp.get('forest',0)),open=int(sp.get('open',0)),n_species=int(g.species.nunique()),effort=int(g.event.nunique()),
        hornbill=int((g.family=='Bucerotidae').any()),h=cell(hb,r,c),h_press=cell(press_only,r,c),h_horn=cell(hh,r,c),
        src_share=cell(src.astype(float),r,c),corr_share=cell(corr.astype(float),r,c),dsrc=cell(dsrc,r,c),om=cell(om100,r,c),
        d_aizawl=float(np.hypot(cx-ax,cy-ay)/1000),block=f"{int((cy-G['y1'])//10000)}_{int((cx-G['x0'])//10000)}"))
S=pd.DataFrame(rows); S=S[np.isfinite(S.h)]
S['near_aizawl']=(S.d_aizawl<=10).astype(int); S['leff']=np.log(S.effort); S['ldsrc']=np.log(S.dsrc+0.1)
S['inside']=(S.src_share>=0.5).astype(int); S['corridor']=((S.corr_share>=0.5)&(S.inside==0)).astype(int)
S['n_fo']=S.forest+S.open; S.to_csv('val_bird_sites.csv',index=False)
U=S[S.n_fo>0].copy(); U['share']=U.forest/U.n_fo
def glm(df,cols):
    X=sm.add_constant(df[cols]); y=np.column_stack([df.forest,df.open])
    m=sm.GLM(y,X,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(df.block)[0]})
    return m
out={'n_sites':int(len(S)),'n_sites_fo':int(len(U)),'events':int(S.effort.sum())}
# sampling bias
valid=np.isfinite(hb); out['share_state_in_sources']=float(src[valid].mean()); out['share_sites_inside']=float(S.inside.mean())
out['share_state_corridor']=float(corr[valid].mean()); out['share_sites_corridor']=float(S.corridor.mean())
# 1 suitability
U['h10']=U.h*10; U['hp10']=U.h_press*10
m1=glm(U,['h10','leff','near_aizawl']); m1p=glm(U,['hp10','leff','near_aizawl'])
out['suit']=dict(OR_per_0_1=float(np.exp(m1.params['h10'])),CI=[float(np.exp(m1.conf_int().loc['h10',0])),float(np.exp(m1.conf_int().loc['h10',1]))],p=float(m1.pvalues['h10']),
                 aic=float(m1.aic),aic_pressure_only=float(m1p.aic),OR_pressure_only=float(np.exp(m1p.params['hp10'])),
                 spearman=float(spearmanr(U.h,U.share).correlation))
both=glm(U,['h10','hp10','leff','near_aizawl']); out['suit']['joint']=dict(OR_h=float(np.exp(both.params['h10'])),p_h=float(both.pvalues['h10']),OR_press=float(np.exp(both.params['hp10'])),p_press=float(both.pvalues['hp10']))
# deciles (Boyce-type)
U['dec']=pd.qcut(U.h,5,labels=False,duplicates='drop')
D=U.groupby('dec').agg(h_mid=('h','median'),forest=('forest','sum'),open=('open','sum'),sites=('site','size')); D['share']=D.forest/(D.forest+D.open)
D['ratio']=D.share/(U.forest.sum()/(U.forest.sum()+U.open.sum())); D.to_csv('val_bird_quintiles.csv'); out['boyce_spearman']=float(spearmanr(D.h_mid,D.ratio).correlation)
# 2 sources
m2=glm(U,['inside','leff','near_aizawl'])
out['sources']=dict(share_in=float(U[U.inside==1].share.mean()),share_out=float(U[U.inside==0].share.mean()),n_in=int(U.inside.sum()),
                    OR=float(np.exp(m2.params['inside'])),CI=[float(x) for x in np.exp(m2.conf_int().loc['inside'])],p=float(m2.pvalues['inside']),
                    mw_p=float(mannwhitneyu(U[U.inside==1].share,U[U.inside==0].share,alternative='greater').pvalue))
# 3 corridors (non-source sites)
N=U[U.inside==0].copy()
m3=glm(N,['corridor','ldsrc','leff','near_aizawl']); m3h=glm(N,['corridor','ldsrc','leff','near_aizawl','h10'])
out['corridor']=dict(n_corr=int(N.corridor.sum()),n_matrix=int((N.corridor==0).sum()),share_corr=float(N[N.corridor==1].share.mean()),share_matrix=float(N[N.corridor==0].share.mean()),
                     OR=float(np.exp(m3.params['corridor'])),CI=[float(x) for x in np.exp(m3.conf_int().loc['corridor'])],p=float(m3.pvalues['corridor']),
                     OR_adj_h=float(np.exp(m3h.params['corridor'])),CI_adj_h=[float(x) for x in np.exp(m3h.conf_int().loc['corridor'])],p_adj_h=float(m3h.pvalues['corridor']))
bands=[(0,1),(1,3),(3,99)]; out['corridor']['bands']={f'{a}-{c}':dict(corr=float(N[(N.dsrc>=a)&(N.dsrc<c)&(N.corridor==1)].share.mean()),matrix=float(N[(N.dsrc>=a)&(N.dsrc<c)&(N.corridor==0)].share.mean()),
     n_corr=int(((N.dsrc>=a)&(N.dsrc<c)&(N.corridor==1)).sum()),n_matrix=int(((N.dsrc>=a)&(N.dsrc<c)&(N.corridor==0)).sum())) for a,c in bands}
# 4 Omniscape
U['om_c']=U.om
m4=glm(U.dropna(subset=['om_c']),['om_c','leff','near_aizawl']); m4h=glm(U.dropna(subset=['om_c']),['om_c','h10','leff','near_aizawl'])
out['omniscape']=dict(OR_per_unit=float(np.exp(m4.params['om_c'])),CI=[float(x) for x in np.exp(m4.conf_int().loc['om_c'])],p=float(m4.pvalues['om_c']),
                      OR_adj_h=float(np.exp(m4h.params['om_c'])),p_adj_h=float(m4h.pvalues['om_c']))
# 5 species-level enrichment in corridors (non-source sites; expectation = share of sampled non-source sites in corridors)
bb=b.merge(S[['site','inside','corridor']],on='site'); nb=bb[bb.inside==0]; exp_=N.corridor.mean() if len(N) else np.nan
sp=nb.groupby(['guild','species']).apply(lambda g: pd.Series(dict(sites=g.site.nunique(),corr=g[g.corridor==1].site.nunique()))).reset_index()
sp=sp[sp.sites>=10]; sp['ratio']=sp['corr']/sp.sites/exp_
out['species']={gname:dict(n=int((sp.guild==gname).sum()),above=int(((sp.guild==gname)&(sp.ratio>1)).sum()),median_ratio=float(sp[sp.guild==gname].ratio.median())) for gname in ('forest','open')}
sp.to_csv('val_bird_species_enrichment.csv',index=False)
# 6 hornbills
mh=sm.GLM(S.hornbill,sm.add_constant(pd.DataFrame(dict(hh10=S.h_horn*10,leff=S.leff))),family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(S.block)[0]})
out['hornbill']=dict(sites=int(S.hornbill.sum()),OR_per_0_1=float(np.exp(mh.params['hh10'])),CI=[float(x) for x in np.exp(mh.conf_int().loc['hh10'])],p=float(mh.pvalues['hh10']),
                     auc_h=float(roc_auc_score(S.hornbill,S.h_horn)),auc_effort=float(roc_auc_score(S.hornbill,S.effort)))
json.dump(out,open('val_birds.json','w'),indent=1); print(json.dumps(out,indent=1))
