"""Drivers of ecological quality (level, 2021-2025 mean) and of its 1990-2025 trend, using predictors that
are not derived from Landsat reflectance. LightGBM + spatial-block cross-validation + SHAP; GeoDetector q for comparison."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
import lightgbm as lgb, shap
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error
S=np.load('rsei_stack100.npz'); P=dict(np.load('predictors100.npz')); T=np.load('trend_mmk100.npy')
V=~np.isnan(S['2025']); H,W=V.shape
lvl=np.nanmean(np.stack([S[str(y)] for y in range(2021,2026)]),0)
base=np.nanmean(np.stack([S[str(y)] for y in [1990,1992,1993,1994]]),0)
P['pop']=np.log1p(P['pop'])
NAMES={'elev':'Elevation','slope':'Slope','tpi':'Topographic position','northness':'Northness','precip':'Annual precipitation',
       'dist_road':'Distance to road','travel_time':'Travel time to city','pop':'Population (log)','dist_built':'Distance to built-up',
       'fire_count':'Fire frequency','pa_inside':'Protected area','dist_pa':'Distance to PA'}
ii,jj=np.mgrid[0:H:5,0:W:5]; ii=ii.ravel(); jj=jj.ravel(); m=V[ii,jj]; ii,jj=ii[m],jj[m]
D=pd.DataFrame({k:P[k][ii,jj] for k in NAMES}); D['level']=lvl[ii,jj]; D['trend']=T[0][ii,jj]; D['baseline']=base[ii,jj]
D['x']=jj*0.1; D['y']=ii*0.1; D=D.dropna().reset_index(drop=True)
BLK=15.0; D['block']=(np.floor(D.x/BLK)*1000+np.floor(D.y/BLK)).astype(int)
print('samples',len(D),'blocks',D.block.nunique())
par=dict(n_estimators=600,learning_rate=0.03,num_leaves=31,min_child_samples=50,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,random_state=1)
out={}
for resp,feats in [('level',list(NAMES)),('trend',list(NAMES)+['baseline'])]:
    X=D[feats]; y=D[resp]; res={}
    for cvname,cv,groups in [('spatial',GroupKFold(5),D.block),('random',KFold(5,shuffle=True,random_state=1),None)]:
        pred=np.zeros(len(y))
        for tr,te in cv.split(X,y,groups):
            mdl=lgb.LGBMRegressor(**par).fit(X.iloc[tr],y.iloc[tr]); pred[te]=mdl.predict(X.iloc[te])
        res[cvname]=dict(R2=r2_score(y,pred),RMSE=float(np.sqrt(mean_squared_error(y,pred))))
    mdl=lgb.LGBMRegressor(**par).fit(X,y)
    sub=X.sample(6000,random_state=2); ex=shap.TreeExplainer(mdl); sv=ex.shap_values(sub)
    imp=pd.Series(np.abs(sv).mean(0),index=feats).sort_values(ascending=False)
    iv=ex.shap_interaction_values(sub.iloc[:1500]); ia=np.abs(iv).mean(0); np.fill_diagonal(ia,0)
    pairs=sorted([(feats[a],feats[b],float(ia[a,b]*2)) for a in range(len(feats)) for b in range(a+1,len(feats))],key=lambda t:-t[2])[:6]
    np.savez_compressed(f'shap_{resp}.npz',sv=sv,X=sub.values,feats=np.array(feats),ia=ia)
    out[resp]=dict(cv=res,importance=imp.round(5).to_dict(),top_interactions=pairs,n=len(D))
    print(resp,res,'\n',imp.round(4).to_string(),'\n',pairs)
# GeoDetector factor detector (q) on the same non-circular factors, quantile strata (5 classes)
def qstat(y,s):
    N=len(y); return 1-sum(len(g)*g.var(ddof=0) for _,g in y.groupby(s))/(N*y.var(ddof=0))
gd={}
for resp in ['level','trend']:
    gd[resp]={}
    for f in NAMES:
        s=D[f] if f=='pa_inside' else pd.qcut(D[f].rank(method='first'),5,labels=False)
        gd[resp][f]=round(qstat(D[resp],s),4)
out['geodetector_q']=gd; json.dump(out,open('drivers_results.json','w'),indent=1,default=float)
print(pd.DataFrame(gd).sort_values('level',ascending=False))
# PA effect check: compare inside vs outside with matching-lite (same elevation/slope/travel-time strata)
D['stratum']=pd.qcut(D.elev,5,labels=False).astype(str)+pd.qcut(D.slope,5,labels=False).astype(str)+pd.qcut(D.travel_time.rank(method='first'),5,labels=False).astype(str)
eff=[]
for st,g in D.groupby('stratum'):
    a=g[g.pa_inside==1]; b=g[g.pa_inside==0]
    if len(a)>=30 and len(b)>=30: eff.append((len(a),a.level.mean()-b.level.mean(),a.trend.mean()-b.trend.mean()))
e=np.array(eff); w=e[:,0]/e[:,0].sum()
pa=dict(strata=len(e),n_inside=int(e[:,0].sum()),level_diff=float((e[:,1]*w).sum()),trend_diff=float((e[:,2]*w).sum()))
out['pa_matched']=pa; json.dump(out,open('drivers_results.json','w'),indent=1,default=float); print(pa)
D.to_csv('drivers_samples.csv',index=False)
