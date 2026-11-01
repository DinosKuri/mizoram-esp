"""Sensitivity: add WorldCover land-cover class to the current-RSEI driver model (shows the circularity)."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
import lightgbm as lgb, shap
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
D=pd.read_csv('drivers_samples.csv'); Vd=np.load('validation100.npz')
ii=np.round(D.y/0.1).astype(int); jj=np.round(D.x/0.1).astype(int)
D['landcover']=Vd['worldcover'][ii,jj]; D=D.dropna(subset=['landcover'])
F=['elev','slope','tpi','northness','precip','dist_road','travel_time','pop','dist_built','fire_count','pa_inside','dist_pa']
par=dict(n_estimators=600,learning_rate=0.03,num_leaves=31,min_child_samples=50,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,random_state=1)
res={}
for tag,feats in [('without_LULC',F),('with_LULC',F+['landcover'])]:
    X=D[feats].copy()
    if 'landcover' in feats: X['landcover']=X.landcover.astype('category')
    pred=np.zeros(len(D))
    for tr,te in GroupKFold(5).split(X,D.level,D.block):
        m=lgb.LGBMRegressor(**par).fit(X.iloc[tr],D.level.iloc[tr]); pred[te]=m.predict(X.iloc[te])
    m=lgb.LGBMRegressor(**par).fit(X,D.level); sub=X.sample(5000,random_state=2)
    sv=shap.TreeExplainer(m).shap_values(sub); imp=pd.Series(np.abs(sv).mean(0),index=feats).sort_values(ascending=False)
    res[tag]=dict(R2_spatial=float(r2_score(D.level,pred)),importance=imp.round(4).to_dict())
    print(tag,round(r2_score(D.level,pred),3),imp.head(4).round(4).to_dict())
def q(y,s): return 1-sum(len(g)*g.var(ddof=0) for _,g in y.groupby(s))/(len(y)*y.var(ddof=0))
res['geodetector_q_landcover']=float(q(D.level,D.landcover)); print('q landcover',res['geodetector_q_landcover'])
json.dump(res,open('lulc_sensitivity.json','w'),indent=1)
