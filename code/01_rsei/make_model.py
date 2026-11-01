"""Pooled normalisation + single covariance PCA from pooled_samples.csv -> rsei_model.json"""
import pandas as pd, numpy as np, json, sys
d=pd.read_csv('pooled_samples.csv').dropna(subset=['NDVI','WET','NDBSI','LST'])
d['LSTc']=d.LST-d.groupby('season').LST.transform('median')
IND=['NDVI','WET','NDBSI','LSTc']; X=d[IND].values
lo=np.percentile(X,2,0); hi=np.percentile(X,98,0); Xn=np.clip((X-lo)/(hi-lo),0,1)
def pca(Z):
    w,V=np.linalg.eigh(np.cov(Z.T)); i=np.argmax(w); v=V[:,i]; v=v*np.sign(v[0]); return v,w[i]/w.sum()
v,ev=pca(Xn); Zs=(Xn-Xn.mean(0))/Xn.std(0); vc,evc=pca(Zs)
r0=Xn@v; p2,p98=np.percentile(r0,[2,98])
M={'IND':IND,'lo':lo.tolist(),'hi':hi.tolist(),'v':v.tolist(),'var_pc1':float(ev),'v_cor':vc.tolist(),'var_pc1_cor':float(evc),'r0_p2':float(p2),'r0_p98':float(p98),
   'sd_norm':Xn.std(0).tolist(),'n':int(len(d))}
print(json.dumps(M,indent=1))
if len(sys.argv)>1: json.dump(M,open(sys.argv[1],'w'),indent=1)
