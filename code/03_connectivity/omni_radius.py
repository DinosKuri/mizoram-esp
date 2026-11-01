"""Omniscape sensitivity: radius variants and the pressure-only surrogate vs baseline normalized current (200 m)."""
import numpy as np, json
from scipy.stats import spearmanr
def read(t):
    a=np.loadtxt(f'cs/out/omni_{t}/normalized_cum_currmap.asc',skiprows=6); a[a==-9999]=np.nan; return a
def shares(a):
    v=a[np.isfinite(a)&(a>0)]
    return dict(impeded=float((v<0.5).mean()),intens_chan=float((v>=1.3).mean()))
out={}
for base,var,lab in [('birds','birds_r25','birds 2.5 km'),('birds','birds_r100','birds 10 km'),('arboreal','arboreal_r25','arboreal 2.5 km'),
                     ('terrestrial','terrestrial_r100','terrestrial 10 km'),('terrestrial','terrestrial_r400','terrestrial 40 km (400 m)'),('terrestrial','terrpress','terrestrial pressure only')]:
    B=read(base); V=read(var)
    if V.shape!=B.shape: V=np.kron(V,np.ones((2,2)))[:B.shape[0],:B.shape[1]]
    h=min(B.shape[0],V.shape[0]); w=min(B.shape[1],V.shape[1]); B=B[:h,:w]; V=V[:h,:w]
    m=np.isfinite(B)&np.isfinite(V)&(B>0)&(V>0)
    out[lab]=dict(spearman=float(spearmanr(B[m][::5],V[m][::5]).correlation),**shares(V),base=shares(B))
    print(lab,{k:(round(v,3) if isinstance(v,float) else v) for k,v in out[lab].items()})
json.dump(out,open('omni_radius.json','w'),indent=1)
