"""Hornbill stress test: least-cost network and Omniscape vs the forest-interior bird guild."""
import numpy as np, json, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import ndimage as ndi
from scipy.stats import spearmanr
def read_asc(fn):
    a=np.loadtxt(fn,skiprows=6); a[a==-9999]=np.nan; return a
CL=[(0,0.5,'Impeded'),(0.5,0.9,'Reduced'),(0.9,1.3,'Diffuse'),(1.3,1.7,'Intensified'),(1.7,99,'Channelized')]
h=np.load('step3_hornbills.npz'); b=np.load('step3_birds.npz')
lh=h['lcp']>0; lb=b['lcp']>0; near=lambda m: ndi.distance_transform_edt(~m)<=5
oh=read_asc('cs/out/omni_hornbills/normalized_cum_currmap.asc'); ob=read_asc('cs/out/omni_birds/normalized_cum_currmap.asc')
ocls=np.zeros(oh.shape,np.int8); v=np.isfinite(oh)&(oh>0)
for k,(lo,hi,_) in enumerate(CL,1): ocls[(oh>=lo)&(oh<hi)&v]=k
np.savez_compressed('step3_post_hornbills.npz',ocls=ocls,om=oh)
T=pd.read_csv('links_hornbills.csv'); good=T[~T.through_source]
m=np.isfinite(oh)&np.isfinite(ob)&(oh>0)&(ob>0); rho=spearmanr(oh[m][::7],ob[m][::7]).correlation
out=dict(n_links=int(len(good)),lcp_km=float(good.path_km.sum()),hornbill_lcp_within500m_of_bird=float(near(lb)[lh].mean()),
         bird_lcp_within500m_of_hornbill=float(near(lh)[lb].mean()),spearman_norm_current_vs_birds=float(rho),
         omni_share={nm:float((ocls==k).sum()/max((ocls>0).sum(),1)) for k,(_,_,nm) in enumerate(CL,1)})
json.dump(out,open('hornbill_stats.json','w'),indent=1); print(out)
