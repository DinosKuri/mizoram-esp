import numpy as np, json
def agg(a,f=2):
    H,W=a.shape; return a[:H//f*f,:W//f*f].reshape(H//f,f,W//f,f).max(axis=(1,3))
out={'n':50}
for g in ('birds','arboreal','terrestrial'):
    f=np.load(f'mc_lcp_freq_{g}.npy'); l=agg(np.load(f'step3_{g}.npz')['lcp']>0)
    out[g]=dict(share_lcp_ge50=float((f[l]>=0.5).mean()),share_lcp_ge80=float((f[l]>=0.8).mean()),median_freq_on_lcp=float(np.median(f[l])))
json.dump(out,open('mc_stats.json','w'),indent=1); print(out)
