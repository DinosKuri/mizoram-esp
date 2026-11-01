import numpy as np
from scipy.stats import norm
S=np.load('rsei_stack100.npz'); yrs=sorted(int(k) for k in S.files)
V=~np.isnan(S['2025'])
A=np.stack([S[str(y)][V] for y in yrs],1).astype(np.float32)  # pixels x seasons
def mk(A,yr):
    n_ok=(~np.isnan(A)).sum(1); s=np.zeros(len(A),np.float32); slopes=[]
    for i in range(len(yr)):
        for j in range(i+1,len(yr)):
            d=A[:,j]-A[:,i]; s+=np.nan_to_num(np.sign(d))
    n=n_ok.astype(float); var=n*(n-1)*(2*n+5)/18
    z=np.where(s>0,(s-1)/np.sqrt(var),np.where(s<0,(s+1)/np.sqrt(var),0)); p=2*(1-norm.cdf(abs(z)))
    return s,p
def report(tag,A,yr):
    s,p=mk(A,yr); N=len(s)
    print(tag,'inc %.1f%%  dec %.1f%%'%(((s>0)&(p<0.05)).mean()*100,((s<0)&(p<0.05)).mean()*100))
report('all 35',A,yrs)
keep=[k for k,y in enumerate(yrs) if y not in (2006,2007,2008)]
report('no mautam',A[:,keep],[yrs[k] for k in keep])
keep=[k for k,y in enumerate(yrs) if y>=2000]
report('2000-2025',A[:,keep],[yrs[k] for k in keep])
# Hamed-Rao-lite: lag-1 autocorr of detrended state mean
m=np.nanmean(A,0); r1=np.corrcoef(m[:-1],m[1:])[0,1]; print('state-mean lag1 r',round(r1,2))
# pixel lag-1 autocorrelation median
X=A-np.nanmean(A,1,keepdims=True); X=np.nan_to_num(X); r=(X[:,:-1]*X[:,1:]).sum(1)/np.maximum((X**2).sum(1),1e-9)
print('median pixel lag1 r',round(float(np.median(r)),2),'share r>0.3',round(float((r>0.3).mean()*100),1))
