"""Inputs for Omniscape sensitivity: radius variants and the pressure-only surrogate (terrestrial)."""
import numpy as np, os, warnings; warnings.filterwarnings('ignore')
import resistance as RS
from cs_prep import agg, write_asc
for src,dst in [('birds','birds_r25'),('birds','birds_r100'),('arboreal','arboreal_r25'),('terrestrial','terrestrial_r100')]:
    for p in ('R','omsrc'):
        if not os.path.exists(f'cs/{p}_{dst}.asc'): os.symlink(f'{p}_{src}.asc',f'cs/{p}_{dst}.asc')
# 40 km radius at 400 m
b=np.load('step3_terrestrial.npz'); R4=agg(b['R'],f=4); h4=agg(b['h'],f=4)
write_asc('cs/R_terrestrial_r400.asc',R4,res=400); thr=np.nanquantile(h4,0.6); write_asc('cs/omsrc_terrestrial_r400.asc',np.where(h4>=thr,h4,0),res=400)
# pressure-only surrogate
g='terrestrial'; G=RS.GUILD[g]
h=(1-RS.pressure(g))*(1-G['hunt_w']*RS.hunting(g)); c=G['c']
Rp=100-99*(1-np.exp(-c*h))/(1-np.exp(-c)); Rp=np.where(RS.BF>0.2,100,Rp).astype('float32'); Rp[~RS.VALID]=np.nan
h[~RS.VALID]=np.nan
R2=agg(Rp); h2=agg(h); write_asc('cs/R_terrpress.asc',R2); thr=np.nanquantile(h2,0.6); write_asc('cs/omsrc_terrpress.asc',np.where(h2>=thr,h2,0))
print('ok')
