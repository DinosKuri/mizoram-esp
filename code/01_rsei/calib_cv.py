"""Leave-one-season-out test of cross-sensor corrections (Table S4) from calib_pairs.csv"""
import pandas as pd, numpy as np
d=pd.read_csv('calib_pairs.csv'); rows=[]
for pair,g in d.groupby('pair'):
    for ind in ['NDVI','WET','NDBSI','LST']:
        res={m:[] for m in ['None','Offset','OLS','RMA']}; rm={m:[] for m in res}
        g2=g.dropna(subset=[ind+'_a',ind+'_b'])
        for s in g2.season.unique():
            tr=g2[g2.season!=s]; te=g2[g2.season==s]; a,b=tr[ind+'_a'],tr[ind+'_b']; x,y=te[ind+'_a'],te[ind+'_b']
            k,c=np.polyfit(a,b,1); r=np.sign(np.corrcoef(a,b)[0,1])*b.std()/a.std(); ci=b.mean()-r*a.mean()
            pr={'None':x,'Offset':x+(b-a).mean(),'OLS':k*x+c,'RMA':r*x+ci}
            for m,p in pr.items(): res[m].append(abs((p-y).mean())); rm[m].append(np.sqrt(((p-y)**2).mean()))
        rows.append(dict(pair=pair,indicator=ind,n_seasons=g2.season.nunique(),n_points=len(g2),offset=round((g2[ind+'_b']-g2[ind+'_a']).mean(),4),
            **{f'bias_{m}':round(np.mean(v),4) for m,v in res.items()},**{f'rmse_{m}':round(np.mean(v),4) for m,v in rm.items()}))
pd.DataFrame(rows).to_csv('tableS3_calibration_cv.csv',index=False)
