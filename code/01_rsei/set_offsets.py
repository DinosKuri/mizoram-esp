import pandas as pd, json
T=pd.read_csv('tableS3_calibration_cv.csv'); off={}
for p,g in T.groupby('pair'):
    off[p]={r.indicator:(float(r.offset) if r.bias_Offset<r.bias_None else 0.0) for r in g.itertuples()}
json.dump(off,open('calib_offsets.json','w'),indent=1); print(off)
