from rsei_lib import *
import pandas as pd
rows=[]
for Y in [1990,1995,2000,2005,2008,2010,2015,2020,2025]:
    cls=rsei_image(Y).multiply(5).floor().clamp(0,4).add(1).toInt().rename('cls')  # 1 Poor..5 Excellent, fixed 0.2 breaks
    h=ee.Image.pixelArea().divide(1e6).addBands(cls).reduceRegion(ee.Reducer.sum().group(groupField=1,groupName='cls'),miz,100,maxPixels=1e13,tileScale=8).getInfo()
    row={'season':Y}; row.update({int(g['cls']):round(g['sum'],1) for g in h['groups']}); rows.append(row); print(row,flush=True)
df=pd.DataFrame(rows).rename(columns={1:'Poor',2:'Fair',3:'Moderate',4:'Good',5:'Excellent'}); df.to_csv('class_areas.csv',index=False); print('SAVED')
