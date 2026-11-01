from rsei_lib import *
import numpy as np, json, io
A=f'projects/{PROJECT}/assets/mizoram_rsei'
crs='EPSG:32646'; res=100
b=miz.transform(crs,1).bounds(1,crs).coordinates().getInfo()[0]
xs=[p[0] for p in b]; ys=[p[1] for p in b]; x0,x1,y0,y1=min(xs),max(xs),min(ys),max(ys)
x0=np.floor(x0/res)*res-2000; y1=np.ceil(y1/res)*res+2000; W=int((x1-x0)/res)+40; H=int((y1-y0)/res)+40
grid={'dimensions':{'width':W,'height':H},'affineTransform':{'scaleX':res,'shearX':0,'translateX':x0,'shearY':0,'scaleY':-res,'translateY':y1},'crsCode':crs}
json.dump({'x0':x0,'y1':y1,'res':res,'W':W,'H':H},open('grid.json','w'))
def get(img,bands,name):
    out=[]
    for b in bands:
        arr=ee.data.computePixels({'expression':img.select([b]).toFloat().unmask(-9999),'fileFormat':'NUMPY_NDARRAY','grid':grid})
        out.append(arr[b])
    np.save(f'{name}.npy',np.stack(out).astype('float32')); print('saved',name,flush=True)
get(ee.Image(f'{A}/RSEI_trend_1990_2025'),['sen_per_decade','mk_tau','mk_p','n_seasons'],'trend100')
for Y in [2000,2005,2010,2015,2020,2025,2008,1990]:
    get(rsei_image(Y),['RSEI'],f'rsei{Y}_100')
