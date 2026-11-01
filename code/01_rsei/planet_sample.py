"""Stratified random sample of RSEI-2025 points (30 per class, >= 1 km apart) and Planet monthly chips (Oct 2024, Jan 2025)."""
import numpy as np, pandas as pd, json, math, io, os, requests
from PIL import Image
from pyproj import Transformer
from concurrent.futures import ThreadPoolExecutor
import os; KEY=os.environ['PLANET_API_KEY']
G=json.load(open('grid.json')); S=np.load('rsei_stack100_obs.npz'); r=S['2025']
rng=np.random.default_rng(2025); pts=[]
cls=np.floor(np.clip(r,0,0.9999)*5)
for k in range(5):
    ii,jj=np.where(cls==k); o=rng.permutation(len(ii)); got=[]
    for t in o:
        x=G['x0']+(jj[t]+0.5)*100; y=G['y1']-(ii[t]+0.5)*100
        if all((x-a)**2+(y-b)**2>=1000**2 for a,b,_ in pts+got): got.append((x,y,k))
        if len(got)==30: break
    pts+=got
tr=Transformer.from_crs(32646,4326,always_xy=True)
P=pd.DataFrame(pts,columns=['x','y','rsei_class']); P['lon'],P['lat']=tr.transform(P.x.values,P.y.values)
P=P.sample(frac=1,random_state=7).reset_index(drop=True); P['point_id']=[f'V{i+1:03d}' for i in range(len(P))]
P['rsei2025']=[float(r[int((G['y1']-y)//100),int((x-G['x0'])//100)]) for x,y in zip(P.x,P.y)]
P.to_csv('planet_points_KEY.csv',index=False)        # contains RSEI class: not opened during interpretation
P[['point_id','lon','lat']].to_csv('planet_points_blind.csv',index=False)
Z=15; n=2**Z
def tile_xy(lon,lat):
    xt=(lon+180)/360*n; yt=(1-math.log(math.tan(math.radians(lat))+1/math.cos(math.radians(lat)))/math.pi)/2*n; return xt,yt
def get(mosaic,z,x,y):
    u=f'https://tiles.planet.com/basemaps/v1/planet-tiles/{mosaic}/gmap/{z}/{x}/{y}.png?api_key={KEY}'
    for a in range(3):
        try:
            q=requests.get(u,timeout=30)
            if q.status_code==200: return Image.open(io.BytesIO(q.content)).convert('RGB')
        except Exception: pass
    return Image.new('RGB',(256,256),(128,128,128))
def chip(row,mosaic,half=36):
    xt,yt=tile_xy(row.lon,row.lat); x0,y0=int(xt),int(yt)
    canvas=Image.new('RGB',(512,512))
    bx=x0-1 if xt-x0<0.5 else x0; by=y0-1 if yt-y0<0.5 else y0
    for dx in (0,1):
        for dy in (0,1): canvas.paste(get(mosaic,Z,bx+dx,by+dy),(dx*256,dy*256))
    px=(xt-bx)*256; py=(yt-by)*256
    return canvas.crop((int(px-half),int(py-half),int(px+half),int(py+half)))
os.makedirs('planet_chips',exist_ok=True)
def job(args):
    row,m,tag=args; c=chip(row,m); c.save(f'planet_chips/{row.point_id}_{tag}.png'); return 1
jobs=[(row,m,t) for row in P.itertuples() for m,t in [('global_monthly_2024_10_mosaic','2024-10'),('global_monthly_2025_01_mosaic','2025-01')]]
with ThreadPoolExecutor(8) as ex: n_ok=sum(ex.map(job,jobs))
print('chips',n_ok, 'points',len(P))
