"""Digitise Fig. 1 of Lallianthanga & Sailo (2012) and georeference it to the Mizoram boundary by chamfer matching."""
import numpy as np, geopandas as gpd, json, colorsys
from PIL import Image
from scipy import ndimage, optimize
IMG='bamboo_map_page1.png'  # page 1 of Lallianthanga and Sailo (2012) rendered as PNG
im=np.array(Image.open(IMG).convert('RGB')).astype(float)/255; H,W,_=im.shape
mx=np.max(im,2); mn=np.min(im,2); sat=np.where(mx>0,(mx-mn)/np.maximum(mx,1e-6),0)
hsv=np.stack(np.vectorize(colorsys.rgb_to_hsv)(im[...,0],im[...,1],im[...,2]),-1); hue=hsv[...,0]*360
area=np.zeros((H,W),bool); area[:, :530]=True; area[1100:, 380:]=False   # exclude legend/inset/scale bar
cls=np.zeros((H,W),np.uint8)       # 0 none, 1 2006, 2 2007-08, 3 2009, 4 non-flowering bamboo
s=(sat>0.45)&(mx>0.45)&area
cls[s&(hue>=190)&(hue<245)]=1; cls[s&(hue>=255)&(hue<320)]=2; cls[s&(hue>=45)&(hue<75)]=3; cls[s&(hue>=78)&(hue<140)]=4
black=(mx<0.3)&area
print({k:int((cls==k).sum()) for k in range(1,5)}, 'black',int(black.sum()))
# true boundary (lon/lat) sampled densely
adm=gpd.read_file('adm1.geojson'); miz=adm[adm.name=='Mizoram'].geometry.union_all()
geom=max(miz.geoms,key=lambda g:g.area) if miz.geom_type=='MultiPolygon' else miz
ring=geom.exterior; pts=np.array([ring.interpolate(d,normalized=True).coords[0] for d in np.linspace(0,1,3000)])
lon,lat=pts[:,0],pts[:,1]
by,bx=np.where(black); x0,x1,y0,y1=bx.min(),bx.max(),by.min(),by.max()
sx0=(x1-x0)/(lon.max()-lon.min()); sy0=(y1-y0)/(lat.max()-lat.min())
p0=[sx0,x0-sx0*lon.min(),sy0,y0+sy0*lat.max(),0.0]
dt=ndimage.distance_transform_edt(~black)
def proj(p,lo,la):
    sx,tx,sy,ty,r=p; X=sx*lo+tx; Y=-sy*la+ty
    cx,cy=X.mean(),Y.mean(); c,s_=np.cos(r),np.sin(r); return cx+(X-cx)*c-(Y-cy)*s_, cy+(X-cx)*s_+(Y-cy)*c
def cost(p):
    X,Y=proj(p,lon,lat); ok=(X>=0)&(X<W-1)&(Y>=0)&(Y<H-1)
    if ok.mean()<0.9: return 1e6
    return np.mean(np.minimum(ndimage.map_coordinates(dt,[Y[ok],X[ok]],order=1),15))
print('start cost',cost(p0))
best=optimize.minimize(cost,p0,method='Nelder-Mead',options=dict(maxiter=4000,xatol=1e-3,fatol=1e-4))
for _ in range(2): best=optimize.minimize(cost,best.x,method='Nelder-Mead',options=dict(maxiter=4000,xatol=1e-4,fatol=1e-5))
print('final cost (px)',best.fun, best.x)
X,Y=proj(best.x,lon,lat); d=ndimage.map_coordinates(dt,[Y,X],order=1); print('median boundary misfit px',np.median(d),'p90',np.percentile(d,90),'px size km ~',111/best.x[2])
json.dump({'p':best.x.tolist(),'median_px':float(np.median(d)),'p90_px':float(np.percentile(d,90)),'km_per_px':111/best.x[2]},open('bamboo_georef.json','w'))
np.save('bamboo_cls_img.npy',cls)
