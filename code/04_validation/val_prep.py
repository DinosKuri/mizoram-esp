"""GBIF birds for validation: filtering, a-priori family guilds, sampling events and 1 km sites with model covariates."""
import pandas as pd, numpy as np, json
from pyproj import Transformer
from scipy import ndimage as ndi
F='gbif_mizoram_2000_2025.csv'
d=pd.read_csv(F,low_memory=False)
G=json.load(open('grid.json'))
FOREST=['Timaliidae','Pellorneidae','Picidae','Bucerotidae','Trogonidae','Megalaimidae','Sittidae','Campephagidae','Eurylaimidae','Pittidae',
        'Chloropseidae','Irenidae','Vireonidae','Elachuridae','Pnoepygidae','Certhiidae','Monarchidae','Stenostiridae','Tephrodornithidae','Podargidae']
OPEN=['Passeridae','Estrildidae','Cisticolidae','Motacillidae','Hirundinidae','Ardeidae','Alaudidae','Ploceidae','Laniidae','Sturnidae','Emberizidae',
      'Acrocephalidae','Locustellidae','Charadriidae','Scolopacidae','Rallidae','Anatidae','Jacanidae','Rostratulidae','Laridae','Podicipedidae',
      'Phalacrocoracidae','Anhingidae','Ciconiidae','Pandionidae','Upupidae','Meropidae','Coraciidae','Artamidae']
b=d[(d['class']=='Aves')&d.species.notna()&(d.year>=2010)&(d.occurrenceStatus=='PRESENT')].copy()
b=b[(b.coordinateUncertaintyInMeters.isna())|(b.coordinateUncertaintyInMeters<=1000)]
b['guild']=np.where(b.family.isin(FOREST),'forest',np.where(b.family.isin(OPEN),'open','excluded'))
T=Transformer.from_crs(4326,32646,always_xy=True); x,y=T.transform(b.decimalLongitude.values,b.decimalLatitude.values)
b['x']=x; b['y']=y
b['col']=((b.x-G['x0'])//100).astype(int); b['row']=((G['y1']-b.y)//100).astype(int)
valid=np.isfinite(np.load('rsei_stack100.npz')['2025'])
ok=(b.col>=0)&(b.col<G['W'])&(b.row>=0)&(b.row<G['H']); b=b[ok]; b=b[valid[b.row,b.col]]
b['site']=(b.row//10).astype(str)+'_'+(b.col//10).astype(str)          # 1 km cells
b['event']=b.recordedBy.fillna('na')+'|'+b.eventDate.astype(str).str[:10]+'|'+b.decimalLatitude.round(3).astype(str)+'|'+b.decimalLongitude.round(3).astype(str)
b['eBird']=b.datasetKey.eq('4fa7b334-ce0d-4e88-aaae-2e0c138d049e')
b.to_csv('val_bird_records.csv',index=False)
st=dict(records=int(len(b)),species=int(b.species.nunique()),events=int(b.event.nunique()),sites_1km=int(b.site.nunique()),
        forest_species=int(b[b.guild=='forest'].species.nunique()),open_species=int(b[b.guild=='open'].species.nunique()),
        excluded_species=int(b[b.guild=='excluded'].species.nunique()),share_ebird=float(b.eBird.mean()),
        top_dataset=b.datasetKey.value_counts().head(3).to_dict(),years=[int(b.year.min()),int(b.year.max())])
json.dump(st,open('val_bird_summary.json','w'),indent=1); print(st)
