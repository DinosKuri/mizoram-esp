import ee, json, warnings
warnings.filterwarnings('ignore')
import os; K=os.environ.get('GEE_KEY','gee_key.json')
_info=json.load(open(K))
ee.Initialize(ee.ServiceAccountCredentials(_info['client_email'], K), project=_info['project_id'])
PROJECT=_info['project_id']
miz=ee.FeatureCollection('FAO/GAUL/2015/level1').filter(ee.Filter.eq('ADM1_NAME','Mizoram')).geometry()
