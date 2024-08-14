# Import necessary libraries
import numpy as np
import os.path as osp
import onnxruntime as ort
import os 
import requests
import joblib
from tqdm.contrib import itertools
from pyproj import Transformer
import rasterio
from ipyleaflet import (
    Map,
    DrawControl,
    LayersControl,
    basemaps,
    basemap_to_tiles,
    FullScreenControl
)
import shapely.geometry
import warnings
warnings.filterwarnings("ignore")

class openeoMap:
    def __init__(self,center,zoom):
        self.map = Map(center=center, zoom=zoom, scroll_wheel_zoom=True, interpolation='nearest')
        self.bbox = []
        self.point_coords = []
        self.figure = None
        self.figure_widget = None
        feature_collection = {
            'type': 'FeatureCollection',
            'features': []
        }

        draw = DrawControl(
            circlemarker={}, polyline={}, polygon={},
            marker= {"shapeOptions": {
                       "original": {},
                       "editing": {},
            }},
            rectangle = {"shapeOptions": {
                       "original": {},
                       "editing": {},
            }})

        self.map.add_control(draw)
        def handle_draw(target, action, geo_json):
            feature_collection['features'] = []
            feature_collection['features'].append(geo_json)
            if feature_collection['features'][0]['geometry']['type'] == 'Point':
                self.point_coords = feature_collection['features'][0]['geometry']['coordinates']
            else:
                coords = feature_collection['features'][0]['geometry']['coordinates'][0]
                polygon = shapely.geometry.Polygon(coords)
                self.bbox = polygon.bounds
        
        layers_control = LayersControl(position='topright')
        self.map.add_control(layers_control)
        self.map.add_control(FullScreenControl())
        self.map.add_layer(basemap_to_tiles(basemaps.Esri.WorldImagery))
        draw.on_draw(handle_draw)
    
    def getBbox(self):
        if(len(self.bbox) == 0):
            mapBox = self.map.bounds     
            return [ mapBox[0][1],mapBox[0][0],mapBox[1][1],mapBox[1][0]]
        else:
            return self.bbox



class PowerScalers:
    def __init__(self, path_to_scalers, bands=None):
        if bands is None:
            bands = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']

        self.bands = bands
        self.scalers = {}
        for band in self.bands:
            assert os.path.exists(osp.join(path_to_scalers, f'{band}_scaler.pkl')), f'{band}_scaler.pkl does not exist in {path_to_scalers}'
            scaler_power = joblib.load(open(osp.join(path_to_scalers, f'{band}_scaler.pkl'), 'rb'))
            self.scalers[band] = scaler_power


    def scale(self, arr, bands=None):
        if bands is None:
            bands = self.bands
        
        out_arr = np.zeros_like(arr).astype(np.float32)
        if arr.ndim == 3:
            assert arr.shape[0] == len(bands), f'arr shape {arr.shape} does not match bands {bands}'
            im_height, im_width = arr.shape[1], arr.shape[2]
            for b_i, band in enumerate(arr):
                out_arr[b_i] = self.scalers[bands[b_i]].transform(band.ravel().reshape(-1, 1)).reshape(im_height, im_width).astype(np.float32)
        
        if arr.ndim==4:
            assert arr.shape[1] == len(bands), f'arr shape {arr.shape} does not match bands {bands}'
            batch_size, im_height, im_width = arr.shape[0], arr.shape[2], arr.shape[3]
            for b_i in range(arr.shape[1]):
                out_arr[:,b_i,:,:] = self.scalers[bands[b_i]].transform(arr[:,b_i,:,:].ravel().reshape(-1, 1)).reshape(batch_size, im_height, im_width).astype(np.float32)

        return out_arr

    def unscale(self, arr, bands=None, for_visuals=False):
        if bands is None:
            bands = self.bands
        
        out_arr = np.zeros_like(arr)

        if arr.ndim == 3:
            assert arr.shape[0] == len(bands), f'arr shape {arr.shape} does not match bands {bands}'
            im_height, im_width = arr.shape[1], arr.shape[2]
            for b_i, band in enumerate(arr):
                out_arr[b_i] = self.scalers[bands[b_i]].inverse_transform(band.ravel().reshape(-1, 1)).reshape(im_height, im_width)
        
        if arr.ndim==4:
            assert arr.shape[1] == len(bands),  f'arr shape {arr.shape} does not match bands {bands}'
            batch_size, im_height, im_width = arr.shape[0], arr.shape[2], arr.shape[3]
            for b_i in range(arr.shape[1]):
                #out_arr[:,b_i,:,:] = self.scalers[bands[b_i]].transform(arr[:,b_i,:,:].ravel().reshape(-1, 1)).reshape(batch_size, im_height, im_width)
                out_arr[:,b_i,:,:] = self.scalers[bands[b_i]].inverse_transform(arr[:,b_i,:,:].ravel().reshape(-1, 1)).reshape(batch_size, im_height, im_width)
    
        return out_arr


class OnnxInference():
    def __init__(self, ort_sess_opt=None, output_dim=10):
        if ort_sess_opt is None:
            ort_sess_opt = ort.SessionOptions()
            ort_sess_opt.intra_op_num_threads = 1

        self.ort_sess_opt = ort_sess_opt
        self.output_dim = output_dim


    def recon(self, model_path, ids_l0, ids_l1):
        ort_session = ort.InferenceSession(model_path, self.ort_sess_opt)
        recon_total = np.zeros((self.output_dim,) + (ids_l0.shape[0]*2, ids_l0.shape[1]*2))

        print('processing patches', ids_l0.shape[-2]//60,'by', ids_l0.shape[-1]//60, '...')
        for i,j in itertools.product(range(0, ids_l0.shape[0], 60), range(0, ids_l0.shape[1], 60)):
            patch_l0 = ids_l0[i:i+60, j:j+60].reshape(1,60,60).astype(np.int64)
            patch_l1 = ids_l1[i//2:i//2+30, j//2:j//2+30].reshape(1,30,30).astype(np.int64)

            ort_inputs = {ort_session.get_inputs()[0].name: patch_l0, ort_session.get_inputs()[1].name: patch_l1}
            ort_recon = ort_session.run(None, ort_inputs)

            recon_total[:,i*2:i*2+120, j*2:j*2+120] = ort_recon[0].reshape(self.output_dim, 120, 120)

        return recon_total
    
    def lc(self, model_path, vec_l0, vec_l1):

        ort_session = ort.InferenceSession(model_path, self.ort_sess_opt)
        lc_total = np.zeros((self.output_dim,) + (vec_l0.shape[-2], vec_l0.shape[-1]))

        print('processing patches', vec_l0.shape[-2]//60,'by', vec_l0.shape[-1]//60, '...')
        for i,j in itertools.product(range(0, vec_l0.shape[-2], 60), range(0, vec_l0.shape[-1], 60)):
            patch_l0 = vec_l0[:, i:i+60, j:j+60].reshape(1,128,60,60)
            patch_l1 = vec_l1[:, i//2:i//2+30, j//2:j//2+30].reshape(1,128,30,30)

            ort_inputs = {ort_session.get_inputs()[0].name: patch_l0, ort_session.get_inputs()[1].name: patch_l1}
            ort_lc = ort_session.run(None, ort_inputs) 
            lc_total[:,i:i+60, j:j+60] = ort_lc[0].reshape(10, 60, 60)


        return lc_total


def reproject_bbox(bbox):
    crs_epsg_ul = 32700-np.round((45+bbox[1])/90,0)*100+np.round((183+bbox[0])/6,0)
    crs_epsg_dr = 32700-np.round((45+bbox[3])/90,0)*100+np.round((183+bbox[2])/6,0)

    if crs_epsg_ul == crs_epsg_dr:
        crs_epsg = int(crs_epsg_ul)

        transformer = Transformer.from_crs("EPSG:4326", crs_epsg)
        ul = transformer.transform(bbox[1],bbox[0])
        dr = transformer.transform(bbox[3],bbox[2])
        bbox = ul + dr
        bbox = tuple([np.floor(c/40)*40 for c in bbox])


    else:
        print('Region does not fall under a single UTM zone. Falling back to long/lat.')
        crs_epsg = 4326
    return bbox, crs_epsg


def optimize_compression(path, level=None, suffix=''):

    if 'level0' in path:
        dtype = 'uint16'
    elif 'level1' in path:
        dtype = 'uint8'
    else:
        dtype = None

    if level is not None:
        dtype_level = 'uint8' if level == 1 else 'uint16'
        assert dtype == dtype_level, f"Expected {dtype_level} but got {dtype}"
    
    assert dtype is not None, f"Could not determine dtype for {path}"

    with rasterio.open(path) as src:
        r = src.read()

        m = src.meta
        m['count'] = 1
        m['dtype'] = dtype
        m['nodata'] = 0
        m['compress'] = 'deflate'
        
        out_path = path.replace('.tif', f'{suffix}.tif')
        with rasterio.open(out_path,'w',**m) as dst:
            dst.write(np.uint8(r)) if dtype == 'uint8' else dst.write(np.uint16(r))

def download_from_artifactory(rel_path, out_path, artifactory):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = requests.get(f'{artifactory}/{rel_path}', allow_redirects=True)
    if not os.path.exists(out_path):
        if r.ok:
            with open(out_path, 'wb') as f:
                f.write(r.content)
            print(f" {rel_path} downloaded")
        else:
            print(r.text)
            raise Exception(f"Could not download {rel_path}")
    else:
        print(f" {rel_path} already downloaded")