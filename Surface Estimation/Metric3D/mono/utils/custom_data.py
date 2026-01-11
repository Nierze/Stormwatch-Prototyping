import glob
import os
import json
import cv2

def load_from_annos(anno_path):
    with open(anno_path, 'r') as f:
        annos = json.load(f)['files']

    datas = []
    for i, anno in enumerate(annos):
        rgb = anno['rgb']
        depth = anno['depth'] if 'depth' in anno else None
        depth_scale = anno['depth_scale'] if 'depth_scale' in anno else 1.0
        intrinsic = anno['cam_in'] if 'cam_in' in anno else None
        normal = anno['normal'] if 'normal' in anno else None

        data_i = {
            'rgb': rgb,
            'depth': depth,
            'depth_scale': depth_scale,
            'intrinsic': intrinsic,
            'filename': os.path.basename(rgb),
            'folder': rgb.split('/')[-3],
            'normal': normal
        }
        datas.append(data_i)
    return datas

def load_data(path: str):
    if os.path.isfile(path):
        rgbs = [path]
    else:
        rgbs = glob.glob(path + '/*.jpg') + glob.glob(path + '/*.png')
    
    data = []
    for i in rgbs:
        folder = os.path.basename(os.path.dirname(os.path.abspath(i)))
        data.append({
            'rgb': i, 
            'depth': None, 
            'intrinsic': None, 
            'filename': os.path.basename(i), 
            'folder': folder
        })
    return data