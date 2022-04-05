import glob
import os
import numpy as np
import pymeshlab
from tqdm import tqdm


# load kit object models and convert it to bop format
INPUT_PATH = '/home/seung/Workspace/custom/6DPoseAnnotator/data/hope/models/'
OUTPUT_PATH = '/home/seung/Workspace/custom/6DPoseAnnotator/data/hope/models_processed/'


for i, input_ply in enumerate(tqdm(sorted(glob.glob(INPUT_PATH + '/*ply')))):
    
    object_name = input_ply.split('/')[-1]
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_ply)
    # ms.apply_filter('transform_scale_normalize', axisx=1000, axisy=1000, axisz=1000, scalecenter='barycenter')

    ms.apply_filter('subdivision_surfaces_midpoint', iterations=10) # bigbird
    # try:
        # ms.apply_filter('subdivision_surfaces_midpoint', threshold=1) # kit
    # except:
        # pass
    ms.apply_filter('turn_into_a_pure_triangular_mesh')
    ms.apply_filter('compute_texcoord_by_function_per_wedge')
    ms.apply_filter('transfer_texture_to_color_per_vertex')
    ms.apply_filter('transfer_color_texture_to_vertex')
    ms.save_current_mesh(OUTPUT_PATH + '/' + object_name, binary=False, save_vertex_normal=True, save_wedge_texcoord=False)
   