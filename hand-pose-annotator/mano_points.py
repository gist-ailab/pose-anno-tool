
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from open3d.visualization import MeshShadeOption
 
import torch
from manopth.manolayer import ManoLayer
from torch import optim

import numpy as np
import cv2
import pickle

import yaml
import time
import json
import datetime
import shutil
import copy
from scipy.spatial.transform import Rotation as Rot

MANO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "mano")




mano_layer = ManoLayer(mano_root=MANO_PATH, side='right',
                        use_pca=False, flat_hand_mean=True, joint_rot_mode='axisang')

pose_param = torch.rand((1, 48))
shape_param = torch.zeros((1, 10))


verts, joints = mano_layer(th_pose_coeffs=pose_param,
                            th_betas=shape_param)
faces = mano_layer.th_faces
# convert face to one_hot face
max_vert = len(verts[0])
face_one_hot = torch.zeros((faces.shape[0], max_vert))
# center of triangle
for i, idx in enumerate(faces):
    face_one_hot[i, idx] = 1


print()




