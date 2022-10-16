import os
import sys
from pyrender import Mesh

from scipy.misc import face
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
import logging
import atexit

from scipy.spatial.transform import Rotation as Rot

from utils.sample_points_from_meshes import get_verts_to_points_tensor

from pytorch3d.structures import Meshes

MANO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "mano")

class HandModel:
    # link pair of hand
    LINK = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20]
    ]
    
    
    def __init__(self, side):
        self.side = side
        self.mano_layer = ManoLayer(mano_root=MANO_PATH, side=side,
                            use_pca=False, flat_hand_mean=True, joint_rot_mode='axisang')
        
        self.pose_param = torch.zeros(1, 48)
        self.pose_param[:, :3] = (np.pi/2)*torch.tensor([0, 1, 0])
        if side=='left':
            self.pose_param[:, :3] = self.pose_param[:, :3] * -1
        
        self.shape_param = torch.zeros(1, 10)
        self.root_trans = torch.zeros(1, 3)

        self.faces = self.mano_layer.th_faces

        self.verts, self.joints = self.mano_layer(th_pose_coeffs=self.pose_param,
                                        th_betas=self.shape_param,
                                        th_trans=self.root_trans)
        self.verts = self.verts.to(torch.float32)/1000
        self.joints = self.joints.to(torch.float32)/1000
    
    def reset(self):
        # reset to default selection
        pass

    def get_geometry(self):
        return {
            "verts": self._get_verts(),
            "mesh": self._get_mesh(),
            "joints": self._get_joints(),
            "links": self._get_links()
        }
    
    def _get_verts(self):
        verts = self.verts.cpu().detach()[0, :]
        verts = o3d.utility.Vector3dVector(verts)
        
        return o3d.geometry.PointCloud(points=verts)
    
    def _get_mesh(self):
        verts = self.verts.cpu().detach()[0, :]
        faces = self.faces.cpu().detach()
        verts = o3d.utility.Vector3dVector(verts)
        faces = o3d.utility.Vector3iVector(faces)
        tri_mesh = o3d.geometry.TriangleMesh(vertices=verts, triangles=faces)
        # lineset = o3d.geometry.LineSet.create_from_triangle_mesh(tri_mesh)
        tri_mesh.compute_triangle_normals()
        
        return tri_mesh
    
    def _get_joints(self, idx=None):
        if idx is None:
            joints = self.joints.detach()[0, :]
        else:
            joints = self.joints.detach()[0, idx]
        joints = o3d.utility.Vector3dVector(joints)
        pcd = o3d.geometry.PointCloud(points=joints)
        return pcd
    
    def _get_links(self):
        joints = self.joints.cpu().detach()[0, :]
        joints = o3d.utility.Vector3dVector(joints)
        lines = o3d.utility.Vector2iVector(np.array(HandModel.LINK))
        lineset = o3d.geometry.LineSet(lines=lines, points=joints)
        
        return lineset

if __name__=="__main__":
    hand_models = {
            "right": HandModel(side='right'),
            "left": HandModel(side='left')
        }
    
    for side, hand_model in hand_models.items():
        verts = hand_model.verts
        faces = hand_model.faces.unsqueeze(0)
        meshes = Meshes(verts=verts, faces=faces)
    
        verts_to_points = get_verts_to_points_tensor(meshes, num_samples=50000)
        torch.save(verts_to_points, os.path.join(MANO_PATH, f"verts_to_points_{side}.pt"))