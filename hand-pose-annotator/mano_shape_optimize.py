# Author: Raeyoung Kang (raeyo@gm.gist.ac.kr)
# GIST AILAB, Republic of Korea
# Modified from the codes of Anas Gouda (anas.gouda@tu-dortmund.de)
# FLW, TU Dortmund, Germany
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

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds


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
hangeul = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "NanumGothic.ttf")


class Utils:
    # file
    def get_file_list(path):
        file_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        return file_list
    def get_file_name(path):
        file_path, _ = os.path.splitext(path)
        return os.path.basename(file_path)
    def get_file_ext(path):
        _, ext = os.path.splitext(path)
        return ext
    
    # directory
    def get_dir_name(path):
        return os.path.basename(path)
    def get_dir_list(path):
        dir_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        return dir_list
    def create_dir(dir_path):
        os.makedirs(dir_path, exist_ok=True)    

    # yaml
    def save_dic_to_yaml(dic, yaml_path):
        with open(yaml_path, 'w') as y_file:
            _ = yaml.dump(dic, y_file, default_flow_style=False)
    def load_yaml_to_dic(yaml_path):
        with open(yaml_path, 'r') as y_file:
            dic = yaml.load(y_file, Loader=yaml.FullLoader)
        return dic

    # json
    def load_json_to_dic(json_path):
        with open(json_path, 'r') as j_file:
            dic = json.load(j_file)
        return dic
    def save_dic_to_json(dic, json_path):
        with open(json_path, 'w') as j_file:
            json.dump(dic, j_file, sort_keys=True, indent=4)

    def save_to_pickle(data, pickle_path):
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def load_pickle(pickle_path):
        with open(pickle_path, 'rb') as f:
            try:
                data = pickle.load(f)
            except ValueError:
                import pickle5
                data = pickle5.load(f)

        return data


    def trimesh_to_open3d(mesh):
        return mesh.as_open3d()

    def open3d_to_trimesh(mesh):
        pass


    #https://gist.github.com/JosueCom/7e89afc7f30761022d7747a501260fe3
    def distance_matrix(x, y=None, p = 2): #pairwise distance of vectors
        
        y = x if type(y) == type(None) else y

        n = x.size(0)
        m = y.size(0)
        d = x.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        
        dist = torch.pow(x - y, p).sum(2) ** (1/p)
        
        return dist


class LabelingMode:
    STATIC      = "F1. 직접 움직여 라벨링"
    OPTIMIZE    = "F2. 가이드 기반 라벨링"
    OPTIMIZE_SHAPE = "F3. 손 모양 최적화"

class HandModel:
    # JOINT IMG () # 180, 340
    LEFT_JOINT_IMG = {0: [91, 166], 1: [66, 148], 2: [46, 131], 3: [33, 113], 17: [125, 110], 13: [112, 99], 18: [136, 96], 5: [73, 94], 4: [14, 89], 9: [93, 88], 19: [144, 83], 14: [116, 76], 20: [153, 69], 6: [72, 67], 10: [98, 63], 15: [123, 56], 7: [71, 49], 11: [101, 44],  16: [129, 38], 8: [68, 30], 12:[105, 25]}
    RIGHT_JOINT_IMG = {0: [248, 166], 1: [272, 148], 2: [292, 131], 3: [306, 113], 17: [213, 110], 13: [226, 99], 18: [203, 96], 5: [266, 94], 4: [325, 89], 9: [245, 88], 19: [194, 83], 14: [222, 76], 20: [185, 69], 6: [267, 67], 10: [240, 63], 15: [216, 56],12: [233, 23],  8: [270, 30], 16: [210, 38],11: [237, 44], 7: [267, 49]}
    
    # link pair of hand
    LINK = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20]
    ]
    
    # index of finger or tips
    _IDX_OF_HANDS = {
        'none'   : [],
        'root'   : [0],
        'thumb'  : [1,2,3],
        'fore'   : [5,6,7],
        'middle' : [9,10,11],
        'ring'   : [13,14,15],
        'little' : [17,18,19],
    }
    _IDX_OF_GUIDE = {
        'none'   : [],
        'root'   : [0],
        'thumb'  : [1,2,3,4],
        'fore'   : [5,6,7,8],
        'middle' : [9,10,11,12],
        'ring'   : [13,14,15,16],
        'little' : [17,18,19,20],
    }
    # finger name
    _FINGER_NAME = [
        "엄지",
        "검지",
        "중지",
        "약지",
        "소지"
    ]
    # ORDER of param
    _ORDER_OF_PARAM = {
        'thumb'  : 4,
        'fore'   : 0,
        'middle' : 1,
        'ring'   : 3,
        'little' : 2
    }

    def __init__(self, side, shape_param=None):
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = 'cpu:0'

        self.side = side
        self.mano_layer = ManoLayer(mano_root=MANO_PATH, side=side,
                            use_pca=False, flat_hand_mean=True, joint_rot_mode='axisang').to(self.device)
        self.learning_rate = 1e-3
        self.joint_loss = torch.nn.MSELoss()
        if side=='right':
            self._active_img = self.RIGHT_JOINT_IMG
            self._nactive_img = self.LEFT_JOINT_IMG
        else:
            self._nactive_img = self.RIGHT_JOINT_IMG
            self._active_img = self.LEFT_JOINT_IMG
        self.init_joint_mask()
        
        

        # self.point_loss = o3d.t.pipelines.registration.TransformationEstimationPointToPoint().compute_rmse
        if shape_param is None:
            shape_param = torch.zeros(10).to(self.device)
        self.reset(shape_param)
        self.init_verts2points()
        
    def undo(self):
        if len(self.undo_stack) > 0:
            state = self.undo_stack.pop()
            self.set_state(state, only_pose=True)
            self.redo_stack.append(state)
            return True
        else:
            return False
    def redo(self):
        if len(self.redo_stack) > 0:
            state = self.redo_stack.pop()
            self.set_state(state, only_pose=True)
            self.undo_stack.append(state)
            return True
        else:
            return False
    def save_undo(self, forced=False):
        if len(self.redo_stack) > 0:
            self.redo_stack = []
            self.undo_stack = []
        if forced or ((time.time()-self._last_undo) > 1):
            if len(self.undo_stack) > 1000:
                self.undo_stack.pop(0)
            self.undo_stack.append(self.get_state())
            self._last_undo = time.time()
        
    def reset(self, shape_param=None, flat_hand=True):
        #1. shape
        if shape_param is None:
            pass
        else:
            self.shape_param = torch.Tensor(shape_param).unsqueeze(0).to(self.device)

        #2. root translation
        self.root_trans = torch.zeros(1, 3).to(self.device)
        
        #3. root, 15 joints
        if flat_hand:
            self.joint_rot = [torch.zeros(1, 3).to(self.device) for _ in range(16)]
        else:
            self.joint_rot = [torch.zeros(1, 3).to(self.device)]
            self.joint_rot += [torch.Tensor(self.mano_layer.smpl_data['hands_mean'][3*i:3*(i+1)]).unsqueeze(0).to(self.device) for i in range(15)]
            
        self.optimizer = None
        self.icp_optimizer = optim.Adam([self.root_trans, *self.joint_rot], lr=0.01)
        self.update_mano()
        self.root_delta = self.joints.cpu().detach()[0, 0].to(self.device)
        self.reset_target()
        
        self.optimize_state = LabelingMode.STATIC
        self.optimize_idx = self._IDX_OF_HANDS
        self.optimize_target = 'none'
        self.active_joints = None
        self.contorl_joint = None
        self.control_idx = -1
        self.undo_stack = []
        self.redo_stack = []
        self._last_undo = time.time()
        self.save_undo(forced=True)
    
    #region mano model
    def update_mano(self):
        pose_param = torch.concat(self.joint_rot, dim=1)
        verts, joints = self.mano_layer(th_pose_coeffs=pose_param,
                                        th_betas=self.shape_param,
                                        th_trans=self.root_trans)
        # self.verts = verts
        # self.joints = joints
        
        self.verts = verts / 1000
        self.joints = joints / 1000
        self.verts = self.verts.to(torch.float32)
        self.joints = self.joints.to(torch.float32)
        self.faces = self.mano_layer.th_faces
    
    def _get_control_joint_param_idx(self):
        if self.optimize_target=='root':
            target_idx = 0
        else:
            target_idx = 1
            if self.optimize_target=='thumb':
                target_idx += self._ORDER_OF_PARAM['thumb']*3+self.control_idx
            elif self.optimize_target=='fore':
                target_idx += self._ORDER_OF_PARAM['fore']*3+self.control_idx
            elif self.optimize_target=='middle':
                target_idx += self._ORDER_OF_PARAM['middle']*3+self.control_idx
            elif self.optimize_target=='ring':
                target_idx += self._ORDER_OF_PARAM['ring']*3+self.control_idx
            elif self.optimize_target=='little':
                target_idx += self._ORDER_OF_PARAM['little']*3+self.control_idx
            else:
                raise NotImplementedError
        return target_idx
    def _get_control_joint_idx(self):
        return self.optimize_idx[self.optimize_target][self.control_idx]
    def get_control_joint_name(self):
        name = ""
        if self.side == 'right':
            name += "오른손 "
        else:
            name += "왼손 "
        if self.optimize_target=='root':
            name += "손목"
            return name
        elif self.optimize_target=='thumb':
            name += "엄지 "
        elif self.optimize_target=='fore':
            name += "검지 "
        elif self.optimize_target=='middle':
            name += "중지 "
        elif self.optimize_target=='ring':
            name += "약지 "
        elif self.optimize_target=='little':
            name += "소지 "
        
        if self.control_idx==0:
            name += "첫번째 관절"
        elif self.control_idx==0:
            name += "두번째 관절"
        elif self.control_idx==0:
            name += "세번째 관절"

        return name

    def reset_pose(self, flat_hand=True):
        if self.optimize_state==LabelingMode.OPTIMIZE:
            if self.optimize_target=='root':
                self.joint_rot[0] = torch.zeros(1, 3).to(self.device)
                self.joint_rot[0].requires_grad = True
            else:
                for target_idx in [self._ORDER_OF_PARAM[self.optimize_target]*3+i+1 for i in range(3)]:
                    if flat_hand:
                        self.joint_rot[target_idx] = torch.zeros(1, 3).to(self.device)
                    else:
                        self.joint_rot[target_idx] = torch.Tensor(self.mano_layer.smpl_data['hands_mean'][3*target_idx-3:3*target_idx]).unsqueeze(0).to(self.device)
                    self.joint_rot[target_idx].requires_grad = True
            self.optimizer = optim.Adam(self.joint_rot, lr=self.learning_rate)
            self.update_mano()
        elif self.optimize_state == LabelingMode.OPTIMIZE_SHAPE:
            self.shape_param = torch.zeros(1, 10)
            self.shape_param.requires_grad = True
        else:
            target_idx = self._get_control_joint_param_idx()
            self.joint_rot[target_idx] = torch.zeros(1, 3).to(self.device)
            self.update_mano()    
    
    def reset_shape(self):
        self.shape_param = torch.zeros(1, 10).to(self.device)
        self.update_mano()

    def get_control_rotation(self):
        target_idx = self._get_control_joint_param_idx()
        return self.joint_rot[target_idx].cpu().detach()[0, :]
    def set_control_rotation(self, rot_mat):
        assert (self.optimize_state==LabelingMode.STATIC or self.optimize_target=='root'), "error on set_control_rotation"
        target_idx = self._get_control_joint_param_idx()
        if self.optimize_state==LabelingMode.OPTIMIZE and self.optimize_target=='root':
            self.joint_rot[0] = torch.Tensor(rot_mat).unsqueeze(0).to(self.device)
            self.joint_rot[0].requires_grad = True
            self.update_mano()
            self.reset_target()
        else:    
            self.joint_rot[target_idx] = torch.Tensor(rot_mat).unsqueeze(0).to(self.device)
            self.update_mano()
    
    def get_control_position(self):
        target_idx = self._get_control_joint_idx()
        if self.optimize_state==LabelingMode.STATIC:
            return self.joints.cpu().detach()[0, target_idx]
        else:
            return self.targets.cpu().detach()[0, target_idx]
    def set_control_position(self, xyz):
        assert (self.optimize_state==LabelingMode.OPTIMIZE or self.optimize_target=='root'), "error on set_control_position"
        if self.contorl_joint is None:
            return False
        
        if self.optimize_target == 'root':
            self.set_root_position(xyz)
            self.reset_target()
        else:
            joints = self.get_target()
            joints[self.contorl_joint] = torch.Tensor(xyz)
            self.set_target(joints)
        return True

    def get_root_position(self):
        return self.root_trans[0, :] + self.root_delta
    def set_root_position(self, xyz):
        self.root_trans = torch.Tensor(xyz).unsqueeze(0).to(self.device) - self.root_delta
        self.update_mano()
    
    def get_optimize_state(self):
        return self.optimize_state
    def set_optimize_state(self, state):
        self.optimize_state = state
        self.reset_target()
        if self.optimize_state==LabelingMode.OPTIMIZE:
            self.optimize_idx = self._IDX_OF_GUIDE
            self.set_optimize_target('root')
            self.update_mano()
        elif self.optimize_state==LabelingMode.STATIC:
            self.optimize_idx = self._IDX_OF_HANDS
            self.set_optimize_target('root')
            for param in self.joint_rot:
                param.requires_grad = False
            self.optimizer = None
        elif self.optimize_state==LabelingMode.OPTIMIZE_SHAPE:
            self.root_trans.requires_grad = True
            for param in self.joint_rot:
                param.requires_grad = True
            self.shape_param.requires_grad = True
            self.optimizer = optim.Adam([self.root_trans, *self.joint_rot, self.shape_param], lr=0.001)
        else:
            raise NotImplementedError

    def get_optimize_target(self):
        return self.optimize_target
    def set_optimize_target(self, target):
        self.optimize_target = target
        for param in self.joint_rot:
            param.requires_grad = False
        if self.optimize_target=='root':
            self.joint_rot[0].requires_grad = True
        else:
            for target_idx in [self._ORDER_OF_PARAM[self.optimize_target]*3+i+1 for i in range(3)]:
                self.joint_rot[target_idx].requires_grad = True
        self.optimizer = optim.Adam(self.joint_rot, lr=self.learning_rate)
        self.active_joints = self.optimize_idx[self.optimize_target]
        self.control_idx = 0
        self.contorl_joint = self.active_joints[0]
    
    def set_control_joint(self, idx):
        assert len(self.active_joints) > 0, "set_control_joint error"
        idx = np.clip(idx, 0, len(self.active_joints)-1) 
        self.control_idx = idx
        self.contorl_joint = self.active_joints[idx]
        
    #region save and load state
    def get_state(self):
        pose_param = torch.concat(self.joint_rot, dim=1) # 1, 48, 3
        return {
            'shape_param': np.array(self.shape_param.cpu().detach()[0, :]),
            'pose_param': np.array(pose_param.cpu().detach()[0, :]), # 48, 3
            'root_trans': np.array(self.root_trans.cpu().detach()[0, :]),
            'root_delta': np.array(self.root_delta.cpu().detach()),
            
            'joints': np.array(self.joints.cpu().detach()[0, :]),
            'verts': np.array(self.verts.cpu().detach()[0, :]),
            'faces': np.array(self.faces.cpu().detach()[0, :])
        }
    def set_state(self, state, only_pose=False):
        if only_pose:
            pose_param = torch.Tensor(state['pose_param']).unsqueeze(0).to(self.device) # 1, 48
            self.joint_rot = [pose_param[:, 3*i:3*(i+1)] for i in range(16)]
            self.root_trans = torch.Tensor(state['root_trans']).unsqueeze(0).to(self.device)
            self.update_mano()
        else:
            self.shape_param = torch.Tensor(state['shape_param']).unsqueeze(0).to(self.device)
            assert state['pose_param'].size==48
            pose_param = torch.Tensor(state['pose_param']).unsqueeze(0).to(self.device) # 1, 48
            self.joint_rot = [pose_param[:, 3*i:3*(i+1)] for i in range(16)]
            self.root_trans = torch.Tensor(state['root_trans']).unsqueeze(0).to(self.device)
            self.root_delta = torch.Tensor(state['root_delta']).to(self.device)
        
            self.update_mano()
            
            self.optimize_state = LabelingMode.STATIC
            self.optimize_idx = self._IDX_OF_HANDS
            self.optimize_target = 'none'
            self.active_joints = None
            self.contorl_joint = None
            self.control_idx = -1
            self.undo_stack = []
            self.redo_stack = []
            self._last_undo = time.time()
            self.save_undo(forced=True)
    
    def set_joint_pose(self, pose):
        pose_param = torch.Tensor(pose).unsqueeze(0) # 48 -> 1, 48
        self.joint_rot = [pose_param[:, 3*i:3*(i+1)] for i in range(16)]
        self.update_mano()
    def get_joint_pose(self):
        pose_param = torch.concat(self.joint_rot, dim=1) # 1, 48, 3
        pose = np.array(pose_param.cpu().detach()[0, :])
        return pose
    def get_hand_position(self):
        pose = np.array(self.joints.cpu().detach()[0, :])
        return pose.tolist()
    def get_hand_pose(self):
        pose_param = torch.concat([self.root_trans, *self.joint_rot], dim=1) # 1, 51, 3
        pose = np.array(pose_param.cpu().detach()[0, :])
        return pose.tolist()
    def get_hand_shape(self):
        shape = np.array(self.shape_param.cpu().detach()[0, :])
        return shape.tolist()
    #endregion

    
    #region joint guide 
    def reset_target(self):
        self.targets = torch.empty_like(self.joints).copy_(self.joints)
        self._target_changed = False
    
    def optimize_to_target(self):
        if self._target_changed:
            self.optimizer.zero_grad()
            # forward
            self.update_mano()
            # loss term
            loss = self._mse_loss()
            loss.backward()
            self.optimizer.step()
            self.update_mano()
            return True
        else:
            return False
    
    def optimize_to_points(self, target_points):
        # previous_grad = []
        # for idx, rot_param in enumerate(self.joint_rot):
        #     previous_grad.append(rot_param.requires_grad)
        #     rot_param.requires_grad = True
        icp_optimizer = optim.Adam(self.joint_rot, lr=0.001)
        for _ in range(10):
            icp_optimizer.zero_grad()
            self.update_mano()
            loss = self._p2p_loss(target_points)
            loss.backward()
            print("loss: ", loss)
            icp_optimizer.step()
        
        # for idx, rot_param in enumerate(self.joint_rot):
        #     rot_param.requires_grad = previous_grad[idx]
        self.update_mano()
    
    def optimize_shape(self, target_points):
        self.shape_param.requires_grad = True
        self.optimizer.zero_grad()
        self.update_mano()
        loss = self._mesh_to_points_loss(target_points)
        print("{}".format(loss))
        loss.backward()
        self.optimizer.step()

        # with torch.no_grad():
        # #     self.shape_param -= torch.clip(0.1*self.shape_param.grad, -0.01, 0.01)
        # #     print("loss: ", loss)
        
            # self.update_mano()

    def get_target(self):
        return self.targets.cpu().detach()[0, :]
    
    def set_target(self, targets):
        self.targets = torch.Tensor(targets).unsqueeze(0)
        self.targets.requires_grad = True
        self._target_changed = True

    def _mse_loss(self):
        assert self.optimize_state==LabelingMode.OPTIMIZE
        target_idx = self.optimize_idx[self.optimize_target]
        return self.joint_loss(self.joints[:, target_idx], self.targets[:, target_idx].to(self.device))
    
    def _mesh_to_points_loss(self, target_points):
        target_points = Pointclouds(torch.Tensor(target_points).unsqueeze(0)).to(self.device)
        meshes = self.get_py3d_mesh().to(self.device)
        return point_mesh_face_distance(meshes, target_points)
    
    def _p2p_loss(self, target_points):
        p1 = self._sampling_points_from_mesh()
        # print(p1)
        p2 = torch.Tensor(target_points).to(self.device)
        dist = Utils.distance_matrix(p1, p2)
        # print(dist)
        min_dist, _ = torch.min(dist, dim=1)
        min_dist = torch.sum(min_dist)
        # print(min_dist)
        
        return min_dist
    
    def _sampling_points_from_mesh(self):
        verts = self.verts[0]
        face_verts = torch.matmul(self.verts2points, verts)
        points = torch.concat((verts, face_verts), dim=0)
        return points

    def set_learning_rate(self, lr):
        self.learning_rate = lr
        if self.optimizer is None:
            return
        for g in self.optimizer.param_groups:
            g['lr'] = lr
    #endregion

    #region open3d geometry
    def get_geometry(self):
        return {
            "mesh": self._get_mesh(),
            "joint": self._get_joints(),
            "link": self._get_links(),
        }
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
            joints = self.joints.cpu().detach()[0, :]
        else:
            joints = self.joints.cpu().detach()[0, idx]
        joints = o3d.utility.Vector3dVector(joints)
        pcd = o3d.geometry.PointCloud(points=joints)
        return pcd
    def _get_links(self):
        joints = self.joints.cpu().detach()[0, :]
        joints = o3d.utility.Vector3dVector(joints)
        lines = o3d.utility.Vector2iVector(np.array(HandModel.LINK))
        lineset = o3d.geometry.LineSet(lines=lines, points=joints)
        
        return lineset
    def _get_optim_points(self):
        p1 = self._sampling_points_from_mesh()
        p1 = p1.cpu().detach()
        p1 = o3d.utility.Vector3dVector(p1)
        pcd = o3d.geometry.PointCloud(points=p1)
        return pcd
    
    def get_py3d_mesh(self):
        verts = self.verts
        faces = self.faces.unsqueeze(0)
        return Meshes(verts=verts, faces=faces)

    def get_target_geometry(self):
        return {
            "joint": self._get_target_joints(),
            "link": self._get_target_links(),
            "optim_points": self._get_optim_points()
        }
    def _get_target_joints(self, idx=None):
        if idx is None:
            joints = self.targets.cpu().detach()[0, :]
        else:
            joints = self.targets.cpu().detach()[0, idx]
        joints = o3d.utility.Vector3dVector(joints)
        pcd = o3d.geometry.PointCloud(points=joints)
        return pcd
    
    def _get_target_links(self):
        joints = self.targets.cpu().detach()[0, :]
        joints = o3d.utility.Vector3dVector(joints)
        lines = o3d.utility.Vector2iVector(np.array(HandModel.LINK))
        lineset = o3d.geometry.LineSet(lines=lines, points=joints)
        
        return lineset

    def get_active_geometry(self):
        if self.optimize_state==LabelingMode.OPTIMIZE:
            return {
            "joint": self._get_target_joints(self.active_joints),
            "control": self._get_target_joints([self.contorl_joint])
        }
        elif self.optimize_state==LabelingMode.STATIC or self.optimize_state==LabelingMode.OPTIMIZE_SHAPE:
            return {
            "control": self._get_joints([self.contorl_joint])
        }
        else:
            raise NotImplementedError

    def init_joint_mask(self):
        total_mask = np.zeros((180, 340, 3), np.uint8)
        mask = (total_mask > 0)[:, :, 0]
        self.joint_mask = {
            'active': {},
            'nactive': {}
        }
        for idx, (j, i) in self._active_img.items():
            temp = mask.copy()
            xs, xe = max(i-5, 0), min(i+5, 180)
            ys, ye = max(j-5, 0), min(j+5, 340)
            temp[xs:xe, ys:ye] = True
            self.joint_mask['active'][idx] = temp
            total_mask[xs:xe, ys:ye] = [0, 0, 255]
        for idx, (j, i) in self._nactive_img.items():
            temp = mask.copy()
            xs, xe = max(i-5, 0), min(i+5, 180)
            ys, ye = max(j-5, 0), min(j+5, 340)
            temp[xs:xe, ys:ye] = True
            self.joint_mask['nactive'][idx] = temp
            total_mask[xs:xe, ys:ye] = [255, 255, 255]
        self.total_mask = total_mask
    def init_verts2points(self):
        max_vert = self.verts.shape[1]
        faces = self.faces
        n = 8
        verts2points = torch.zeros((faces.shape[0]*(n+1), max_vert)).to(self.device)
        # center
        for i, idx in enumerate(faces):
            verts2points[i*(n+1), idx] = 1/3
        # random N points
        for i, idx in enumerate(faces):
            for j in range(n):
                weight = torch.rand(3).to(self.device)
                weight /= torch.sum(weight)
                verts2points[i*(n+1) + j + 1, idx] = weight
        self.verts2points = verts2points

    def get_joint_mask(self):
        mask = self.total_mask.copy()
        mask[self.joint_mask['active'][self.contorl_joint]] = [255, 0, 0]
        return mask
        
class PoseTemplate:
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_template")
    def __init__(self, side='right'):
        self.template_dir = os.path.join(self.template_dir, side)
        if not os.path.isdir(self.template_dir):
            os.makedirs(self.template_dir)
        self.template = {}
        for temp_npy in [os.path.join(self.template_dir, p) for p in os.listdir(self.template_dir)]:
            try:
                template_name = os.path.basename(temp_npy).replace(".npy", "")
                self.template[template_name] = np.load(temp_npy)
            except:
                continue
    def save_pose2template(self, name, pose):
        self.template[name] = pose
        npy_path = os.path.join(self.template_dir, "{}.npy".format(name))
        np.save(npy_path, pose)
    def get_template2pose(self, name):
        return self.template[name]
    def get_template_list(self):
        return list(self.template.keys())
    def remove_template(self, name):
        del(self.template[name])
        os.remove(os.path.join(self.template_dir, "{}.npy".format(name)))
 
class Scene:
    def __init__(self, file_path):
        self._pcd_path = file_path
        self._hands = {
            'right': HandModel('right'),
            'left': HandModel('left'),
        }
        
        self._json_path = file_path.replace('.ply', '.json')
        self._label_path = file_path.replace('.ply', '.npz')
        self._pcd = self._load_point_cloud(file_path)
        self._pcd.scale(0.001, [0, 0, 0])
        # self._pcd = self._pcd.voxel_down_sample(0.003)
        self._label = None


    @staticmethod
    def _load_point_cloud(pc_file) -> o3d.geometry.PointCloud:
        pcd = o3d.io.read_point_cloud(pc_file)
        if pcd is not None:
            print("[Info] Successfully read scene ")
            if not pcd.has_normals():
                pcd.estimate_normals()
            pcd.normalize_normals()
        else:
            print("[WARNING] Failed to read points")
        
        return pcd
    
    def save_json(self):
        json_path = self._json_path
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                try:
                    json_label = json.load(f)
                except:
                    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = json_path.replace(".json", "_backup_{}.json".format(date_time))
                    shutil.copy(json_path, backup_path)
                    json_label = {}
        else:
            json_label = {}
        json_label.setdefault('hand_mask_info', {})
        for side, hand_model in self._hands.items():
            json_label[str(side)] = hand_model.get_hand_shape()
        with open(json_path, 'w') as f:
            json.dump(json_label, f, sort_keys=True, indent=4)

    def save_label(self):
        # get current label
        label = {}
        for side, hand_model in self._hands.items():
            h_state = hand_model.get_state()
            for k, v in h_state.items():
                label['{}_{}'.format(side, k)] = v
        np.savez(self._label_path, **label)
        self.save_json()
        self._label = label
    
    def load_label(self):
        # hand label
        if self._label is not None: # default is previous frame
            self._previous_label = self._label.copy()
        try:
            self._label = dict(np.load(self._label_path))
            hand_states = {}
            for k, v in self._label.items():
                side = k.split('_')[0]
                hand_states.setdefault(side, {})
                param = k.replace(side + "_", "")
                hand_states[side][param] = v
            for side, state in hand_states.items():
                self._hands[side].set_state(state)
            return True
        except:
            print("Fail to load AI Label")
            for hand_model in self._hands.values():
                hand_model.reset()
            return False
    
class Settings:
    SHADER_UNLIT = "defaultUnlit"
    SHADER_LINE = "unlitLine"
    SHADER_LIT_TRANS = "defaultLitTransparency"

    def __init__(self):
        self.bg_color = gui.Color(1, 1, 1)
        self.show_axes = False
        self.show_coord_frame = False
        self.show_hand = True
        self.show_objects = True
        self.show_pcd = True
        self.point_transparency =0
        self.hand_transparency =0.2
        self.obj_transparency = 0.5

        # ----- Material Settings -----
        self.apply_material = True  # clear to False after processing

        # ----- scene material
        self.scene_material = rendering.MaterialRecord()
        self.scene_material.base_color = [1.0, 1.0, 1.0, 1.0]
        self.scene_material.base_reflectance = 0
        self.scene_material.base_roughness = 0.5
        self.scene_material.shader = Settings.SHADER_UNLIT
        self.scene_material.point_size = 1.0

        # ----- hand model setting
        self.hand_mesh_material = rendering.MaterialRecord()
        self.hand_mesh_material.base_color = [0.8, 0.8, 0.8, 1.0-self.hand_transparency]
        self.hand_mesh_material.shader = Settings.SHADER_LIT_TRANS
        # self.hand_mesh_material.line_width = 2.0
        
        self.hand_joint_material = rendering.MaterialRecord()
        self.hand_joint_material.base_color = [1.0, 0.0, 0.0, 1.0]
        self.hand_joint_material.shader = Settings.SHADER_UNLIT
        self.hand_joint_material.point_size = 5.0
        
        self.hand_link_material = rendering.MaterialRecord()
        self.hand_link_material.base_color = [1.0, 0.0, 0.0, 1.0]
        self.hand_link_material.shader = Settings.SHADER_LINE
        self.hand_link_material.line_width = 2.0
        
        self.active_hand_mesh_material = rendering.MaterialRecord()
        self.active_hand_mesh_material.base_color = [0.0, 1.0, 0.0, 1.0-self.hand_transparency]
        self.active_hand_mesh_material.shader = Settings.SHADER_LIT_TRANS
        # self.active_hand_mesh_material.line_width = 2.0

        # ----- hand label setting
        self.target_joint_material = rendering.MaterialRecord()
        self.target_joint_material.base_color = [0.0, 0.0, 1.0, 1.0]
        self.target_joint_material.shader = Settings.SHADER_UNLIT
        self.target_joint_material.point_size = 10.0
        
        self.target_link_material = rendering.MaterialRecord()
        self.target_link_material.base_color = [0.0, 0.0, 1.0, 1.0]
        self.target_link_material.shader = Settings.SHADER_LINE
        self.target_link_material.line_width = 3.0
        
        self.active_target_joint_material = rendering.MaterialRecord()
        self.active_target_joint_material.base_color = [1.0, 0.75, 0.75, 1.0]
        self.active_target_joint_material.shader = Settings.SHADER_UNLIT
        self.active_target_joint_material.point_size = 20.0
        
        self.active_target_link_material = rendering.MaterialRecord()
        self.active_target_link_material.base_color = [0.0, 0.7, 0.0, 1.0]
        self.active_target_link_material.shader = Settings.SHADER_LINE
        self.active_target_link_material.line_width = 5.0
        
        self.control_target_joint_material = rendering.MaterialRecord()
        self.control_target_joint_material.base_color = [0.0, 1.0, 0.0, 1.0]
        self.control_target_joint_material.shader = Settings.SHADER_UNLIT
        self.control_target_joint_material.point_size = 30.0
        
        self.coord_material = rendering.MaterialRecord()
        self.coord_material.base_color = [1.0, 1.0, 1.0, 1.0]
        self.coord_material.shader = Settings.SHADER_LINE
        self.coord_material.point_size = 2.0
        
        # object 
        self.obj_material = rendering.MaterialRecord()
        self.obj_material.base_color = [0, 0, 1, 1 - self.obj_transparency]
        self.obj_material.shader = Settings.SHADER_LIT_TRANS

class HeadlessRenderer:
    flat_hand_position = {'right': [[0.0956699401140213, 0.0063834283500909805, 0.006186304613947868], [0.12148278951644897, -0.009138907305896282, 0.030276024714112282], [0.14518234133720398, -0.008247620426118374, 0.04990927129983902], [0.1597064584493637, -0.013680592179298401, 0.07212700694799423], [0.1803981363773346, -0.01809500716626644, 0.09962499886751175], [0.11635593324899673, 0.0011830711737275124, 0.0942835733294487], [0.11857300251722336, 0.005192427430301905, 0.12696248292922974], [0.11845888197422028, 0.003894004039466381, 0.14911839365959167], [0.12250815331935883, 0.004611973185092211, 0.17238116264343262], [0.09231240302324295, 0.00490446574985981, 0.1008467748761177], [0.08671789616346359, 0.006765794008970261, 0.1320294439792633], [0.08277337998151779, 0.0055136894807219505, 0.1549340784549713], [0.07744278013706207, 0.006146649364382029, 0.180849090218544], [0.06899674236774445, 0.0024260072968900204, 0.0879218801856041], [0.06389820575714111, 0.004493014886975288, 0.11623615771532059], [0.05626438930630684, 0.002804903080686927, 0.1397566795349121], [0.049281422048807144, 0.00734306126832962, 0.16266049444675446], [0.052460599690675735, -0.0035569006577134132, 0.07497329264879227], [0.039961814880371094, -0.0034950755070894957, 0.09198770672082901], [0.029629912227392197, -0.004186231642961502, 0.10785461217164993], [0.019351951777935028, -0.001628118334338069, 0.12375511229038239]], 'left': [[-0.0956699401140213, 0.0063834283500909805, 0.006186304613947868], [-0.12148278951644897, -0.009138907305896282, 0.030276024714112282], [-0.14518234133720398, -0.008247620426118374, 0.04990927129983902], [-0.1597064584493637, -0.013680592179298401, 0.07212700694799423], [-0.1803981363773346, -0.01809500716626644, 0.09962499886751175], [-0.11635593324899673, 0.0011830711737275124, 0.0942835733294487], [-0.11857300251722336, 0.005192427430301905, 0.12696248292922974], [-0.11845888197422028, 0.003894004039466381, 0.14911839365959167], [-0.12250815331935883, 0.004611973185092211, 0.17238116264343262], [-0.09231240302324295, 0.00490446574985981, 0.1008467748761177], [-0.08671789616346359, 0.006765794008970261, 0.1320294439792633], [-0.08277337998151779, 0.0055136894807219505, 0.1549340784549713], [-0.07929610460996628, 0.010507885366678238, 0.17927193641662598], [-0.06899674236774445, 0.0024260072968900204, 0.0879218801856041], [-0.06389820575714111, 0.004493014886975288, 0.11623615771532059], [-0.05626438930630684, 0.002804903080686927, 0.1397566795349121], [-0.049281422048807144, 0.00734306126832962, 0.16266049444675446], [-0.052460599690675735, -0.0035569006577134132, 0.07497329264879227], [-0.039961814880371094, -0.0034950755070894957, 0.09198770672082901], [-0.029629912227392197, -0.004186231642961502, 0.10785461217164993], [-0.019351951777935028, -0.001628118334338069, 0.12375511229038239]]}
    
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.render = rendering.OffscreenRenderer(width=self.W, height=self.H)
        self.render.scene.set_background([0, 0, 0, 0]) # black background color
        self.render.scene.set_lighting(self.render.scene.LightingProfile.NO_SHADOWS, [0,0,0])

        self.obj_mtl = o3d.visualization.rendering.MaterialRecord()
        self.obj_mtl.shader = "defaultUnlit"
        
        self.hand_mtl = o3d.visualization.rendering.MaterialRecord()
        self.hand_mtl.shader = "defaultUnlit"
        
    def add_objects(self, objects, color=[1, 0, 0]):
        for obj_id, obj in objects.items():
            obj_name = "obj_{}".format(obj_id)
            geo = obj.get_geometry()
            geo = copy.deepcopy(geo)
            geo.paint_uniform_color(color)
            self.render.scene.add_geometry(obj_name, geo, self.obj_mtl)
    
    def add_hands(self, hands):
        for side, hand in hands.items():
            hand_geo = hand.get_geometry()
            geo = hand_geo['mesh']
            geo = copy.deepcopy(geo)
            if side=='right':
                geo.paint_uniform_color([0, 1, 0])
            else:    
                geo.paint_uniform_color([0, 0, 1])
            self.render.scene.add_geometry(side, geo, self.hand_mtl)

    def set_camera(self, intrinsic, extrinsic, W, H):
        self.render.setup_camera(intrinsic, extrinsic, W, H)
        center = np.dot(extrinsic, np.array([0, 0, 1, 1]))[:3]  # look_at target 
        eye = np.dot(extrinsic, np.array([0, 0, 0, 1]))[:3]  # camera position
        up = np.dot(extrinsic, np.array([0, -1, 0, 0]))[:3]  # camera rotation
        self.render.scene.camera.look_at(center, eye, up)
        self.render.scene.camera.set_projection(intrinsic, 0.01, 3.0, W, H)

    def render_depth(self):
        return self.render.render_to_depth_image(z_in_view_space=True)
    
    def render_rgb(self):
        return self.render.render_to_image()
    
    def reset(self):
        self.render.scene.clear_geometry()

class AppWindow:
    
    
    def __init__(self, width, height):
        #---- geometry name
        self._window_name = "3D Hand Pose Annotator by GIST AILAB"
        self._scene_name = "annotation_scene"
        
        #----- hand model geometry name
        self._right_hand_mesh_name = "right_hand_mesh"
        self._right_hand_joint_name = "right_hand_joint"
        self._right_hand_link_name = "right_hand_link"
        self._left_hand_mesh_name = "left_hand_mesh"
        self._left_hand_joint_name = "left_hand_joint"
        self._left_hand_link_name = "left_hand_link"
        #----- target geometry name
        self._target_joint_name = "target_joint"
        self._target_link_name = "target_link"
        
        #----- active/control geometry name
        self._active_joint_name = "active_joint"
        self._active_link_name = "active_link"
        self._control_joint_name = "control_joint"
        
        #----- intialize values
        self.dataset = None
        self.annotation_scene = None
        
        self._labeling_mode = LabelingMode.STATIC
        self._pcd = None
        self._hands = None
        self._active_hand = None
        self.upscale_responsiveness = False
        self._left_shift_modifier = False
        self._annotation_changed = False
        self._last_change = time.time()
        self._last_saved = time.time()
        self.coord_labels = []
        self._objects = None
        self._object_names = []
        self._camera_idx = -1
        self._cam_name_list = []
        self.scale_factor = None
        self.reset_flat = False
        self.joint_back = False
        
        
        self._template = {
            "right": PoseTemplate(side='right'),
            "left": PoseTemplate(side='left')
            } 
        self._activate_template = None
        
        self.window = gui.Application.instance.create_window(self._window_name, width, height)
        w = self.window
        
        self.settings = Settings()
        
        # 3D widget
        self._scene = gui.SceneWidget()
        scene = rendering.Open3DScene(w.renderer)
        scene.set_lighting(scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
        self._scene.scene = scene
        
        # ---- Settings panel
        em = w.theme.font_size
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        
        self._init_fileeidt_layout()
        self._init_viewctrl_layout()
        self._init_handedit_layout()
        self._init_stageedit_layout()
        self._init_scene_control_layout()
        self._init_label_control_layout()
        
        # ---- image viewer panel
        self._images_panel = gui.CollapsableVert("이미지 보기", 0.33 * em,
                                                 gui.Margins(em, 0, 0, 0))
        
        self._init_image_view_layout()

        # ---- validation panel
        self._validation_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        
        self._init_show_error_layout()
        self._init_preset_layout()
        
        # ---- log panel
        self._log_panel = gui.VGrid(1, em)
        self._log = gui.Label("\t 라벨링 대상 파일을 선택하세요. ")
        self._log_panel.add_child(self._log)
        
        
        # 3D Annotation tool options
        w.add_child(self._scene)
        
        w.add_child(self._settings_panel)
        w.add_child(self._log_panel)
        w.add_child(self._images_panel)
        w.add_child(self._validation_panel)
        w.set_on_layout(self._on_layout)
        
        # ---- annotation tool settings ----
        self._initialize_background()
        self._on_scene_point_size(5) # set default size to 1
        # self._on_point_transparency(0)
        self._on_object_transparency(0.5)
        self._on_hand_transparency(0.2)
        self._on_hand_point_size(10) # set default size to 10
        self._on_hand_line_size(2) # set default size to 2
        self._on_responsiveness(5) # set default responsiveness to 5
        
        self._scene.set_on_mouse(self._on_mouse)
        self._scene.set_on_key(self._on_key)
        
        self.window.set_on_tick_event(self._on_tick)
        self._log.text = "\t라벨링 대상 파일을 선택하세요."
        self.window.set_needs_layout()

    #region Debug
    def _on_error(self, err_msg):
        dlg = gui.Dialog("Error")

        em = self.window.theme.font_size
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(err_msg))

        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)
    
    def _on_about_ok(self):
        self.window.close_dialog()
        self.window.set_needs_layout()
    
    def _check_annotation_scene(self):
        if self.annotation_scene is None:
            self._on_error("라벨링 대상 파일을 선택하세요.")
            return False
        return True
    #endregion
    
    #region Layout and Callback
    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        width_set = 17 * layout_context.theme.font_size
        height_set = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width_set, r.y, width_set,
                                              height_set)

        width_val = 20 * layout_context.theme.font_size
        height_val = min(
            r.height,
            self._validation_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
    
        self._validation_panel.frame = gui.Rect(r.get_right() - width_set - width_val, r.y, width_val,
                                              height_val)                    
        width_im = min(
            r.width,
            self._images_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).width * 1.1)
        height_im = min(
            r.height,
            self._images_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height * 1.1)                     
        self._images_panel.frame = gui.Rect(0, r.y, width_im, height_im)   
        self.image_panel_xywh = [0, r.y, width_im, height_im]
        width_obj = 1.5 * width_set
        height_obj = 1.5 * layout_context.theme.font_size
        self._log_panel.frame = gui.Rect(0, r.get_bottom() - height_obj, width_obj, height_obj) 
    
    def _initialize_background(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._set_background_color(bg_color)

    def _on_initial_viewpoint(self):
        if self.bounds is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_initial_viewpoint)")
            return
        self._log.text = "\t 처음 시점으로 이동합니다."
        self.window.set_needs_layout()
        self._scene.setup_camera(60, self.bounds, self.bounds.get_center())
        center = np.array(self.bounds.get_center())
        eye = np.array([0, 0, 0])
        up = np.array([0, -1, 0])
        self._scene.look_at(center, eye, up)
        self._init_view_control()
    def _on_active_viewpoint(self):
        if self.bounds is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_initial_viewpoint)")
            return
        self._log.text = "\t 조작중인 객체 시점으로 이동합니다."
        self.window.set_needs_layout()
        center = np.array([0, 0, 0])
        self._scene.setup_camera(60, self.bounds, center)
        eye_on = self._active_hand.get_control_position()
        center = eye_on - 0.4* eye_on/np.linalg.norm(eye_on)
        up = np.array([0, -1, 0])
        self._scene.look_at(eye_on, center, up)
        self._init_view_control()
    def _on_active_camera_viewpoint(self):
        if self._camera_idx==-1:
            self._on_initial_viewpoint()
            return
        cam_name = self._cam_name_list[self._camera_idx]
        self._log.text = "\t {} 시점으로 이동합니다.".format(cam_name)
        self.window.set_needs_layout()
        intrinsic = self._frame.cameras[cam_name].intrinsic
        extrinsic = self._frame.cameras[cam_name].extrinsics
        # self._scene.setup_camera(o3d.camera.PinholeCameraIntrinsic(self.W, self.H, intrinsic), extrinsic,  self.bounds)
        self._scene.setup_camera(o3d.camera.PinholeCameraIntrinsic(self.W, self.H, 
                                intrinsic[0, 0],intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]), extrinsic,  self.bounds)
        # master to active camera
        center = np.dot(extrinsic, np.array([0, 0, 1, 1]))[:3]  # look_at target 
        eye = np.dot(extrinsic, np.array([0, 0, 0, 1]))[:3]  # camera position
        up = np.dot(extrinsic, np.array([0, -1, 0, 0]))[:3]  # camera rotation
        self._scene.look_at(center, eye, up)
        self._init_view_control()

    def _init_view_control(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
    
    # file edit        
    def _init_fileeidt_layout(self):
        em = self.window.theme.font_size
        # ---- File IO
        fileedit_layout = gui.Horiz()
        fileedit_layout.add_fixed(0.25 * em)
        
        filedlgbutton = gui.Button("파일 열기")
        filedlgbutton.horizontal_padding_em = 0.5
        filedlgbutton.vertical_padding_em = 0
        filedlgbutton.set_on_clicked(self._on_filedlg_button)
        
        self._fileedit = gui.TextEdit()
        
        fileedit_layout.add_child(gui.Label("파일 경로"))
        fileedit_layout.add_child(self._fileedit)
        fileedit_layout.add_child(filedlgbutton)
        self._settings_panel.add_child(fileedit_layout)
    def _on_filedlg_button(self):
        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "파일 선택",
                                self.window.theme)
        filedlg.add_filter(".ply", "손 모양 포인트")
        filedlg.set_on_cancel(self._on_filedlg_cancel)
        filedlg.set_on_done(self._on_filedlg_done)
        self.window.show_dialog(filedlg)
    def _on_filedlg_cancel(self):
        self.window.close_dialog()
    def _on_filedlg_done(self, file_path):
        """Load file -> Scene

        Args:
            file_path (_type_): _description_
        """
        self._fileedit.text_value = file_path
        try:
            self.annotation_scene = Scene(file_path)
            ret = self.annotation_scene.load_label()
            self._hands = self.annotation_scene._hands
            self._pcd = self.annotation_scene._pcd
            if not ret:
                for s, hand_model in self._hands.items():
                    hand_model.set_root_position(self._pcd.get_center())
                
            self.bounds = self._pcd.get_axis_aligned_bounding_box()
            self._add_geometry(self._scene_name, self._pcd, self.settings.scene_material)
            self._init_hand_layer()
            self._on_initial_viewpoint()
            self.window.close_dialog()
            self._log.text = "\t 라벨링 대상 파일을 불러왔습니다."
        except Exception as e:
            print(e)
            self._on_error("잘못된 경로가 입력되었습니다. (error at _on_filedlg_done)")
            self._log.text = "\t 올바른 파일 경로를 선택하세요."

    # scene edit 편의 기능
    def _init_viewctrl_layout(self):
        em = self.window.theme.font_size
        
        viewctrl_layout = gui.CollapsableVert("편의 기능", 0.33*em,
                                          gui.Margins(em, 0, 0, 0))
        viewctrl_layout.set_is_open(True)
        
        self._show_hands = gui.Checkbox("손 라벨 보기 (Z)")
        self._show_hands.set_on_checked(self._on_show_hand)
        viewctrl_layout.add_child(self._show_hands)
        self._show_hands.checked = True

        self._auto_optimize = gui.Checkbox("자동 정렬 활성화 (X)")
        viewctrl_layout.add_child(self._auto_optimize)
        self._auto_optimize.checked = False

        self._show_objects = gui.Checkbox("물체 라벨 보기 (C)")
        self._show_objects.set_on_checked(self._on_show_object)
        viewctrl_layout.add_child(self._show_objects)
        self._show_objects.checked = True
        
        self._show_pcd = gui.Checkbox("포인트 보기 (V)")
        self._show_pcd.set_on_checked(self._on_show_pcd)
        viewctrl_layout.add_child(self._show_pcd)
        self._show_pcd.checked = True

        self._auto_save = gui.Checkbox("자동 저장 활성화")
        viewctrl_layout.add_child(self._auto_save)
        self._auto_save.checked = True
        
        grid = gui.VGrid(2, 0.25 * em)
        self._scene_point_size = gui.Slider(gui.Slider.INT)
        self._scene_point_size.set_limits(1, 20)
        self._scene_point_size.set_on_value_changed(self._on_scene_point_size)
        grid.add_child(gui.Label("포인트 크기"))
        grid.add_child(self._scene_point_size)
        
        # self._point_transparency = gui.Slider(gui.Slider.DOUBLE)
        # self._point_transparency.set_limits(0, 1)
        # self._point_transparency.set_on_value_changed(self._on_point_transparency)
        # grid.add_child(gui.Label("포인트 투명도"))
        # grid.add_child(self._point_transparency)
        
        self._object_transparency = gui.Slider(gui.Slider.DOUBLE)
        self._object_transparency.set_limits(0, 1)
        self._object_transparency.set_on_value_changed(self._on_object_transparency)
        grid.add_child(gui.Label("물체 투명도"))
        grid.add_child(self._object_transparency)
        
        self._hand_transparency = gui.Slider(gui.Slider.DOUBLE)
        self._hand_transparency.set_limits(0, 1)
        self._hand_transparency.set_on_value_changed(self._on_hand_transparency)
        grid.add_child(gui.Label("손 투명도"))
        grid.add_child(self._hand_transparency)
        
        self._hand_point_size = gui.Slider(gui.Slider.INT)
        self._hand_point_size.set_limits(1, 20)
        self._hand_point_size.set_on_value_changed(self._on_hand_point_size)
        grid.add_child(gui.Label("손 관절 크기"))
        grid.add_child(self._hand_point_size)
        
        self._hand_line_size = gui.Slider(gui.Slider.INT)
        self._hand_line_size.set_limits(1, 20)
        self._hand_line_size.set_on_value_changed(self._on_hand_line_size)
        grid.add_child(gui.Label("손 연결선 두께"))
        grid.add_child(self._hand_line_size)
        
        self._responsiveness = gui.Slider(gui.Slider.INT)
        self._responsiveness.set_limits(1, 20)
        self._responsiveness.set_on_value_changed(self._on_responsiveness)
        grid.add_child(gui.Label("민감도"))
        grid.add_child(self._responsiveness)
        
        self._optimize_rate = gui.Slider(gui.Slider.INT)
        self._optimize_rate.set_limits(1, 20) # 1-> 1e-3
        self._optimize_rate.double_value = 1
        self._optimize_rate.set_on_value_changed(self._on_optimize_rate)
        grid.add_child(gui.Label("최적화 속도"))
        grid.add_child(self._optimize_rate)
        
        self._auto_save_interval = gui.Slider(gui.Slider.INT)
        self._auto_save_interval.set_limits(1, 20) # 1-> 1e-3
        self._auto_save_interval.double_value = 5
        self._auto_save_interval.set_on_value_changed(self._on_auto_save_interval)
        grid.add_child(gui.Label("자동 저장 간격"))
        grid.add_child(self._auto_save_interval)
        
        viewctrl_layout.add_child(grid)
        
        self._settings_panel.add_child(viewctrl_layout)
    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._scene.scene.show_axes(self.settings.show_axes)
    def _on_show_hand(self, show):
        if self._active_hand is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_show_hand)")
            self._show_hands.checked = not show
            return
        self.settings.show_hand = show
        self._update_activate_hand()
    def _on_show_object(self, show):
        if self._objects is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_show_object)")
            self._show_objects.checked = not show
            return
        self.settings.show_objects = show
        self._init_obj_layer()
    def _on_show_pcd(self, show):
        if self._pcd is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_show_pcd)")
            self._show_objects.checked = not show
            return
        self.settings.show_pcd = show
        self._init_pcd_layer()
    def _on_show_coord_frame(self, show):
        self.settings.show_coord_frame = show
        if show:
            self._add_coord_frame("world_coord_frame")
        else:
            self._scene.scene.remove_geometry("world_coord_frame")
            for label in self.coord_labels:
                self._scene.remove_3d_label(label)
            self.coord_labels = []
    def _on_scene_point_size(self, size):
        self._log.text = "\t 포인트 사이즈 값을 변경합니다."
        self.window.set_needs_layout()
        self._last_change = time.time()
        mat = self.settings.scene_material
        mat.point_size = int(size)
        if self._check_geometry(self._scene_name):
            self._set_geometry_material(self._scene_name, mat)
        self._scene_point_size.double_value = size
    def _on_point_transparency(self, transparency):
        self._log.text = "\t 투명도 값을 변경합니다."
        self.window.set_needs_layout()
        self._last_change = time.time()
        self.settings.point_transparency = transparency
        self.settings.scene_material.base_color = [1.0, 1.0, 1.0, 1.0-transparency]
        self._point_transparency.double_value = transparency
        if self._check_geometry(self._scene_name):
            self._set_geometry_material(self._scene_name, self.settings.scene_material)
    def _on_object_transparency(self, transparency):
        self._log.text = "\t 투명도 값을 변경합니다."
        self.window.set_needs_layout()
        self._last_change = time.time()
        self.settings.obj_transparency = transparency
        self.settings.obj_material.base_color = [0, 0.0, 1.0, 1 - transparency]
        self._object_transparency.double_value = transparency
        if self._objects is not None:
            for obj_id, _ in self._objects.items():
                obj_name = "obj_{}".format(obj_id)
                self._set_geometry_material(obj_name, self.settings.obj_material)
    def _on_hand_transparency(self, transparency):
        self._log.text = "\t 투명도 값을 변경합니다."
        self.window.set_needs_layout()
        self._last_change = time.time()
        self.settings.hand_transparency = transparency
        self.settings.hand_mesh_material.base_color = [0.8, 0.8, 0.8, 1.0-transparency]
        self.settings.active_hand_mesh_material.base_color = [0.0, 1.0, 0.0, 1.0-transparency]
        self._hand_transparency.double_value = transparency
        if self._active_hand is not None:
            active_side = self._active_hand.side
            if active_side == 'right':
                self._set_geometry_material(self._right_hand_mesh_name, self.settings.active_hand_mesh_material)
                self._set_geometry_material(self._left_hand_mesh_name, self.settings.hand_mesh_material)
            else:
                self._set_geometry_material(self._right_hand_mesh_name, self.settings.hand_mesh_material)
                self._set_geometry_material(self._left_hand_mesh_name, self.settings.active_hand_mesh_material)
    def _on_hand_point_size(self, size):
        self._log.text = "\t 손 관절 사이즈 값을 변경합니다."
        self.window.set_needs_layout()
        self._last_change = time.time()
        mat = self.settings.hand_joint_material
        mat.point_size = int(size)
        if self._check_geometry(self._right_hand_joint_name):
            self._set_geometry_material(self._right_hand_joint_name, mat)
        if self._check_geometry(self._left_hand_joint_name):
            self._set_geometry_material(self._left_hand_joint_name, mat)
        self._hand_point_size.double_value = size
    def _on_hand_line_size(self, size):
        self._log.text = "\t 손 연결선 두께 값을 변경합니다."
        self.window.set_needs_layout()
        self._last_change = time.time()
        mat = self.settings.hand_mesh_material
        mat.line_width = int(size)
        if self._check_geometry(self._right_hand_mesh_name):
            self._set_geometry_material(self._right_hand_mesh_name, mat)
        if self._check_geometry(self._left_hand_mesh_name):
            self._set_geometry_material(self._left_hand_mesh_name, mat)

        if self._active_hand is not None:
            mat = self.settings.active_hand_mesh_material
            mat.line_width = int(size)
            active_side = self._active_hand.side
            if active_side == 'right':
                self._set_geometry_material(self._right_hand_mesh_name, mat)
            else:
                self._set_geometry_material(self._left_hand_mesh_name, mat)
        # link        
        mat = self.settings.hand_link_material
        mat.line_width = int(size)
        if self._check_geometry(self._right_hand_link_name):
            self._set_geometry_material(self._right_hand_link_name, mat)
        if self._check_geometry(self._left_hand_link_name):
            self._set_geometry_material(self._left_hand_link_name, mat)

        self._hand_line_size.double_value = size
    def _on_responsiveness(self, responsiveness):
        self._log.text = "\t 라벨링 민감도 값을 변경합니다."
        self.window.set_needs_layout()
        self._last_change = time.time()
        
        self.dist = 0.0004 * responsiveness
        self.deg = 0.2 * responsiveness
        self._responsiveness.double_value = responsiveness
    def _on_optimize_rate(self, optimize_rate):
        if self._active_hand is None: # shsh
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_optimize_rate)")
            return
        self._log.text = "\t 자동 정렬 민감도 값을 변경합니다."
        self._active_hand.set_learning_rate(optimize_rate*1e-3)
        self.window.set_needs_layout()
        self._last_change = time.time()
        self._optimize_rate.double_value = optimize_rate
    def _on_auto_save_interval(self, interval):
        self._log.text = "\t 자동 저장 간격을 변경합니다."
        self.window.set_needs_layout()
        self._auto_save_interval.double_value = interval
    
    
    # labeling stage edit
    def _init_stageedit_layout(self):
        em = self.window.theme.font_size
        stageedit_layout = gui.CollapsableVert("라벨링 단계 선택 (F1, F2)", 0.33*em,
                                                  gui.Margins(0.25*em, 0, 0, 0))
        stageedit_layout.set_is_open(True)
        self._current_stage_str = gui.Label("현재 상태: 준비중")
        stageedit_layout.add_child(self._current_stage_str)
        
        button = gui.Button(LabelingMode.STATIC)
        button.set_on_clicked(self._on_static_mode)
        stageedit_layout.add_child(button)
        
        button = gui.Button(LabelingMode.OPTIMIZE)
        button.set_on_clicked(self._on_optimize_mode)
        stageedit_layout.add_child(button)

        self._settings_panel.add_child(stageedit_layout)
    def _on_static_mode(self):
        self._convert_mode(LabelingMode.STATIC)
    def _on_optimize_mode(self):
        self._convert_mode(LabelingMode.OPTIMIZE)
    def _convert_mode(self, labeling_mode):
        if self._hands is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _convert_mode)")
            return 
        self._labeling_mode = labeling_mode
        self._current_stage_str.text = "현재 상태: {}".format(self._labeling_mode)
        self._active_hand.set_optimize_state(labeling_mode)
        self._update_target_hand()
        self._update_joint_mask()
        
    # labeling hand edit
    def _init_handedit_layout(self):
        em = self.window.theme.font_size
        self._is_right_hand = True # default is right
        self._hand_visible = True
        
        handedit_layout = gui.CollapsableVert("라벨 대상(손) 조절", 0.33*em,
                                                  gui.Margins(0.25*em, 0, 0, 0))
        handedit_layout.set_is_open(True)
        self._current_hand_str = gui.Label("현재 대상: 준비중")
        handedit_layout.add_child(self._current_hand_str)
        
        grid = gui.VGrid(2, 0.25 * em)
        
        button = gui.Button("현재 대상 리셋 (R)")
        button.set_on_clicked(self._reset_current_hand)
        grid.add_child(button)
        
        button = gui.Button("손 바꾸기 (Tab)")
        button.set_on_clicked(self._convert_hand)
        grid.add_child(button)
        
        button = gui.Button("이전 관절")
        button.horizontal_padding_em = 0.3
        button.vertical_padding_em = 0.3
        button.set_on_clicked(self._control_joint_down)
        grid.add_child(button)
        button = gui.Button("다음 관절")
        button.horizontal_padding_em = 0.3
        button.vertical_padding_em = 0.3
        button.set_on_clicked(self._control_joint_up)
        grid.add_child(button)
        
        handedit_layout.add_child(grid)
        
        grid = gui.VGrid(3, 0.25 * em)
        
        button = gui.Button("손목 (`)")
        button.horizontal_padding_em = 0.95
        button.vertical_padding_em = 0.2
        button.set_on_clicked(self._convert_to_root)
        grid.add_child(button)  
        button = gui.Button("엄지 (1)")
        button.horizontal_padding_em = 0.75
        button.vertical_padding_em = 0.2
        button.set_on_clicked(self._convert_to_thumb)
        grid.add_child(button)
        button = gui.Button("검지 (2)")
        button.horizontal_padding_em = 0.75
        button.vertical_padding_em = 0.2
        button.set_on_clicked(self._convert_to_fore)
        grid.add_child(button)
        
        button = gui.Button("중지 (3)")
        button.horizontal_padding_em = 0.75
        button.vertical_padding_em = 0.2
        button.set_on_clicked(self._convert_to_middle)
        grid.add_child(button)
        button = gui.Button("약지 (4)")
        button.horizontal_padding_em = 0.75
        button.vertical_padding_em = 0.2
        button.set_on_clicked(self._convert_to_ring)
        grid.add_child(button)
        button = gui.Button("소지 (5)")
        button.horizontal_padding_em = 0.75
        button.vertical_padding_em = 0.2
        button.set_on_clicked(self._convert_to_little)
        grid.add_child(button)
        
        handedit_layout.add_child(grid)

        

        
        self._settings_panel.add_child(handedit_layout)
    def _reset_current_hand(self):
        if self._active_hand is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _reset_current_hand)")
            return 
        self._active_hand.reset_pose()
        self._update_activate_hand()
        self._update_target_hand()
    def _convert_hand(self):
        if self._hands is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _convert_hand)")
            return 

        self._is_right_hand = not self._is_right_hand
        if self._is_right_hand:
            active_side = 'right'
        else:
            active_side = 'left'
        if self._check_geometry(self._right_hand_mesh_name):
            if active_side == 'right':
                self._set_geometry_material(self._right_hand_mesh_name, self.settings.active_hand_mesh_material)
            else:
                self._set_geometry_material(self._right_hand_mesh_name, self.settings.hand_mesh_material)
        if self._check_geometry(self._left_hand_mesh_name):
            if active_side == 'left':
                self._set_geometry_material(self._left_hand_mesh_name, self.settings.active_hand_mesh_material)
            else:
                self._set_geometry_material(self._left_hand_mesh_name, self.settings.hand_mesh_material)
        
        if self._check_geometry(self._target_joint_name):
            self._remove_geometry(self._target_joint_name)
        if self._check_geometry(self._target_link_name):
            self._remove_geometry(self._target_link_name)
        if self._check_geometry(self._active_joint_name):
            self._remove_geometry(self._active_joint_name)
            
        self._active_hand = self._hands[active_side]
        self._activate_template = self._template[active_side]
        self.preset_list.set_items(self._activate_template.get_template_list())
        self._convert_mode(LabelingMode.STATIC)
        self._update_current_hand_str()
        self._update_valid_error()
        self._update_joint_mask()

    # convert finger
    def _convert_to_root(self):
        self._convert_finger('root')
    def _convert_to_thumb(self):
        self._convert_finger('thumb')
    def _convert_to_fore(self):
        self._convert_finger('fore')
    def _convert_to_middle(self):
        self._convert_finger('middle')
    def _convert_to_ring(self):
        self._convert_finger('ring')
    def _convert_to_little(self):
        self._convert_finger('little')
    def _convert_finger(self, name):
        if self._active_hand is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _convert_finger)")
            return
        self._active_hand.set_optimize_target(name)
        self._update_target_hand()
        self._update_current_hand_str()
    def _control_joint_up(self):
        if self._active_hand is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _control_joint_up)")
            return
        ctrl_idx = self._active_hand.control_idx + 1
        self._active_hand.set_control_joint(ctrl_idx)
        self._update_joint_mask()
        self._update_target_hand()
        self._update_current_hand_str()
    def _control_joint_down(self):
        if self._active_hand is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _control_joint_down)")
            return
        ctrl_idx = self._active_hand.control_idx - 1
        self._active_hand.set_control_joint(ctrl_idx)
        self._update_joint_mask()
        self._update_target_hand()
        self._update_current_hand_str()
    def _update_current_hand_str(self):
        self._current_hand_str.text = "현재 대상: {}".format(self._active_hand.get_control_joint_name())

    # scene control
    def _init_scene_control_layout(self):
        em = self.window.theme.font_size
        scene_control_layout = gui.CollapsableVert("작업 파일 리스트", 0.33 * em,
                                                   gui.Margins(0.25 * em, 0, 0, 0))
        scene_control_layout.set_is_open(True)
        
        self._current_progress_str = gui.Label("작업 폴더: 준비중 | 현재 파일: 준비중")
        scene_control_layout.add_child(self._current_progress_str)
        
        h = gui.Horiz(0.4 * em)
        self._current_scene_pg = gui.Label("작업 폴더: [00/00]")
        h.add_child(self._current_scene_pg)
        button = gui.Button("이전")
        button.horizontal_padding_em = 0.8
        button.vertical_padding_em = 0
        button.set_on_clicked(self._on_previous_scene)
        h.add_child(button)
        button = gui.Button("다음")
        button.horizontal_padding_em = 0.8
        button.vertical_padding_em = 0
        button.set_on_clicked(self._on_next_scene)
        h.add_child(button)
        h.add_stretch()
        scene_control_layout.add_child(h)
        
        h = gui.Horiz(0.4 * em)
        self._current_file_pg = gui.Label("현재 파일: [00/00]")
        h.add_child(self._current_file_pg)
        button = gui.Button("이전")
        button.horizontal_padding_em = 0.8
        button.vertical_padding_em = 0
        button.set_on_clicked(self._on_previous_frame)
        h.add_child(button)
        button = gui.Button("다음")
        button.horizontal_padding_em = 0.8
        button.vertical_padding_em = 0
        button.set_on_clicked(self._on_next_frame)
        h.add_child(button)
        h.add_stretch()
        scene_control_layout.add_child(h)
        self._settings_panel.add_child(scene_control_layout)
    def _check_changes(self):
        if self._annotation_changed:
            self._on_error("라벨링 결과를 저장하지 않았습니다. 저장하지 않고 넘어가려면 버튼을 다시 눌러주세요.")
            self._annotation_changed = False
            return True
        else:
            return False
    def _load_scene(self):
        self._frame = self.annotation_scene.get_current_frame()
        try:
            ret = self.annotation_scene.load_label()
        except:
            self._log.text = "\t 저장된 라벨이 없습니다."
            pass
        if not ret:
            hands = self._frame.hands
            pcd = self._frame.scene_pcd
            for s, hand_model in hands.items():
                hand_model.set_root_position(pcd.get_center())
        self._update_progress_str()
        self._on_change_camera_merge()
        # self._init_pcd_layer()
        self._init_hand_layer()
        self._init_obj_layer()
        self._update_valid_error()
        
    def _update_progress_str(self):
        self._current_progress_str.text = self.dataset.get_current_file()
        self._current_file_pg.text = self.annotation_scene.get_progress()
        self._current_scene_pg.text = self.dataset.get_progress()
    def _on_previous_frame(self):
        if self._check_changes():
            return
        if not self._check_annotation_scene():
            return
        if not self.annotation_scene.moveto_previous_frame():
            self._on_error("이전 포인트 클라우드가 존재하지 않습니다.")
            return
        self._log.text = "\t 이전 포인트 클라우드로 이동했습니다."
        self._load_scene()
    def _on_next_frame(self):
        if self._check_changes():
            return
        if not self._check_annotation_scene():
            return
        if not self.annotation_scene.moveto_next_frame():
            self._on_error("다음 포인트 클라우드가 존재하지 않습니다.")
            return
        self._log.text = "\t 다음 포인트 클라우드로 이동했습니다."
        self._load_scene()
    def _on_previous_scene(self):
        if self._check_changes():
            return
        if not self._check_annotation_scene():
            return
        scene = self.dataset.get_previous_scene()
        if scene is None:
            self._on_error("이전 작업 폴더가 존재하지 않습니다.")
            return
        self._log.text = "\t 이전 작업 폴더로 이동했습니다."
        self.annotation_scene = scene
        self._load_scene()
    def _on_next_scene(self):
        if self._check_changes():
            return
        if not self._check_annotation_scene():
            return
        scene = self.dataset.get_next_scene()
        if scene is None:
            self._on_error("다음 작업 폴더가 존재하지 않습니다.")
            return
        self._log.text = "\t 다음 작업 폴더로 이동했습니다."
        self.annotation_scene = scene
        self._load_scene()

    # label control
    def _init_label_control_layout(self):
        em = self.window.theme.font_size
        label_control_layout = gui.CollapsableVert("라벨 저장 및 불러오기", 0.33 * em,
                                                   gui.Margins(0.25 * em, 0, 0, 0))
        label_control_layout.set_is_open(True)
        
        button = gui.Button("라벨링 결과 저장하기 (F)")
        button.set_on_clicked(self._on_save_label)
        label_control_layout.add_child(button)
        
        button = gui.Button("이전 이미지 라벨 불러오기")
        button.set_on_clicked(self._on_load_previous_label)
        label_control_layout.add_child(button)
        self._settings_panel.add_child(label_control_layout)

    def _on_save_label(self):
        self._log.text = "\t라벨링 결과를 저장 중입니다."
        self.window.set_needs_layout()
        
        if not self._check_annotation_scene():
            return
        
        self.annotation_scene.save_json()
        self.annotation_scene.save_label()
        self._update_valid_error()
        self._update_diff_viewer()
        self._last_saved = time.time()
        self._log.text = "\t라벨링 결과를 저장했습니다."
        self._annotation_changed = False
    def _on_load_previous_label(self):
        if not self._check_annotation_scene():
            return
        self._log.text = "\t라벨링 결과를 불러오는 중입니다."
        self.window.set_needs_layout()
        
        ret = self.annotation_scene.load_previous_label()
        if ret:
            self._log.text = "\t이전 라벨링 결과를 불러왔습니다."
            self._init_hand_layer()
            self._annotation_changed = False
        else:
            self._on_error("저장된 라벨이 없습니다. (error at _on_load_previous_label)")
            return
    
    def _init_preset_layout(self):
        em = self.window.theme.font_size
        preset_layout = gui.CollapsableVert("프리셋 저장 및 불러오기", 0.33 * em,
                                                gui.Margins(0.25 * em, 0, 0, 0))
        
        label = gui.Label("{0:-^50}".format("프리셋"))
        preset_layout.add_child(label)
        self.preset_list = gui.ListView()
        preset_layout.add_child(self.preset_list)
        self.preset_list.set_on_selection_changed(self._on_change_preset_select)
        h = gui.Horiz(0.4 * em)
        self.preset_name = gui.TextEdit()
        self.preset_name.text_value = "프리셋 이름"
        h.add_child(self.preset_name)
        button = gui.Button("불러오기")
        button.set_on_clicked(self._on_load_preset)
        h.add_child(button)
        button = gui.Button("저장하기")
        button.set_on_clicked(self._on_save_preset)
        h.add_child(button)
        preset_layout.add_child(h)

        # label = gui.Label("{0:-^45}".format("왼손 프리셋"))
        # preset_layout.add_child(label)
        # self.l_preset_list = gui.ListView()
        # preset_layout.add_child(self.l_preset_list)
        # self.l_preset_list.set_on_selection_changed(self._on_change_preset_select_l)
        # self.l_preset_list.set_items(self.left_template.get_template_list())
        # h = gui.Horiz(0.4 * em)
        # self._l_preset_name = gui.TextEdit()
        # self._l_preset_name.text_value = "프리셋 이름"
        # h.add_child(self._l_preset_name)
        # button = gui.Button("불러오기")
        # button.set_on_clicked(self._on_load_preset_l)
        # h.add_child(button)
        # button = gui.Button("저장하기")
        # button.set_on_clicked(self._on_save_preset_l)
        # h.add_child(button)
        # preset_layout.add_child(h)
        
        self._joint_mask_proxy = gui.WidgetProxy()
        self._joint_mask_proxy.set_widget(gui.ImageWidget())
        self._validation_panel.add_child(self._joint_mask_proxy)
        self._validation_panel.add_child(preset_layout)
    
    def _on_load_preset(self):
        if not self._check_annotation_scene():
            return
        name = self.preset_name.text_value
        try:
            pose = self._activate_template.get_template2pose(name)
            self._active_hand.set_joint_pose(pose)
            self._update_activate_hand()
            self._update_target_hand()
        except:
            self._on_error("프리셋 이름을 확인하세요. (error at _on_load_preset)")
    def _on_save_preset(self):
        if not self._check_annotation_scene():
            return
        name = self.preset_name.text_value
        pose = self._active_hand.get_joint_pose()
        self._activate_template.save_pose2template(name, pose)
        self.preset_list.set_items(self._activate_template.get_template_list())
    def _on_change_preset_select(self, preset_name, double):
        self.preset_name.text_value = preset_name
        if double:
            self._on_load_preset()
    def _update_joint_mask(self):
        img = self._active_hand.get_joint_mask()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = o3d.geometry.Image(img)
        self._joint_mask_proxy.set_widget(gui.ImageWidget(img))

    # image viewer
    def _init_image_view_layout(self):
        self._rgb_proxy = gui.WidgetProxy()
        self._rgb_proxy.set_widget(gui.ImageWidget())
        self._images_panel.add_child(self._rgb_proxy)
        self._images_panel.set_is_open(False)
        self._diff_proxy = gui.WidgetProxy()
        self._diff_proxy.set_widget(gui.ImageWidget())
        self._images_panel.add_child(self._diff_proxy)
        self._images_panel.set_is_open(False)
    
    # validate
    def _init_show_error_layout(self):
        em = self.window.theme.font_size
        show_error_layout = gui.CollapsableVert("카메라 시점 조정", 0.33 * em,
                                                   gui.Margins(0.25 * em, 0, 0, 0))
        show_error_layout.set_is_open(True)
        
        self._view_error_layout_list = []
        
        for i in range(8):
            h = gui.Horiz(0)
            
            button = gui.Button("카메라 {}".format(i+1))
            button.set_on_clicked(self.__getattribute__("_on_change_camera_{}".format(i)))
            button.vertical_padding_em = 0.1
            h.add_child(button)
            h.add_child(gui.Label(" | \n | "))
            
            v = gui.Vert(0)
            right_error_txt = gui.Label("준비 안됨")
            v.add_child(right_error_txt)
            left_error_txt = gui.Label("준비 안됨")
            v.add_child(left_error_txt)
            h.add_child(v)
            h.add_child(gui.Label(" | \n | "))
            
            box = gui.Checkbox("")
            box.set_on_checked(self._on_change_bbox)
            h.add_child(box)
            h.add_child(gui.Label(" | \n | "))

            _button = gui.Button("*")
            _button.set_on_clicked(self.__getattribute__("_on_click_focus_{}".format(i)))
            _button.vertical_padding_em = 0.1
            h.add_child(_button)
            
            show_error_layout.add_child(h)
            
            self._view_error_layout_list.append((button, right_error_txt, left_error_txt, box))

        show_error_layout.add_child(gui.Label("-"*60))

        h = gui.Horiz(0.4 * em)
        button = gui.Button("합친 상태")
        button.vertical_padding_em = 0.1
        button.set_on_clicked(self._on_change_camera_merge)
        h.add_child(button)
        h.add_child(gui.Label(" | \n | "))
        
        v = gui.Vert(0)
        right_error_txt = gui.Label("준비 안됨")
        v.add_child(right_error_txt)
        left_error_txt = gui.Label("준비 안됨")
        v.add_child(left_error_txt)
        h.add_child(v)
        
        button = gui.Button("손가락 정렬")
        button.vertical_padding_em = 0.1
        button.set_on_clicked(self._on_icp)
        h.add_child(button)
        
        self._total_error_txt = (right_error_txt, left_error_txt)
        show_error_layout.add_child(h)
        self._activate_cam_txt = gui.Label("현재 활성화된 카메라: 없음")
        show_error_layout.add_child(self._activate_cam_txt)

        self._validation_panel.add_child(show_error_layout)
    def _on_change_camera_0(self):
        if not self._check_annotation_scene():
            return
        self._camera_idx = 0
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_error_layout_list[self._camera_idx][0].text)
    def _on_change_camera_1(self):
        if not self._check_annotation_scene():
            return
        self._camera_idx = 1
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_error_layout_list[self._camera_idx][0].text)
    def _on_change_camera_2(self):
        if not self._check_annotation_scene():
            return
        self._camera_idx = 2
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_error_layout_list[self._camera_idx][0].text)
    def _on_change_camera_3(self):
        if not self._check_annotation_scene():
            return
        self._camera_idx = 3
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_error_layout_list[self._camera_idx][0].text)
    def _on_change_camera_4(self):
        if not self._check_annotation_scene():
            return
        self._camera_idx = 4
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_error_layout_list[self._camera_idx][0].text)
    def _on_change_camera_5(self):
        if not self._check_annotation_scene():
            return
        self._camera_idx = 5
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_error_layout_list[self._camera_idx][0].text)
    def _on_change_camera_6(self):
        if not self._check_annotation_scene():
            return
        self._camera_idx = 6
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_error_layout_list[self._camera_idx][0].text)
    def _on_change_camera_7(self):
        if not self._check_annotation_scene():
            return
        self._camera_idx = 7
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_error_layout_list[self._camera_idx][0].text)
    def _on_change_camera_merge(self):
        if not self._check_annotation_scene():
            return
        self._camera_idx = -1
        for but, _, _, bbox in self._view_error_layout_list:
            bbox.checked = True
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: 합쳐진 뷰"
    def _on_change_camera(self):
        self._reset_image_viewer()
        self._update_image_viewer()
        self._update_diff_viewer()
        self._init_pcd_layer()
        self._on_active_camera_viewpoint()
    def _on_change_bbox(self, visible):
        if not self._check_annotation_scene():
            return
        self._init_pcd_layer()
    def _get_activate_cam(self):
        cam_list = []
        for but, _, _, bbox in self._view_error_layout_list:
            if bbox.checked:
                cam_list.append(but.text)
        return cam_list
    def _on_click_focus_0(self):
        self._on_focus(0)
    def _on_click_focus_1(self):
        self._on_focus(1)
    def _on_click_focus_2(self):
        self._on_focus(2)
    def _on_click_focus_3(self):
        self._on_focus(3)
    def _on_click_focus_4(self):
        self._on_focus(4)
    def _on_click_focus_5(self):
        self._on_focus(5)
    def _on_click_focus_6(self):
        self._on_focus(6)
    def _on_click_focus_7(self):
        self._on_focus(7)
    def _on_focus(self, idx):
        for i, (_, _, _, bbox) in enumerate(self._view_error_layout_list):
            if i == idx:
                bbox.checked = True
            else:
                bbox.checked = False
        self._init_pcd_layer()
    

    def _init_cam_name(self):
        self._cam_name_list = list(self.annotation_scene._cameras.keys())
        self._cam_name_list.sort()
        for idx, (cam_button, _, _, _) in enumerate(self._view_error_layout_list):
            cam_button.text = self._cam_name_list[idx]
        self._diff_images = {cam_name: None for cam_name in self._cam_name_list}
    def _update_image_viewer(self):
        if self._camera_idx == -1:
            self._rgb_proxy.set_widget(gui.ImageWidget())
            return
        current_cam = self._cam_name_list[self._camera_idx]
        rgb_img = self._frame.get_rgb(current_cam)
        self.rgb_img = rgb_img
        self.H, self.W, _ = rgb_img.shape
        self._rgb_proxy.set_widget(gui.ImageWidget(self._img_wrapper(self.rgb_img)))
    def _update_diff_viewer(self):
        if self._camera_idx == -1:
            self._diff_proxy.set_widget(gui.ImageWidget())
            return
        current_cam = self._cam_name_list[self._camera_idx]
        diff_img = self._diff_images[current_cam]
        self.diff_img = diff_img
        if diff_img is not None:
            self._diff_proxy.set_widget(gui.ImageWidget(self._img_wrapper(diff_img)))
        else:
            self._diff_proxy.set_widget(gui.ImageWidget())
    def _update_valid_error(self):
        return
        self._log.text = "\t라벨링 검증용 이미지를 생성 중입니다."
        self.window.set_needs_layout()   

        self.hl_renderer.reset()
        
        self.hl_renderer.add_objects(self._objects, color=[1, 0, 0])
        self.hl_renderer.add_hands(self._hands) # right [0, 1, 0] left [0, 0, 1]

        # rendering depth for each camera
        depth_diff_list = []
        self._diff_images = {}
        for error_layout in self._view_error_layout_list:
            cam_name = error_layout[0].text
            intrinsic = self._frame.cameras[cam_name].intrinsic
            extrinsic = self._frame.cameras[cam_name].extrinsics

            self.hl_renderer.set_camera(intrinsic, extrinsic, self.W, self.H)
            # rendering depth
            depth_rendered = self.hl_renderer.render_depth()
            depth_rendered = np.array(depth_rendered, dtype=np.float32)
            depth_rendered[np.isposinf(depth_rendered)] = 0
            depth_rendered *= 1000 # convert meter to mm

            rgb_rendered = self.hl_renderer.render_rgb()
            rgb_rendered = np.array(rgb_rendered)

            # only hand mask
            right_hand_mask = np.bitwise_and(rgb_rendered[:, :, 0]==0, np.bitwise_and(rgb_rendered[:, :, 1]!=0, rgb_rendered[:, :, 2]==0))
            left_hand_mask = np.bitwise_and(rgb_rendered[:, :, 0]==0, np.bitwise_and(rgb_rendered[:, :, 1]==0, rgb_rendered[:, :, 2]!=0))

            # set mask as rendered depth
            valid_mask = depth_rendered > 0

            # get captured image
            rgb_captured = self._frame.get_rgb(cam_name)
            depth_captured = self._frame.get_depth(cam_name)

            # diff_vis = np.zeros_like(rgb_captured)
            diff_vis = rgb_rendered
            # calculate diff
            depth_diff = depth_captured - depth_rendered
            depth_diff_abs = np.abs(np.copy(depth_diff))
            inlier_mask = depth_diff_abs < 50

            # right_hand
            r_valid_mask = valid_mask * right_hand_mask * inlier_mask
            if np.sum(r_valid_mask) > 0:
                depth_diff_mean = np.sum(depth_diff_abs[r_valid_mask]) / np.sum(r_valid_mask)
            else:
                depth_diff_mean = -1
            r_diff_mean = copy.deepcopy(depth_diff_mean)
            # left hand
            l_valid_mask = valid_mask * left_hand_mask * inlier_mask
            if np.sum(l_valid_mask) > 0:
                depth_diff_mean = np.sum(depth_diff_abs[l_valid_mask]) / np.sum(l_valid_mask)
            else:
                depth_diff_mean = -1
            l_diff_mean = copy.deepcopy(depth_diff_mean)
            
            depth_diff_list.append([r_diff_mean, l_diff_mean])
            error_layout[1].text = "오른손: {:.2f}".format(r_diff_mean)
            error_layout[2].text = "왼손: {:.2f}".format(l_diff_mean)
            
            # diff_vis[depth_rendered > 0] = [255, 0, 0]
            diff_vis = cv2.addWeighted(rgb_captured, 0.8, diff_vis, 1.0, 0)
            self._diff_images[cam_name] = diff_vis
            
        total_mean = [0, 0]
        count = [0, 0]
        max_v = [-np.inf, -np.inf]
        max_idx = [None, None]
        for idx, diff in enumerate(depth_diff_list):
            for s_idx, dif in enumerate(diff):
                if dif==-1:
                    continue
                if dif > max_v[s_idx]:
                    max_v[s_idx] = dif
                    max_idx[s_idx] = idx
                total_mean[s_idx] += dif
                count[s_idx] += 1
        if self._active_hand.side=='right':
            max_idx = max_idx[0]
        else:
            max_idx = max_idx[1]
        for idx, error_layout in enumerate(self._view_error_layout_list):
            if idx==max_idx:
                error_layout[1].text_color = gui.Color(1, 0, 0)
                error_layout[2].text_color = gui.Color(1, 0, 0)
            else:
                error_layout[1].text_color = gui.Color(1, 1, 1)
                error_layout[2].text_color = gui.Color(1, 1, 1)
        try:
            total_mean[0] /= count[0]
        except:
            total_mean[0] = -1
        try:
            total_mean[1] /= count[1]
        except:
            total_mean[1] = -1
        self._total_error_txt[0].text = "오른손: {:.2f}".format(total_mean[0])
        self._total_error_txt[1].text = "왼손: {:.2f}".format(total_mean[1])
        
        # clear geometry
        self._log.text = "\t라벨링 검증용 이미지를 생성했습니다."
        self.window.set_needs_layout()
    def _img_wrapper(self, img):
        ratio = 640 / self.W
        img = cv2.resize(img.copy(), (640, int(self.H*ratio)))
        return o3d.geometry.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    def _reset_image_viewer(self):
        self.icx, self.icy = self.W / 2, self.H / 2
        self.scale_factor = 1
        
    #endregion
    
    #region ----- Open3DScene 
    #----- geometry
    def _check_geometry(self, name):
        return self._scene.scene.has_geometry(name)
    
    def _remove_geometry(self, name):
        self._scene.scene.remove_geometry(name)
    
    def _add_geometry(self, name, geo, mat):
        if self._check_geometry(name):
            self._remove_geometry(name)
        self._scene.scene.add_geometry(name, geo, mat,
                                       add_downsampled_copy_for_fast_rendering=False)
    
    def _clear_geometry(self):
        self._scene.scene.clear_geometry()

    def _set_geometry_transform(self, name, transform):
        self._scene.scene.set_geometry_transform(name, transform)
    
    def _get_geometry_transform(self, name):
        return self._scene.scene.get_geometry_transform(name)
 
    #----- 
    def _set_background_color(self, color):
        self._scene.scene.set_background(color)
    
    def _set_geometry_visible(self, name, is_visible):
        self._scene.scene.show_geometry(name, is_visible)
    
    def _check_geometry_visible(self, name):
        return self._scene.scene.geometry_is_visible(name)
    
    def _set_axes_visible(self, is_visible):
        self._scene.scene.show_axes(is_visible)
    
    def _set_ground_plane_visible(self, is_visible):
        self._scene.scene.show_ground_plane(is_visible)
    
    def _set_skybox_visible(self, is_visible):
        self._scene.scene.show_skybox(is_visible)
        
    def _set_geometry_material(self, name, mat):
        self._scene.scene.modify_geometry_material(name, mat)
    
    #endregion
    def _add_hand_frame(self, size=0.05, origin=[0, 0, 0]):
        if self._active_hand is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _add_hand_frame)")
        self._remove_hand_frame()
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
        transform = np.eye(4)
        current_xyz = self._active_hand.get_control_rotation()
        transform[:3, :3] = Rot.from_rotvec(current_xyz).as_matrix()
        transform[:3, 3] = self._active_hand.get_control_position()
        coord_frame.transform(transform)
        
        self.coord_labels = []
        size = size * 0.8
        self.coord_labels.append(self._scene.add_3d_label(np.matmul(transform, np.array([size, 0, 0, 1]))[:3], "W, S"))
        self.coord_labels.append(self._scene.add_3d_label(np.matmul(transform, np.array([0, size, 0, 1]))[:3], "A, D"))
        self.coord_labels.append(self._scene.add_3d_label(np.matmul(transform, np.array([0, 0, size, 1]))[:3], "Q, E"))
        self._add_geometry("hand_frame", coord_frame, self.settings.coord_material)
    def _remove_hand_frame(self):
        self._remove_geometry("hand_frame")
        for label in self.coord_labels:
            self._scene.remove_3d_label(label)
        self.coord_labels = []
    def _on_mouse(self, event):
        # We could override BUTTON_DOWN without a modifier, but that would
        # interfere with manipulating the scene.
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.ALT) and event.is_button_down(gui.MouseButton.LEFT):
            if self._active_hand is None:
                self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_mouse)")
                return gui.Widget.EventCallbackResult.IGNORED
            def depth_callback(depth_image):
                # Coordinates are expressed in absolute coordinates of the
                # window, but to dereference the image correctly we need them
                # relative to the origin of the widget. Note that even if the
                # scene widget is the only thing in the window, if a menubar
                # exists it also takes up space in the window (except on macOS).
                x = event.x - self._scene.frame.x
                y = event.y - self._scene.frame.y
                # Note that np.asarray() reverses the axes.
                depth_area = np.asarray(depth_image)[y-10:y+10, x-10:x+10]
                if depth_area.min == 1.0: # clicked on nothing (i.e. the far plane)
                    pass
                
                else:
                    depth = np.mean(depth_area[depth_area!=1.0])
                
                    world_xyz = self._scene.scene.camera.unproject(
                        event.x, event.y, depth, self._scene.frame.width,
                        self._scene.frame.height)
                    def move_joint():
                        self._move_control_joint(world_xyz)
                
                    gui.Application.instance.post_to_main_thread(
                        self.window, move_joint)
                
            self._scene.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.CONSUMED
        
        return gui.Widget.EventCallbackResult.IGNORED
    def move(self, x, y, z, rx, ry, rz):
        self._log.text = "{} 라벨 이동 중입니다.".format(self._active_hand.get_control_joint_name())
        self.window.set_needs_layout()
        self._last_change = time.time()
        if x != 0 or y != 0 or z != 0:
            current_xyz = self._active_hand.get_control_position()
            # convert x, y, z cam to world
            R = self._scene.scene.camera.get_view_matrix()[:3,:3]
            R_inv = np.linalg.inv(R)
            xyz = np.dot(R_inv, np.array([x, y, z]))
            xyz = current_xyz + xyz
            self._move_control_joint(xyz)
        else:
            current_xyz = self._active_hand.get_control_rotation()
            r = Rot.from_rotvec(current_xyz)
            current_rot_mat = r.as_matrix()
            rot_mat = o3d.geometry.get_rotation_matrix_from_xyz((rx, ry, rz))
            r = Rot.from_matrix(np.matmul(current_rot_mat, rot_mat))
            xyz = r.as_rotvec()
            self._active_hand.set_control_rotation(xyz)
            self._active_hand.save_undo()
            self._annotation_changed = True
        self._update_activate_hand()
        self._update_target_hand()
    def _move_control_joint(self, xyz):
        self._active_hand.set_control_position(xyz)
        if self._active_hand.get_optimize_target()=='root':
            self._active_hand.save_undo(forced=True)
            self._annotation_changed = True
        self._update_activate_hand()
        self._update_target_hand()
    
    def _init_pcd_layer(self):
        if self.settings.show_pcd:
            # cam_name_list = []
            # for cam_name in self._get_activate_cam():
            #     cam_name_list.append(cam_name)
            # if cam_name_list == []:
            #     self._remove_geometry(self._scene_name)
            #     return    
            # self._pcd = self._frame.get_pcd(cam_name_list)
            self.bounds = self._pcd.get_axis_aligned_bounding_box()
            self._add_geometry(self._scene_name, self._pcd, self.settings.scene_material)
        else:
            self._remove_geometry(self._scene_name)
    def _toggle_pcd_visible(self):
        show = self._show_pcd.checked
        self._show_pcd.checked = not show
        self._on_show_pcd(not show)
    
    def _init_obj_layer(self):
        # reset
        for obj_name in self._object_names:
            self._remove_geometry(obj_name)
        if self.settings.show_objects:
            self._objects = self._frame.objects
            self._object_names = []
            for obj_id, obj in self._objects.items():
                obj_name = "obj_{}".format(obj_id)
                geo = obj.get_geometry()
                self._add_geometry(obj_name, geo, self.settings.obj_material)
                self._object_names.append(obj_name)
    def _toggle_obj_visible(self):
        show = self._show_objects.checked
        self._show_objects.checked = not show
        self._on_show_object(not show)
    
    def _init_hand_layer(self):
        # visualize hand
        hands = self._hands
        if self._is_right_hand > 0:
            active_side = 'right'
        else:
            active_side = 'left'
        self._hands = hands
        self._active_hand = hands[active_side]
        self._activate_template = self._template[active_side]
        self.preset_list.set_items(self._activate_template.get_template_list())
        
        for side, hand in hands.items():
            hand_geo = hand.get_geometry()
            if side == 'right':
                mesh_name = self._right_hand_mesh_name
                joint_name = self._right_hand_joint_name
                link_name = self._right_hand_link_name
            elif side == 'left':
                mesh_name = self._left_hand_mesh_name
                joint_name = self._left_hand_joint_name
                link_name = self._left_hand_link_name
            else:
                assert False, "visualize hand error"
            if side == active_side:
                mesh_mat = self.settings.active_hand_mesh_material
            else:
                mesh_mat = self.settings.hand_mesh_material
            self._add_geometry(mesh_name, hand_geo['mesh'], mesh_mat)
            self._add_geometry(joint_name, hand_geo['joint'], self.settings.hand_joint_material)
            self._add_geometry(link_name, hand_geo['link'], self.settings.hand_link_material)
    
        self._convert_mode(LabelingMode.STATIC)
        active_geo = self._active_hand.get_active_geometry()
        self._add_geometry(self._control_joint_name, 
                           active_geo['control'], self.settings.control_target_joint_material)
        self.control_joint_geo = active_geo['control']
        self._update_joint_mask()
        self._update_current_hand_str()
    def _update_target_hand(self):
        if self._labeling_mode==LabelingMode.OPTIMIZE:
            target_geo = self._active_hand.get_target_geometry()
            self._add_geometry(self._target_joint_name, 
                            target_geo['joint'], self.settings.target_joint_material)
            self._add_geometry(self._target_link_name, 
                            target_geo['link'], self.settings.target_link_material)
            # self._add_geometry("test_points", 
            #                 target_geo['optim_points'], self.settings.target_joint_material)


            active_geo = self._active_hand.get_active_geometry()
            self._add_geometry(self._active_joint_name, 
                            active_geo['joint'], self.settings.active_target_joint_material)
            self._add_geometry(self._control_joint_name, 
                            active_geo['control'], self.settings.control_target_joint_material)
            self.control_joint_geo = active_geo['control']
        else:
            if self._check_geometry(self._target_joint_name):
                self._remove_geometry(self._target_joint_name)
            if self._check_geometry(self._target_link_name):
                self._remove_geometry(self._target_link_name)
            if self._check_geometry(self._active_joint_name):
                self._remove_geometry(self._active_joint_name)
            active_geo = self._active_hand.get_active_geometry()
            self._add_geometry(self._control_joint_name, 
                            active_geo['control'], self.settings.control_target_joint_material)
            self.control_joint_geo = active_geo['control']
    def _update_activate_hand(self):
        if self.settings.show_hand:
            hand_geo = self._active_hand.get_geometry()
            side = self._active_hand.side
            if side == 'right':
                mesh_name = self._right_hand_mesh_name
                joint_name = self._right_hand_joint_name
                link_name = self._right_hand_link_name
            elif side == 'left':
                mesh_name = self._left_hand_mesh_name
                joint_name = self._left_hand_joint_name
                link_name = self._left_hand_link_name
            else:
                assert False, "visualize hand error"
            mesh_mat = self.settings.active_hand_mesh_material
            self._add_geometry(mesh_name, hand_geo['mesh'], mesh_mat)
            self._add_geometry(joint_name, hand_geo['joint'], self.settings.hand_joint_material)
            self._add_geometry(link_name, hand_geo['link'], self.settings.hand_link_material)
        else:
            side = self._active_hand.side
            if side == 'right':
                mesh_name = self._right_hand_mesh_name
                joint_name = self._right_hand_joint_name
                link_name = self._right_hand_link_name
            elif side == 'left':
                mesh_name = self._left_hand_mesh_name
                joint_name = self._left_hand_joint_name
                link_name = self._left_hand_link_name
            self._remove_geometry(mesh_name)
            self._remove_geometry(joint_name)
            self._remove_geometry(link_name)
    def _toggle_hand_visible(self):
        show = self._show_hands.checked
        self._show_hands.checked = not show
        self._on_show_hand(not show)
    def _on_optimize(self):
        self._log.text = "\t {} 자동 정렬 중입니다.".format(self._active_hand.get_control_joint_name())
        self.window.set_needs_layout()
        self._last_change = time.time()
        self._annotation_changed = self._active_hand.optimize_to_target()
        self._update_target_hand()
        self._update_activate_hand()
    def _on_icp(self):
        if not self._check_annotation_scene():
            return
        hand_mesh = self._active_hand.get_geometry()['mesh']
        bounds = hand_mesh.get_oriented_bounding_box()
        target_pcd = self._pcd.crop(bounds.scale(1.1, bounds.center))
        if len(target_pcd.points) < 1000:
            self._on_error("활성화된 손 근처에 포인트가 부족합니다. 손을 더 맞춰주세요")
            return
        # downsampling
        sample = min(10000, len(target_pcd.points))
        target_pcd = target_pcd.uniform_down_sample(sample)
        target_points = np.asarray(target_pcd.points)
        self._active_hand.optimize_to_points(target_points)
        self._update_target_hand()
        self._update_activate_hand()
        self._active_hand.save_undo(forced=True)
    def _on_shape_optim(self):
        if not self._check_annotation_scene():
            return
        hand_mesh = self._active_hand.get_geometry()['mesh']
        target_pcd = self._pcd
        if len(target_pcd.points) < 1000:
            self._on_error("활성화된 손 근처에 포인트가 부족합니다. 손을 더 맞춰주세요")
            return
        # downsampling
        target_pcd = target_pcd.voxel_down_sample(0.003)
        target_points = np.asarray(target_pcd.points)
        self._active_hand.optimize_shape(target_points)
        self._update_target_hand()
        self._update_activate_hand()
        # self._active_hand.save_undo(forced=True)

    def _undo(self):
        self._auto_optimize.checked = False
        ret = self._active_hand.undo()
        if not ret:
            self._on_error("이전 상태가 없습니다. (error at _undo)")
        else:
            self._log.text = "이전 상태를 불러옵니다."
            self.window.set_needs_layout()
            self._update_activate_hand()
            self._update_target_hand()
    def _redo(self):
        self._auto_optimize.checked = False
        ret = self._active_hand.redo()
        if not ret:
            self._on_error("이후 상태가 없습니다. (error at _redo)")
        else:
            self._log.text = "이후 상태를 불러옵니다."
            self.window.set_needs_layout()
            self._update_activate_hand()
            self._update_target_hand()
    
    def _on_key(self, event):
        if self._active_hand is None:
            return gui.Widget.EventCallbackResult.IGNORED

        # if shift pressed then 
        # 1. translation -> rotation
        if (event.key == gui.KeyName.LEFT_SHIFT or event.key == gui.KeyName.RIGHT_SHIFT) \
            and (self._labeling_mode==LabelingMode.STATIC or self._active_hand.get_optimize_target()=='root'):
            if event.type == gui.KeyEvent.DOWN:
                self._left_shift_modifier = True
                self._add_hand_frame()
            elif event.type == gui.KeyEvent.UP:
                self._left_shift_modifier = False
                self._remove_hand_frame()
            return gui.Widget.EventCallbackResult.HANDLED
        
        # if ctrl is pressed then 
        # increase translation
        # reset finger to flat
        if event.key == gui.KeyName.LEFT_CONTROL or event.key == gui.KeyName.RIGHT_CONTROL:
            if event.type == gui.KeyEvent.DOWN:
                if not self.upscale_responsiveness:
                    self.dist = self.dist * 15
                    self.deg = self.deg * 15
                    self.upscale_responsiveness = True
                if not self.reset_flat:
                    self.reset_flat = True
                if not self.joint_back:
                    self.joint_back = True
            elif event.type == gui.KeyEvent.UP:
                if self.upscale_responsiveness:
                    self.dist = self.dist / 15
                    self.deg = self.deg / 15
                    self.upscale_responsiveness = False
                if self.reset_flat:
                    self.reset_flat = False
                if self.joint_back:
                    self.joint_back = False
            return gui.Widget.EventCallbackResult.HANDLED


        if event.key == gui.KeyName.T and event.type == gui.KeyEvent.DOWN:
            self._on_initial_viewpoint()
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key == gui.KeyName.Y and event.type == gui.KeyEvent.DOWN:
            self._on_active_camera_viewpoint()
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key == gui.KeyName.G and event.type == gui.KeyEvent.DOWN:
            self._on_active_viewpoint()
            return gui.Widget.EventCallbackResult.HANDLED
        
        if event.key == gui.KeyName.B:
            if self._active_hand is None:
                return gui.Widget.EventCallbackResult.IGNORED
            self._on_icp()
         
        if (event.key==gui.KeyName.COMMA and event.type==gui.KeyEvent.DOWN):
            self._undo()
            return gui.Widget.EventCallbackResult.CONSUMED
        elif (event.key==gui.KeyName.PERIOD and event.type==gui.KeyEvent.DOWN):
            self._redo()
            return gui.Widget.EventCallbackResult.CONSUMED
        
        # activate autosave
        if event.key==gui.KeyName.Z and event.type==gui.KeyEvent.DOWN:
            self._toggle_hand_visible()
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.key==gui.KeyName.X and event.type==gui.KeyEvent.DOWN:
            self._auto_optimize.checked = not self._auto_optimize.checked
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.key==gui.KeyName.C and event.type==gui.KeyEvent.DOWN:
            self._toggle_obj_visible()
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.key==gui.KeyName.V and event.type==gui.KeyEvent.DOWN:
            self._toggle_pcd_visible()
            return gui.Widget.EventCallbackResult.CONSUMED
        
        if event.key in [gui.KeyName.I, gui.KeyName.J, gui.KeyName.K, gui.KeyName.L, gui.KeyName.U, gui.KeyName.O, gui.KeyName.P] and self.scale_factor is not None:
            def translate(tx=0, ty=0):
                T = np.eye(3)
                T[0:2,2] = [tx, ty]
                return T
            def scale(s=1, sx=1, sy=1):
                T = np.diag([s*sx, s*sy, 1])
                return T
            def rotate(degrees):
                T = np.eye(3)
                # just involves some sin() and cos()
                T[0:2] = cv2.getRotationMatrix2D(center=(0,0), angle=-degrees, scale=1.0)
                return T
            translate_factor = 10
            if event.key == gui.KeyName.I:
                self.icy -= translate_factor
            if event.key == gui.KeyName.K:
                self.icy += translate_factor
            if event.key == gui.KeyName.J:
                self.icx -= translate_factor
            if event.key == gui.KeyName.L:
                self.icx += translate_factor
            if event.key == gui.KeyName.U:
                self.scale_factor += 0.1
            if event.key == gui.KeyName.O:
                self.scale_factor -= 0.1
            if event.key == gui.KeyName.P:
                self._reset_image_viewer()
            if self.icy < 0:
                self.icy = 0
            if self.icx < 0:
                self.icx = 0
            if self.icy > self.H:
                self.icy = self.H
            if self.icx > self.W:
                self.icx = self.W
            if self.scale_factor < 0.1:
                self.scale_factor = 0.1
            if self.scale_factor > 10:
                self.scale_factor = 10
            (ow, oh) = (self.W, self.H) # output size
            (ocx, ocy) = ((ow-1)/2, (oh-1)/2) # put there in output (it's the exact center)
            H = translate(+ocx, +ocy) @ rotate(degrees=0) @ scale(self.scale_factor) @ translate(-self.icx, -self.icy)
            M = H[0:2]
            def img_wrapper(img):
                out = cv2.warpAffine(img.copy(), dsize=(ow,oh), M=M, flags=cv2.INTER_NEAREST)
                ratio = 640 / self.W
                img = cv2.resize(out, (640, int(self.H*ratio)))
                return o3d.geometry.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            self._img_wrapper = img_wrapper
            
            
            # out = cv2.warpAffine(self.rgb_img.copy(), dsize=(ow,oh), M=M, flags=cv2.INTER_NEAREST)
            # ratio = 640 / self.W
            # _rgb_img = cv2.resize(out, (640, int(self.H*ratio)))
            # _rgb_img = o3d.geometry.Image(cv2.cvtColor(_rgb_img, cv2.COLOR_BGR2RGB))
            self._rgb_proxy.set_widget(gui.ImageWidget(self._img_wrapper(self.rgb_img)))
            
            # out = cv2.warpAffine(self.diff_img.copy(), dsize=(ow,oh), M=M, flags=cv2.INTER_NEAREST)
            # ratio = 640 / self.W
            # _diff_img = cv2.resize(out, (640, int(self.H*ratio)))
            # _diff_img = o3d.geometry.Image(cv2.cvtColor(_diff_img, cv2.COLOR_BGR2RGB))
            self._diff_proxy.set_widget(gui.ImageWidget(self._img_wrapper(self.diff_img)))
            
            return gui.Widget.EventCallbackResult.HANDLED
        
        # save label
        if event.key==gui.KeyName.F and event.type==gui.KeyEvent.DOWN:
            self._on_save_label()
            return gui.Widget.EventCallbackResult.HANDLED
        
        # convert hand
        if (event.key == gui.KeyName.TAB) and (event.type==gui.KeyEvent.DOWN):
            self._convert_hand()
            return gui.Widget.EventCallbackResult.CONSUMED
        
        # mode change
        if event.key == gui.KeyName.F1:
            self._convert_mode(LabelingMode.STATIC)
            return gui.Widget.EventCallbackResult.CONSUMED
        elif event.key == gui.KeyName.F2:
            self._convert_mode(LabelingMode.OPTIMIZE)
            return gui.Widget.EventCallbackResult.CONSUMED
        elif event.key == gui.KeyName.F3:
            self._convert_mode(LabelingMode.OPTIMIZE_SHAPE)
            return gui.Widget.EventCallbackResult.CONSUMED
        # reset hand pose
        if event.key == gui.KeyName.R:
            if self._labeling_mode==LabelingMode.OPTIMIZE_SHAPE:
                self._active_hand.reset_shape()
            else:
                if self.reset_flat:
                    self._active_hand.reset_pose(flat_hand=True)
                else:
                    self._active_hand.reset_pose()
            self._update_activate_hand()
            self._update_target_hand()
            return gui.Widget.EventCallbackResult.CONSUMED

        
        
        # convert finger
        is_converted_finger = True
        if event.key == gui.KeyName.BACKTICK:
            self._active_hand.set_optimize_target('root')
        elif event.key == gui.KeyName.ONE and (event.type==gui.KeyEvent.DOWN):
            if self._active_hand.get_optimize_target()=='thumb':
                if self.joint_back:
                    ctrl_idx = self._active_hand.control_idx - 1
                else:
                    ctrl_idx = self._active_hand.control_idx + 1
                self._active_hand.set_control_joint(ctrl_idx)
            else:
                self._active_hand.set_optimize_target('thumb')
        elif event.key == gui.KeyName.TWO and (event.type==gui.KeyEvent.DOWN):
            if self._active_hand.get_optimize_target()=='fore':
                if self.joint_back:
                    ctrl_idx = self._active_hand.control_idx - 1
                else:
                    ctrl_idx = self._active_hand.control_idx + 1
                self._active_hand.set_control_joint(ctrl_idx)
            else:
                self._active_hand.set_optimize_target('fore')
        elif event.key == gui.KeyName.THREE and (event.type==gui.KeyEvent.DOWN):
            if self._active_hand.get_optimize_target()=='middle':
                if self.joint_back:
                    ctrl_idx = self._active_hand.control_idx - 1
                else:
                    ctrl_idx = self._active_hand.control_idx + 1
                self._active_hand.set_control_joint(ctrl_idx)
            else:
                self._active_hand.set_optimize_target('middle')
        elif event.key == gui.KeyName.FOUR and (event.type==gui.KeyEvent.DOWN):
            if self._active_hand.get_optimize_target()=='ring':
                if self.joint_back:
                    ctrl_idx = self._active_hand.control_idx - 1
                else:
                    ctrl_idx = self._active_hand.control_idx + 1
                self._active_hand.set_control_joint(ctrl_idx)
            else:
                self._active_hand.set_optimize_target('ring')
        elif event.key == gui.KeyName.FIVE and (event.type==gui.KeyEvent.DOWN):
            if self._active_hand.get_optimize_target()=='little':
                if self.joint_back:
                    ctrl_idx = self._active_hand.control_idx - 1
                else:
                    ctrl_idx = self._active_hand.control_idx + 1
                self._active_hand.set_control_joint(ctrl_idx)
            else:
                self._active_hand.set_optimize_target('little')
        else:
            is_converted_finger = False
        
        if is_converted_finger:
            self._update_target_hand()
            self._update_joint_mask()
            return gui.Widget.EventCallbackResult.CONSUMED

        # if event.key == gui.KeyName.B:
        #     if self._active_hand is None:
        #         return gui.Widget.EventCallbackResult.IGNORED
        #     self._on_icp()
        #     return gui.Widget.EventCallbackResult.CONSUMED
        
        if self._labeling_mode==LabelingMode.OPTIMIZE:
            # optimze
            if event.key == gui.KeyName.SPACE:
                if self._active_hand is None:
                    return gui.Widget.EventCallbackResult.IGNORED
                self._on_optimize()
                return gui.Widget.EventCallbackResult.CONSUMED
            # reset guide pose
            elif event.key == gui.KeyName.HOME:
                self._active_hand.reset_target()
                self._update_target_hand()
                return gui.Widget.EventCallbackResult.CONSUMED
        elif self._labeling_mode==LabelingMode.OPTIMIZE_SHAPE:
            if event.key == gui.KeyName.SPACE:
                if self._active_hand is None:
                    return gui.Widget.EventCallbackResult.IGNORED
                self._on_shape_optim()
                return gui.Widget.EventCallbackResult.CONSUMED
        
        # Translation
        if event.type!=gui.KeyEvent.UP:
            if not self._left_shift_modifier and \
                (self._labeling_mode==LabelingMode.OPTIMIZE or self._active_hand.get_optimize_target()=='root'):
                if event.key == gui.KeyName.D:
                    self.move( self.dist, 0, 0, 0, 0, 0)
                elif event.key == gui.KeyName.A:
                    self.move( -self.dist, 0, 0, 0, 0, 0)
                elif event.key == gui.KeyName.S:
                    self.move( 0, -self.dist, 0, 0, 0, 0)
                elif event.key == gui.KeyName.W:
                    self.move( 0, self.dist, 0, 0, 0, 0)
                elif event.key == gui.KeyName.Q:
                    self.move( 0, 0, -self.dist, 0, 0, 0)
                elif event.key == gui.KeyName.E:
                    self.move( 0, 0, self.dist, 0, 0, 0)
            # Rot - keystrokes are not in same order as translation to make movement more human intuitive
            elif self._left_shift_modifier and \
                (self._labeling_mode==LabelingMode.STATIC or self._active_hand.get_optimize_target()=='root'):
                if event.key == gui.KeyName.E:
                    self.move( 0, 0, 0, 0, 0, self.deg * np.pi / 180)
                elif event.key == gui.KeyName.Q:
                    self.move( 0, 0, 0, 0, 0, -self.deg * np.pi / 180)
                elif event.key == gui.KeyName.A:
                    self.move( 0, 0, 0, 0, self.deg * np.pi / 180, 0)
                elif event.key == gui.KeyName.D:
                    self.move( 0, 0, 0, 0, -self.deg * np.pi / 180, 0)
                elif event.key == gui.KeyName.S:
                    self.move( 0, 0, 0, self.deg * np.pi / 180, 0, 0)
                elif event.key == gui.KeyName.W:
                    self.move( 0, 0, 0, -self.deg * np.pi / 180, 0, 0)
                self._add_hand_frame()

        return gui.Widget.EventCallbackResult.IGNORED
    def _on_tick(self):
        if (time.time()-self._last_change) > 1:
            if self._active_hand is None:
                self._log.text = "\t라벨링 대상 파일을 선택하세요."
                self.window.set_needs_layout()
            else:
                self._log.text = "{} 라벨링 중입니다.".format(self._active_hand.get_control_joint_name())
                self.window.set_needs_layout()
        
        if self._auto_optimize.checked and self._active_hand is not None:
            if self._labeling_mode==LabelingMode.OPTIMIZE:
                self._on_optimize()
        
        if self._auto_save.checked and self.annotation_scene is not None:
            if (time.time()-self._last_saved) > self._auto_save_interval.double_value and self._annotation_changed:
                self._annotation_changed = False
                self.annotation_scene.save_label()
                # self._update_valid_error()
                # self._update_diff_viewer()
                self._last_saved = time.time()
                self._log.text = "라벨 결과 자동 저장중입니다."
                self.window.set_needs_layout()
                    
        
        self._init_view_control()

def main():
    gui.Application.instance.initialize()
    
    font = gui.FontDescription(hangeul)
    font.add_typeface_for_language(hangeul, "ko")
    gui.Application.instance.set_font(gui.Application.DEFAULT_FONT_ID, font)

    w = AppWindow(1920, 1080)
    gui.Application.instance.run()


if __name__ == "__main__":
    main()