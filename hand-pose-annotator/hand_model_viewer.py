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

class LabelingMode:
    STATIC      = "F1. 직접 움직여 라벨링"
    OPTIMIZE    = "F2. 가이드 기반 라벨링"




class HandModel:
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
        self.side = side
        self.mano_layer = ManoLayer(mano_root=MANO_PATH, side=side,
                            use_pca=False, flat_hand_mean=True, joint_rot_mode='axisang')
        self.learning_rate = 1e-3
        self.joint_loss = torch.nn.MSELoss()
        if shape_param is None:
            shape_param = torch.zeros(10)
        self.reset(shape_param)
        
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
            self.shape_param = torch.Tensor(shape_param).unsqueeze(0)

        #2. root translation
        self.root_trans = torch.zeros(1, 3)
        
        #3. root, 15 joints
        if flat_hand:
            self.joint_rot = [torch.zeros(1, 3) for _ in range(16)]
        else:
            self.joint_rot = [torch.zeros(1, 3)]
            self.joint_rot += [torch.Tensor(self.mano_layer.smpl_data['hands_mean'][3*i:3*(i+1)]).unsqueeze(0) for i in range(15)]
            
        self.optimizer = None
        self.update_mano()
        self.root_delta = self.joints.cpu().detach()[0, 0]
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

    #region mano model
    def update_mano(self):
        pose_param = torch.concat(self.joint_rot, dim=1)
        verts, joints = self.mano_layer(th_pose_coeffs=pose_param,
                                        th_betas=self.shape_param,
                                        th_trans=self.root_trans)
        self.verts = verts / 1000
        self.joints = joints / 1000
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

    def reset_pose(self):
        if self.optimize_state==LabelingMode.OPTIMIZE:
            if self.optimize_target=='root':
                self.joint_rot[0] = torch.zeros(1, 3)
                self.joint_rot[0].requires_grad = True
            else:
                for target_idx in [self._ORDER_OF_PARAM[self.optimize_target]*3+i+1 for i in range(3)]:
                    self.joint_rot[target_idx] = torch.Tensor(self.mano_layer.smpl_data['hands_mean'][3*target_idx-3:3*target_idx]).unsqueeze(0)
                    self.joint_rot[target_idx].requires_grad = True
            self.optimizer = optim.Adam(self.joint_rot, lr=self.learning_rate)
            self.update_mano()
        else:
            target_idx = self._get_control_joint_param_idx()
            self.joint_rot[target_idx] = torch.zeros(1, 3)
            self.update_mano()    
    
    def get_control_rotation(self):
        target_idx = self._get_control_joint_param_idx()
        return self.joint_rot[target_idx].detach()[0, :]
    def set_control_rotation(self, rot_mat):
        assert (self.optimize_state==LabelingMode.STATIC or self.optimize_target=='root'), "error on set_control_rotation"
        target_idx = self._get_control_joint_param_idx()
        if self.optimize_state==LabelingMode.OPTIMIZE and self.optimize_target=='root':
            self.joint_rot[0] = torch.Tensor(rot_mat).unsqueeze(0)
            self.joint_rot[0].requires_grad = True
            self.update_mano()
            self.reset_target()
        else:    
            self.joint_rot[target_idx] = torch.Tensor(rot_mat).unsqueeze(0)
            self.update_mano()
    
    def get_control_position(self):
        target_idx = self._get_control_joint_idx()
        if self.optimize_state==LabelingMode.STATIC:
            return self.joints.detach()[0, target_idx]
        else:
            return self.targets.detach()[0, target_idx]
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
        self.root_trans = torch.Tensor(xyz).unsqueeze(0) - self.root_delta
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
            'root_delta': np.array(self.root_delta),
            
            'joints': np.array(self.joints.cpu().detach()[0, :]),
            'verts': np.array(self.verts.cpu().detach()[0, :]),
            'faces': np.array(self.faces.cpu().detach()[0, :])
        }
    def set_state(self, state, only_pose=False):
        if only_pose:
            pose_param = torch.Tensor(state['pose_param']).unsqueeze(0) # 1, 48
            self.joint_rot = [pose_param[:, 3*i:3*(i+1)] for i in range(16)]
            self.root_trans = torch.Tensor(state['root_trans']).unsqueeze(0)
            self.update_mano()
        else:
            self.shape_param = torch.Tensor(state['shape_param']).unsqueeze(0)
            assert state['pose_param'].size==48
            pose_param = torch.Tensor(state['pose_param']).unsqueeze(0) # 1, 48
            self.joint_rot = [pose_param[:, 3*i:3*(i+1)] for i in range(16)]
            self.root_trans = torch.Tensor(state['root_trans']).unsqueeze(0)
            self.root_delta = torch.Tensor(state['root_delta'])
        
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
            return True
        else:
            return False
    
    def get_target(self):
        return self.targets.cpu().detach()[0, :]
    
    def set_target(self, targets):
        self.targets = torch.Tensor(targets).unsqueeze(0)
        self.targets.requires_grad = True
        self._target_changed = True

    def _mse_loss(self):
        assert self.optimize_state==LabelingMode.OPTIMIZE
        target_idx = self.optimize_idx[self.optimize_target]
        return self.joint_loss(self.joints[:, target_idx], self.targets[:, target_idx])
    
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
            "link": self._get_links()
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

    def get_target_geometry(self):
        return {
            "joint": self._get_target_joints(),
            "link": self._get_target_links(),
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
        elif self.optimize_state==LabelingMode.STATIC:
            return {
            "control": self._get_joints([self.contorl_joint])
        }
        else:
            raise NotImplementedError

class PCAHandModel(HandModel):
    def __init__(self, side, shape_param=None):
        self.side = side
        self.mano_layer = ManoLayer(mano_root=MANO_PATH, side=side,
                            use_pca=True, flat_hand_mean=True, joint_rot_mode='axisang')
        self.learning_rate = 1e-2
        self.joint_loss = torch.nn.MSELoss()
        if shape_param is None:
            shape_param = torch.zeros(10)
        self.reset(shape_param)

    def reset(self, shape_param=None, flat_hand=True):
        #1. shape
        if shape_param is None:
            pass
        else:
            self.shape_param = torch.Tensor(shape_param).unsqueeze(0)

        #2. root translation
        self.root_trans = torch.zeros(1, 3)
        
        #3. root, 15 joints
        self.joint_rot = [torch.zeros(1, 3)]
        self.joint_rot += [torch.zeros(1, 1) for _ in range(6)]
            
        self.optimizer = None
        self.update_mano()
        self.root_delta = self.joints.cpu().detach()[0, 0]
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

    def update_mano(self):
        pose_param = torch.concat(self.joint_rot, dim=1)
        verts, joints = self.mano_layer(th_pose_coeffs=pose_param,
                                        th_betas=self.shape_param,
                                        th_trans=self.root_trans)
        self.verts = verts / 1000
        self.joints = joints / 1000
        self.faces = self.mano_layer.th_faces

    def set_param(self, idx, value):
        self.joint_rot[idx+1] = torch.Tensor([value]).unsqueeze(0)
        self.update_mano()

    def get_param_num(self):
        return 6

class BetaHandModel(HandModel):
    def __init__(self, side, shape_param=None):
        self.side = side
        self.mano_layer = ManoLayer(mano_root=MANO_PATH, side=side,
                            use_pca=True, flat_hand_mean=True, joint_rot_mode='axisang')
        self.learning_rate = 1e-2
        self.joint_loss = torch.nn.MSELoss()
        if shape_param is None:
            shape_param = torch.zeros(10)
        self.reset(shape_param)

    def reset(self, shape_param=None, flat_hand=True):
        #1. shape
        if shape_param is None:
            pass
        else:
            self.shape_param = torch.Tensor(shape_param).unsqueeze(0)
            self.shape_param_control = [torch.zeros(1, 1) for _ in range(10)]

        #2. root translation
        self.root_trans = torch.zeros(1, 3)
        
        #3. root, 15 joints
        self.joint_rot = [torch.zeros(1, 3)]
        self.joint_rot += [torch.zeros(1, 1) for _ in range(6)]
            
        self.optimizer = None
        self.update_mano()
        self.root_delta = self.joints.cpu().detach()[0, 0]
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

    def update_mano(self):
        pose_param = torch.concat(self.joint_rot, dim=1)
        shape_param = torch.concat(self.shape_param_control, dim=1)
        verts, joints = self.mano_layer(th_pose_coeffs=pose_param,
                                        th_betas=shape_param,
                                        th_trans=self.root_trans)
        self.verts = verts / 1000
        self.joints = joints / 1000
        self.faces = self.mano_layer.th_faces

    def set_param(self, idx, value):
        self.shape_param_control[idx] = torch.Tensor([value]).unsqueeze(0)
        self.update_mano()

    def get_param_num(self):
        return 10


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


class AppWindow:
    
    
    def __init__(self, width, height):
        #---- geometry name
        self._window_name = "Mano Hand Model Viewer by GIST AILAB"
        
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
        
        self._labeling_mode = LabelingMode.STATIC
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
        # self.hand_models = {
        #     "right": PCAHandModel(side='right'),
        #     "left": PCAHandModel(side='left')
        # }
        # self.hand_models = {
        #     "right": HandModel(side='right'),
        #     "left": HandModel(side='left')
        # }
        self.hand_models = {
            "right": BetaHandModel(side='right'),
            "left": BetaHandModel(side='left')
        }
        
        
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
        
        self._init_viewctrl_layout()
        self._init_handedit_layout()
        self._init_stageedit_layout()
        self._init_parameter_edit_layout()

        # 3D Annotation tool options
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        w.set_on_layout(self._on_layout)
        
        # ---- annotation tool settings ----
        self._initialize_background()
        self._on_hand_transparency(0.2)
        self._on_hand_point_size(10) # set default size to 10
        self._on_hand_line_size(2) # set default size to 2
        self._on_responsiveness(5) # set default responsiveness to 5
        
        self._scene.set_on_mouse(self._on_mouse)
        self._scene.set_on_key(self._on_key)
        self.window.set_on_tick_event(self._on_tick)
        self.window.set_needs_layout()

        self._init_hand_layer()
        self._on_initial_viewpoint()

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
    
    def _initialize_background(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._set_background_color(bg_color)

    def _on_initial_viewpoint(self):
        self.bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, self.bounds, self.bounds.get_center())
        center = np.array([0, 0, 1])
        eye = np.array([0, 0, 0])
        up = np.array([0, -1, 0])
        self._scene.look_at(center, eye, up)
        self._init_view_control()
    
    def _init_view_control(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
    
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

        grid = gui.VGrid(2, 0.25 * em)
        
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
        
        viewctrl_layout.add_child(grid)
        
        self._settings_panel.add_child(viewctrl_layout)
    def _on_show_hand(self, show):
        if self._active_hand is None:
            self._show_hands.checked = not show
            return
        self.settings.show_hand = show
        self._update_activate_hand()
    def _on_hand_transparency(self, transparency):
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
        self._last_change = time.time()
        mat = self.settings.hand_joint_material
        mat.point_size = int(size)
        if self._check_geometry(self._right_hand_joint_name):
            self._set_geometry_material(self._right_hand_joint_name, mat)
        if self._check_geometry(self._left_hand_joint_name):
            self._set_geometry_material(self._left_hand_joint_name, mat)
        self._hand_point_size.double_value = size
    def _on_hand_line_size(self, size):
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
        self._last_change = time.time()
        self.dist = 0.0004 * responsiveness
        self.deg = 0.2 * responsiveness
        self._responsiveness.double_value = responsiveness
    def _on_optimize_rate(self, optimize_rate):
        self._active_hand.set_learning_rate(optimize_rate*1e-3)
        self.window.set_needs_layout()
        self._last_change = time.time()
        self._optimize_rate.double_value = optimize_rate
    
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
        
        button = gui.Button("현재 대상 리셋 (R)")
        button.set_on_clicked(self._reset_current_hand)
        handedit_layout.add_child(button)
        
        h = gui.Horiz(0.4 * em)
        button = gui.Button("손목 (`)")
        button.horizontal_padding_em = 0.8
        button.vertical_padding_em = 0.2
        button.set_on_clicked(self._convert_to_root)
        h.add_child(button)  
        button = gui.Button("엄지 (1)")
        button.horizontal_padding_em = 0.6
        button.vertical_padding_em = 0.2
        button.set_on_clicked(self._convert_to_thumb)
        h.add_child(button)
        button = gui.Button("검지 (2)")
        button.horizontal_padding_em = 0.6
        button.vertical_padding_em = 0.2
        button.set_on_clicked(self._convert_to_fore)
        h.add_child(button)
        handedit_layout.add_child(h)

        h = gui.Horiz(0.4 * em)
        button = gui.Button("중지 (3)")
        button.horizontal_padding_em = 0.6
        button.vertical_padding_em = 0.2
        button.set_on_clicked(self._convert_to_middle)
        h.add_child(button)
        button = gui.Button("약지 (4)")
        button.horizontal_padding_em = 0.6
        button.vertical_padding_em = 0.2
        button.set_on_clicked(self._convert_to_ring)
        h.add_child(button)
        button = gui.Button("소지 (5)")
        button.horizontal_padding_em = 0.6
        button.vertical_padding_em = 0.2
        button.set_on_clicked(self._convert_to_little)
        h.add_child(button)
        handedit_layout.add_child(h)
        
        h = gui.Horiz(0.4 * em)
        button = gui.Button("이전 관절(PgDn)")
        button.horizontal_padding_em = 0.3
        button.vertical_padding_em = 0
        button.set_on_clicked(self._control_joint_down)
        h.add_child(button)
        button = gui.Button("다음 관절(PgUp)")
        button.horizontal_padding_em = 0.3
        button.vertical_padding_em = 0
        button.set_on_clicked(self._control_joint_up)
        h.add_child(button)
        handedit_layout.add_child(h)

        button = gui.Button("손 바꾸기 (Tab)")
        button.set_on_clicked(self._convert_hand)
        handedit_layout.add_child(button)
        
        self._settings_panel.add_child(handedit_layout)
    def _reset_current_hand(self):
        self._active_hand.reset_pose()
        self._update_activate_hand()
        self._update_target_hand()
    def _convert_hand(self):
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
        self._convert_mode(LabelingMode.STATIC)
        self._update_current_hand_str()

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
        self._update_target_hand()
        self._update_current_hand_str()
    def _control_joint_down(self):
        if self._active_hand is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _control_joint_down)")
            return
        ctrl_idx = self._active_hand.control_idx - 1
        self._active_hand.set_control_joint(ctrl_idx)
        self._update_target_hand()
        self._update_current_hand_str()
    def _update_current_hand_str(self):
        self._current_hand_str.text = "현재 대상: {}".format(self._active_hand.get_control_joint_name())

    def _init_parameter_edit_layout(self):
        em = self.window.theme.font_size
        param_layout = gui.CollapsableVert("파라미터 제어", 0.33*em,
                                          gui.Margins(em, 0, 0, 0))
        param_layout.set_is_open(True)
        
        grid = gui.VGrid(2, 0.25 * em)

        def add_param_slide(idx):
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(-10, 10)
            slider.set_on_value_changed(self.__getattribute__("_on_param_{}".format(idx)))
            grid.add_child(gui.Label("param {}".format(idx)))
            grid.add_child(slider)
            self.__setattr__("param_{}".format(idx), slider)

        nparam = self.hand_models['right'].get_param_num()
        for i in range(nparam):
            add_param_slide(i)
        param_layout.add_child(grid)
        self._settings_panel.add_child(param_layout)

    def _on_param_0(self, value):
        self.param_0.double_value = value
        self._active_hand.set_param(0, value)
        self._update_activate_hand()
        self._update_target_hand()
    def _on_param_1(self, value):
        self.param_1.double_value = value
        self._active_hand.set_param(1, value)
        self._update_activate_hand()
        self._update_target_hand()
    def _on_param_2(self, value):
        self.param_2.double_value = value
        self._active_hand.set_param(2, value)
        self._update_activate_hand()
        self._update_target_hand()
    def _on_param_3(self, value):
        self.param_3.double_value = value
        self._active_hand.set_param(3, value)
        self._update_activate_hand()
        self._update_target_hand()
    def _on_param_4(self, value):
        self.param_4.double_value = value
        self._active_hand.set_param(4, value)
        self._update_activate_hand()
        self._update_target_hand()
    def _on_param_5(self, value):
        self.param_5.double_value = value
        self._active_hand.set_param(5, value)
        self._update_activate_hand()
        self._update_target_hand()
    def _on_param_6(self, value):
        self.param_6.double_value = value
        self._active_hand.set_param(6, value)
        self._update_activate_hand()
        self._update_target_hand()
    def _on_param_7(self, value):
        self.param_7.double_value = value
        self._active_hand.set_param(7, value)
        self._update_activate_hand()
        self._update_target_hand()
    def _on_param_8(self, value):
        self.param_8.double_value = value
        self._active_hand.set_param(8, value)
        self._update_activate_hand()
        self._update_target_hand()
    def _on_param_9(self, value):
        self.param_9.double_value = value
        self._active_hand.set_param(9, value)
        self._update_activate_hand()
        self._update_target_hand()



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
                                       add_downsampled_copy_for_fast_rendering=True)
    
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
    
    def _init_hand_layer(self):
        # visualize hand
        hands = self.hand_models
        if self._is_right_hand > 0:
            active_side = 'right'
        else:
            active_side = 'left'
        self._hands = hands
        self._active_hand = hands[active_side]
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
        self._update_current_hand_str()
    def _update_target_hand(self):
        if self._labeling_mode==LabelingMode.OPTIMIZE:
            target_geo = self._active_hand.get_target_geometry()
            self._add_geometry(self._target_joint_name, 
                            target_geo['joint'], self.settings.target_joint_material)
            self._add_geometry(self._target_link_name, 
                            target_geo['link'], self.settings.target_link_material)
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
        self.window.set_needs_layout()
        self._last_change = time.time()
        self._annotation_changed = self._active_hand.optimize_to_target()
        self._update_target_hand()
        self._update_activate_hand()
    
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

        if event.key == gui.KeyName.T and event.type == gui.KeyEvent.DOWN:
            self._on_active_camera_viewpoint()
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key == gui.KeyName.Y and event.type == gui.KeyEvent.DOWN:
            self._on_active_viewpoint()
            return gui.Widget.EventCallbackResult.HANDLED

        # undo / redo
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
        # reset hand pose
        if event.key == gui.KeyName.R:
            self._active_hand.reset_pose()
            self._update_activate_hand()
            self._update_target_hand()
            return gui.Widget.EventCallbackResult.CONSUMED

        # if shift pressed then rotation else translation
        if (event.key == gui.KeyName.LEFT_SHIFT or event.key == gui.KeyName.RIGHT_SHIFT) \
            and (self._labeling_mode==LabelingMode.STATIC or self._active_hand.get_optimize_target()=='root'):
            if event.type == gui.KeyEvent.DOWN:
                self._left_shift_modifier = True
                self._add_hand_frame()
            elif event.type == gui.KeyEvent.UP:
                self._left_shift_modifier = False
                self._remove_hand_frame()
            return gui.Widget.EventCallbackResult.HANDLED
        
        # if ctrl is pressed then increase translation
        if event.key == gui.KeyName.LEFT_CONTROL or event.key == gui.KeyName.RIGHT_CONTROL:
            if event.type == gui.KeyEvent.DOWN:
                if not self.upscale_responsiveness:
                    self.dist = self.dist * 15
                    self.deg = self.deg * 15
                    self.upscale_responsiveness = True
            elif event.type == gui.KeyEvent.UP:
                if self.upscale_responsiveness:
                    self.dist = self.dist / 15
                    self.deg = self.deg / 15
                    self.upscale_responsiveness = False
            return gui.Widget.EventCallbackResult.HANDLED
        
        # convert finger
        is_converted_finger = True
        if event.key == gui.KeyName.BACKTICK:
            self._active_hand.set_optimize_target('root')
        elif event.key == gui.KeyName.ONE:
            self._active_hand.set_optimize_target('thumb')
        elif event.key == gui.KeyName.TWO:
            self._active_hand.set_optimize_target('fore')
        elif event.key == gui.KeyName.THREE:
            self._active_hand.set_optimize_target('middle')
        elif event.key == gui.KeyName.FOUR:
            self._active_hand.set_optimize_target('ring')
        elif event.key == gui.KeyName.FIVE:
            self._active_hand.set_optimize_target('little')
        else:
            is_converted_finger = False
        
        # convert joint
        is_convert_joint = True
        if event.key == gui.KeyName.PAGE_UP and (event.type==gui.KeyEvent.DOWN):
            ctrl_idx = self._active_hand.control_idx + 1
            self._active_hand.set_control_joint(ctrl_idx)
        elif event.key == gui.KeyName.PAGE_DOWN and (event.type==gui.KeyEvent.DOWN):
            ctrl_idx = self._active_hand.control_idx - 1
            self._active_hand.set_control_joint(ctrl_idx)
        else:
            is_convert_joint = False
        
        if is_converted_finger or is_convert_joint:
            self._update_target_hand()
            return gui.Widget.EventCallbackResult.CONSUMED

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
        if self._auto_optimize.checked and self._active_hand is not None:
            if self._labeling_mode==LabelingMode.OPTIMIZE:
                self._on_optimize()
        
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
