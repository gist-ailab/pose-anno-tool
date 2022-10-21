# Author: Raeyoung Kang (raeyo@gm.gist.ac.kr)
# GIST AILAB, Republic of Koreawidget
# Modified from the codes of Anas Gouda (anas.gouda@tu-dortmund.de)
# FLW, TU Dortmund, Germany
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import logging
import atexit
import matplotlib as mpl

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

import psutil

MANO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "mano")
hangeul = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "NanumGothic.ttf")

temp_side_info = {
4:	['left'],
5:	['right'],
6:	['right'],
7:	['right'],
9:	['left'],
10:	['left'],
11:	['left'],
14:	['right'],
15:	['right'],
16:	['right'],
18:	['left'],
22:	['right'],
23:	['right'],
24:	['right'],
25:	['left'],
26:	['left'],
27:	['left'],
28:	['left'],
29:	['left'],
30:	['left'],
31:	['left'],
32:	['left'],
33:	['left'],
34:	['left'],
35:	['left'],
36:	['left'],
37:	['right'],
38:	['right'],
39:	['right'],
40:	['right'],
41:	['right'],
42:	['right'],
43:	['right'],
44:	['right'],
45:	['right'],
46:	['right'],
47:	['right'],
48:	['right'],
49:	['left'],
50:	['left'],
51:	['left'],
52:	['left'],
53:	['left'],
54:	['left'],
55:	['left'],
56:	['left'],
57:	['left'],
58:	['left'],
59:	['left'],
60:	['left'],
61:	['right'],
62:	['right'],
63:	['right'],
64:	['right'],
65:	['right'],
66:	['right'],
67:	['right'],
68:	['right'],
69:	['right'],
70:	['right'],
71:	['right'],
72:	['right'],
73:	['left'],
74:	['left'],
75:	['left'],
76:	['left'],
77:	['left'],
78:	['left'],
79:	['left'],
80:	['left'],
81:	['left'],
82:	['left'],
83:	['left'],
84:	['left'],
85:	['right'],
86:	['right'],
87:	['right'],
88:	['right'],
89:	['right'],
90:	['right'],
91:	['right'],
92:	['right'],
93:	['right'],
94:	['right'],
95:	['right'],
96:	['right'],
}



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

    @staticmethod
    def calc_chamfer_distance(p1, p2):
        p1 = Pointclouds(torch.Tensor(p1).unsqueeze(0))
        p2 = Pointclouds(torch.Tensor(p2).unsqueeze(0))
        return chamfer_distance(p1, p2)[0]

class Logger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)

        fomatter = logging.Formatter('[%(levelname)s|%(lineno)s] %(asctime)s > %(message)s')
        
        fileHandler = logging.FileHandler('./debug.log')
        fileHandler.setFormatter(fomatter)
        self.logger.addHandler(fileHandler)

        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(fomatter)
        self.logger.addHandler(streamHandler)

        self.logger.setLevel(logging.DEBUG)

        self.last_msg = "="*50
        self.logger.info("="*50)
        self.logger.info("Start Logging")
        
        self.last_check_memory = time.time()
        self.check_memory_inteval = 30

    def handle_exit(self):
        self.logger.info("End Logging")
        self.logger.info("="*50)

    def info(self, msg):
        if self.last_msg == msg:
            return
        self.last_msg = msg
        self.logger.info(msg)
    def debug(self, msg):
        if self.last_msg == msg:
            return
        self.last_msg = msg
        self.logger.debug(msg)
    
    def memory_usage(self):
        if (time.time()-self.last_check_memory) < self.check_memory_inteval:
            return
        self.last_check_memory = time.time()
        p = psutil.Process()
        rss = p.memory_info().rss / 2 ** 20 # Bytes to MB
        msg = f"memory usage: {rss: 10.5f} MB"
        self.logger.info(msg)
        
class LabelingMode:
    STATIC      = "F1. 직접 움직여 라벨링"
    OPTIMIZE    = "F2. 가이드 기반 라벨링"
    OBJECT      = "F3. 물체 라벨링"
    # MESH        = "F4. 메쉬 라벨링"

class MaskMode:
    RGB_ALL     = "RGB 전체"
    RGB_RIGHT   = "RGB 오른손"
    RGB_LEFT    = "RGB 왼손"
    RGB_OBJECT  = "RGB 물체"
    MASK_ALL    = "MASK 전체"
    MASK_RIGHT  = "MASK 오른손"
    MASK_LEFT   = "MASK 왼손"
    MASK_OBJECT = "MASK 물체"
       
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
        self.side = side
        self.mano_layer = ManoLayer(mano_root=MANO_PATH, side=side,
                            use_pca=False, flat_hand_mean=True, joint_rot_mode='axisang')
        
        self.verts_to_points, self.sampled_face_idx = torch.load(os.path.join(MANO_PATH, 'verts_to_points_{}.pt'.format(side)))
        self.active_faces = []
        
        self.learning_rate = 1e-3
        self.joint_loss = torch.nn.MSELoss()
        if side=='right':
            self._active_img = self.RIGHT_JOINT_IMG
            self._nactive_img = self.LEFT_JOINT_IMG
        else:
            self._nactive_img = self.RIGHT_JOINT_IMG
            self._active_img = self.LEFT_JOINT_IMG
        self.init_joint_mask()

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
        
    def reset(self, shape_param=None, flat_hand=False):
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
        with torch.no_grad():
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
        self.lock_state = [False for _ in range(17)]
        self.save_undo(forced=True)
    
    #region mano model
    def update_mano(self):
        pose_param = torch.concat(self.joint_rot, dim=1)
        verts, joints = self.mano_layer(th_pose_coeffs=pose_param,
                                        th_betas=self.shape_param,
                                        th_trans=self.root_trans)
        self.verts = verts / 1000
        self.joints = joints / 1000
        # self.verts = verts
        # self.joints = joints
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
    def _param2control_idx(self, param_idx):
        order = (param_idx-1)//3 
        if order==0: 
            finger = 1
        elif order==1: 
            finger = 2
        elif order==2: 
            finger = 4
        elif order==3: 
            finger = 3
        elif order==4: 
            finger = 0
        else:
            return 0
        return 4*finger + ((param_idx+2)%3 + 1)
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

    def reset_pose(self, flat_hand=False):
        if self.optimize_state==LabelingMode.OPTIMIZE:
            if self.optimize_target=='root':
                if not self.lock_state[1]:
                    self.joint_rot[0] = torch.zeros(1, 3)
                    self.joint_rot[0].requires_grad = True
            else:
                for target_idx in [self._ORDER_OF_PARAM[self.optimize_target]*3+i+1 for i in range(3)]:
                    if self.lock_state[target_idx+1]:
                        continue
                    if flat_hand:
                        self.joint_rot[target_idx] = torch.zeros(1, 3)
                    else:
                        self.joint_rot[target_idx] = torch.Tensor(self.mano_layer.smpl_data['hands_mean'][3*target_idx-3:3*target_idx]).unsqueeze(0)
                    self.joint_rot[target_idx].requires_grad = True
            self.optimizer = optim.Adam(self.joint_rot, lr=self.learning_rate)
            self.update_mano()
        else:
            target_idx = self._get_control_joint_param_idx()
            if not self.lock_state[target_idx+1]:
                self.joint_rot[target_idx] = torch.zeros(1, 3)
            self.update_mano()
    
    def get_control_rotation(self):
        target_idx = self._get_control_joint_param_idx()
        return self.joint_rot[target_idx].detach()[0, :]
    def set_control_rotation(self, rot_mat):
        assert (self.optimize_state==LabelingMode.STATIC or self.optimize_target=='root'), "error on set_control_rotation"
        target_idx = self._get_control_joint_param_idx()
        if self.lock_state[target_idx+1]:
            return
        if self.optimize_state==LabelingMode.OPTIMIZE and self.optimize_target=='root':
            self.joint_rot[target_idx] = torch.Tensor(rot_mat).unsqueeze(0)
            self.joint_rot[target_idx].requires_grad = True
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
        if self.lock_state[0]:
            return False
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
        # elif self.optimize_state==LabelingMode.MESH:
        #     self.optimize_idx = self._IDX_OF_GUIDE
        #     self.set_optimize_target('whole')
        #     self.update_mano()
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
        if self.optimize_target=='whole':
            self.root_trans.requires_grad = True
            for param in self.joint_rot:
                param.requires_grad = True
            self.optimizer = optim.Adam([self.root_trans, *self.joint_rot], lr=0.005)
            return
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
        
    def toggle_current_joint_lock(self):
        if self.control_idx > 2:
            return
        target_idx = self._get_control_joint_param_idx()
        if target_idx==0: # root
            self.lock_state[0] = not self.lock_state[0]
            self.lock_state[1] = not self.lock_state[1]
        else:
            self.lock_state[target_idx+1] = not self.lock_state[target_idx+1]

    def _check_lock_state(self):
        for idx, lock_state in enumerate(self.lock_state):
            if lock_state:
                if idx == 0:
                    self.root_trans.requires_grad = False
                    self.root_trans.grad = None
                else:
                    idx = idx - 1
                    self.joint_rot[idx].requires_grad = False
                    self.joint_rot[idx].grad = None
    @property
    def _can_be_move(self):
        is_lock = []
        if self.optimize_target == 'whole':
            is_lock = self.lock_state
        else:
            for i in range(3):
                is_lock.append(self.lock_state[self._ORDER_OF_PARAM[self.optimize_target]*3 + i + 2])
        return not all(is_lock)

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
            assert state['pose_param'].size==48
            pose_param = torch.Tensor(state['pose_param']).unsqueeze(0) # 1, 48
            self.joint_rot = [pose_param[:, 3*i:3*(i+1)] for i in range(16)]
            self.root_trans = torch.Tensor(state['root_trans']).unsqueeze(0)
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
        if self.optimize_target=='root':
            return False
        if self._target_changed and self._can_be_move:
            self.optimizer.zero_grad()
            # forward
            self.update_mano()
            # loss term
            loss = self._mse_loss()
            loss.backward()

            self._check_lock_state()

            self.optimizer.step()
            self._target_changed = True
            return True
        else:
            return False
    
    def optimize_to_points(self, target_points):
        if self.optimize_target=='root':
            return False
        if self._can_be_move:
            self.optimizer.zero_grad()
            # forward
            self.update_mano()
            # loss term
            loss = self._mesh_to_points_loss(target_points, only_activate=True)
            loss.backward()
            self._check_lock_state()
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
    
    def _mesh_to_points_loss(self, target_points, only_activate=False):
        target_points = Pointclouds(torch.Tensor(target_points).unsqueeze(0))
        meshes = self.get_py3d_mesh(only_activate)
        # meshes = self.get_py3d_mesh()
        return point_mesh_face_distance(meshes, target_points)
    
    def _sampling_points_from_mesh(self):
        meshes = self.get_py3d_mesh()
        return sample_points_from_meshes(meshes, 5000)

    def set_learning_rate(self, lr):
        self.learning_rate = lr
        if self.optimizer is None:
            return
        for g in self.optimizer.param_groups:
            g['lr'] = lr
    #endregion
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
        
    def get_joint_mask(self):
        mask = self.total_mask.copy()
        mask[self.joint_mask['active'][self.contorl_joint]] = [255, 0, 0]
        
        # draw lock state
        for i in range(16):
            if i == 0:
                lock_idx = 0 # + 1
            else:
                lock_idx = i + 1
            is_lock = self.lock_state[lock_idx]

            contorl_joint = self._param2control_idx(i)
            cnt = self._active_img[contorl_joint]

            if is_lock:
                s1, e1 = [cnt[0]-7, cnt[1]-7], [cnt[0]+7, cnt[1]+7]
                s2, e2 = [cnt[0]-7, cnt[1]+7], [cnt[0]+7, cnt[1]-7]
                mask = cv2.line(mask, s1, e1, [255, 0, 0], 2)
                mask = cv2.line(mask, s2, e2, [255, 0, 0], 2)
            else:
                mask = cv2.circle(mask, cnt, 7, [0, 255, 0], 2)
        
        return mask

    def get_py3d_mesh(self, only_activate=False):
        verts = self.verts
        if only_activate:
            # vert_indices = self.faces[self.active_faces]
            # vert_indices = torch.unique(vert_indices)
            # verts = self.verts[:, vert_indices]
            faces = self.faces[self.active_faces].unsqueeze(0)
        else:
            
            faces = self.faces.unsqueeze(0)
        return Meshes(verts=verts, faces=faces)
    
    def get_mesh_points(self):
        verts = self.verts.cpu().detach()[0, :]
        return torch.matmul(self.verts_to_points, verts)

    def get_active_mesh(self, return_inactive=False):
        vert_indices = self.faces[self.active_faces]
        indices = torch.unique(vert_indices).tolist()
        mesh = self._get_mesh()
        active_mesh = mesh.select_by_index(indices)
        if return_inactive:
            inactive_mesh = mesh.__copy__()
            inactive_mesh.remove_triangles_by_index(self.active_faces)
            return active_mesh, inactive_mesh
        else:
            return active_mesh

    def get_face_indices_from_points(self, point_idxs):
        return self.sampled_face_idx[point_idxs]
    def update_active_faces(self, xyz, radius, invert=False):
        active_faces = self.active_faces
        points = self.get_mesh_points()
        dist = np.linalg.norm(points - xyz, axis=1)
        inlier_points = np.where(dist < radius)[0].tolist()
        inlier_faces = np.unique(self.get_face_indices_from_points(inlier_points))
        if invert:
            active_faces = list(set(active_faces) - set(inlier_faces))
        else:
            active_faces = list(set(active_faces) | set(inlier_faces))
        self.active_faces = active_faces

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
 
class Camera:
    def __init__(self, name, serial, intrinsic, extrinsics, folder):
        self.name = name
        self.serial = serial
        self.intrinsic = intrinsic
        self.extrinsics = extrinsics
        self.folder = folder

class SceneObject:
    def __init__(self, obj_id, model_path):
        self.id = obj_id
        self.model_path = model_path
        self.obj_geo = self._load_point_cloud()
        self.obj_mesh = self._load_mesh()
        self.H = np.eye(4)

    def reset(self):
        self.obj_geo.clear()
        self.obj_mesh.clear()
        self.obj_geo = self._load_point_cloud()
        self.obj_mesh = self._load_mesh()
        self.H = np.eye(4)

    def _load_point_cloud(self):
        pcd = o3d.io.read_point_cloud(self.model_path)
        if not pcd.has_normals():
            pcd.estimate_normals()
        pcd.normalize_normals()
        pcd.scale(0.001, [0, 0, 0])
        pcd.translate(-pcd.get_center())
        return pcd
    
    def _load_mesh(self):
        mesh = o3d.io.read_triangle_mesh(self.model_path)
        if len(mesh.triangles)==0:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.obj_geo)[0]
        else:
            mesh.scale(0.001, [0, 0, 0])
        mesh.translate(-mesh.get_center())
        return mesh
    def set_transform(self, H):
        self.reset()
        self.transform(H)
    def get_transform(self):
        return self.H
    def transform(self, H):
        self.obj_geo.transform(H)
        self.obj_mesh.transform(H)
        self.H = np.matmul(H, self.H)

    def set_position(self, xyz):
        delta_xyz = xyz - self.H[:3, 3]
        h_transform = np.eye(4)
        h_transform[:3, 3] = delta_xyz
        self.transform(h_transform)

    def get_geometry(self):
        return self.obj_geo
    
    def get_mesh(self):
        return self.obj_mesh

class Dataset:
    def __init__(self):
        """
        self._data_dir # data_root
        self._cameras # list of camera name (dir name of camera)
        
        self._total_scene = [] # list of scene, idx -> sc_name
        self._scene_path = {} # sc_name: scene dir
        self._scene_hand = {} # sc_name: scene hand info
        self._scene_object = {} # sc_name: scene object info
        self._scene_camera = {} # sc_name: scene camera info
        self._scene_pcd = {} # sc_name: scene points(merged)
        
        hand
        - right: shape(10)
        - left : shape(10)
        object
        - ids: [], # idx of object models
        - models: {} # idx: model_path
        camera
        - intrinsic # 1 per scene
        - extrinsic # 1 per frame
        pcd # list of path of frame
        
        Raises:
            NotImplementedError: _description_
        """
        try:
            if not os.path.isdir(self._data_dir):
                raise NotImplementedError
            if len(self._cameras)==0:
                raise NotImplementedError
            if len(self._total_scene)==0:
                raise NotImplementedError
        except:
            raise NotImplementedError
        
        self.hand_models = {
            "right": HandModel(side='right'),
            "left": HandModel(side='left')
        }
        self.print_dataset_info()
        self.current_scene_idx = -1
    
    def print_dataset_info(self):
        total_scene = 0
        total_frame = 0
        for sc_name in self._total_scene:
            total_scene += 1
            frame_num = len(self._scene_frame[sc_name])
            total_frame += frame_num
            print("Scene: {}| Frame: {}".format(sc_name, frame_num))
        print("Total\nScene: {}\nFrame: {}".format(total_scene, total_frame)) 
        
    def get_scene_from_file(self, file_path):
        sc_name, camera_name, frame_id = self.path_to_info(file_path)
        return self.get_scene(sc_name, frame_id)
    
    def get_scene(self, sc_name, frame_id):
        # scene meta
        sc_path = self._scene_path[sc_name]
        sc_hand_shapes = self._scene_hand[sc_name] 
        sc_objects = self._scene_object[sc_name] 
        sc_cam_info = self._scene_camera[sc_name] 
        sc_frame_list = self._scene_frame[sc_name] 
        
        # hand
        hands = {}
        for side, hand_shape in sc_hand_shapes.items():
            self.hand_models[side].reset(hand_shape)
            hands[side] = self.hand_models[side]

        # object
        objects = {}
        for obj_id, model_path in sc_objects.items():
            objects[obj_id] = SceneObject(obj_id, model_path)

        # camera
        cameras = {}
        for cam, cam_info in sc_cam_info.items():
            cameras[cam] = Camera(cam, self._cam2serial[cam], cam_info['intrinsic'], cam_info['extrinsics'], self._cam2serial[cam])
        
        self.current_scene_idx = self._total_scene.index(sc_name)
        self.current_scene_file = sc_name
        self.current_frame_file = frame_id
            
        return Scene(scene_dir=sc_path, 
                     hands=hands, 
                     objects=objects,
                     cameras=cameras,
                     frame_list=sc_frame_list,
                     current_frame=frame_id,
                     load_pcd=self.load_point_cloud)

    def get_progress(self):
        return "작업 폴더: {} [{}/{}]".format(self.current_scene_file, self.current_scene_idx+1, len(self._total_scene))
    
    def get_next_scene(self):
        scene_idx = self.current_scene_idx + 1
        if scene_idx > len(self._total_scene) -1:
            return None
        else:
            self.current_scene_idx = scene_idx    
            return self.get_scene_with_first_file()

    def get_previous_scene(self):
        scene_idx = self.current_scene_idx - 1
        if scene_idx < 0:
            return None
        else:
            self.current_scene_idx = scene_idx    
            return self.get_scene_with_first_file()
    
    def get_scene_with_first_file(self):
        sc_name = self._total_scene[self.current_scene_idx]
        frame_id = self._scene_frame[sc_name][0]
        return self.get_scene(sc_name, frame_id)
        
    def check_same_data(self, file_path):
        camera_dir = os.path.dirname(file_path)
        scene_dir = os.path.dirname(camera_dir)
        subject_dir = os.path.dirname(scene_dir)
        dataset_dir = os.path.dirname(subject_dir)
        return self._data_dir == dataset_dir

    @staticmethod
    def path_to_info(file_path):
        data_dir = os.path.dirname(file_path)
        camera_dir = os.path.dirname(data_dir)
        scene_dir = os.path.dirname(camera_dir)

        frame_id = int(Utils.get_file_name(file_path).split("_")[-1])
        camera_name = os.path.basename(camera_dir)
        sc_name = os.path.basename(scene_dir)
        
        return sc_name, camera_name, frame_id

    @staticmethod
    def load_point_cloud(pc_file):
        pcd = o3d.io.read_point_cloud(pc_file)
        if pcd is not None:
            print("[Info] Successfully read scene ")
            if not pcd.has_normals():
                pcd.estimate_normals()
            pcd.normalize_normals()
        else:
            print("[WARNING] Failed to read points")
        return pcd

class OurDataset(Dataset):
    _SERIALS = [
        '000390922112', # master
        '000480922112',
        '000056922112',
        '000210922112',
        '000363922112',
        '000375922112',
        '000355922112',
        # '000364922112', # ego_centric
    ]
    _CAMERAS = [
        "좌하단",
        "정하단",
        "정중단",
        "우하단",
        "우중단",
        "정상단",
        "좌중단",
        # "후상단",
    ]

    def __init__(self, data_root):
        # Total Data Statics
        self._data_dir = data_root
        self._calib_dir = os.path.join(self._data_dir, "calibration")
        self._model_dir = os.path.join(self._data_dir, "models", 'object')

        self._mano_shape_path = os.path.join(self._calib_dir, "mano-hand", '{}.json') # {}-subject-{:02d}.format(scan-date, id)
        self._cam_calib_path = os.path.join(self._calib_dir, 'camera', '{}.json') # {}.format(calib-date)

        self._cameras = self._CAMERAS
        self._cam2serial = {cam: self._SERIALS[i] for i , cam in enumerate(self._cameras)}
        self._master_cam = self._SERIALS[0]

        self._obj_path = os.path.join(self._model_dir, "obj_{:06d}.ply")
        
        self._total_scene = [] # list of scene, idx -> sc_name
        self._scene_path = {} # sc_name: scene dir
        self._scene_hand = {} # sc_name: scene hand info
        self._scene_object = {} # sc_name: scene object info
        self._scene_camera = {} # sc_name: scene camera info
        self._scene_frame = {} # sc_name: scene frames(master)
        
        for sc_dir in [os.path.join(self._data_dir, p) for p in os.listdir(self._data_dir)]:
            
            sc_name = os.path.basename(sc_dir)
            if sc_name in ['calibration', 'models']:
                continue
            meta_file = os.path.join(sc_dir, "meta.yml")
            if not os.path.isfile(meta_file):
                continue
            with open(meta_file, 'r') as f:
                meta = yaml.load(f, Loader=yaml.FullLoader)
            
            # points
            pcd_dir = os.path.join(sc_dir, self._master_cam, "pcd")
            if not os.path.isdir(pcd_dir):
                continue
            
            frame_list = [int(os.path.splitext(p)[0]) for p in os.listdir(pcd_dir)]
            frame_list.sort()
            self._scene_frame[sc_name] = frame_list
            
            # scene list
            self._total_scene.append(sc_name)
            
            # scene dir
            self._scene_path[sc_name] = sc_dir
            
            # hand
            shape = Utils.load_json_to_dic(self._mano_shape_path.format(meta['mano_calib']))
            if 'mano_sides' in meta.keys():
                mano_sides = meta['mano_sides']
            else:
                mano_sides = temp_side_info[int(sc_name)]
                meta.update({'mano_sides': mano_sides})
                Utils.save_dic_to_yaml(meta, meta_file)
            
            self._scene_hand[sc_name] = {side: shape[side] for side in mano_sides}
            # objects
            self._scene_object[sc_name] = {
                obj_id : self._obj_path.format(obj_id) for obj_id in meta['obj_ids']
            }

            # camera 
            cam_calib = Utils.load_json_to_dic(self._cam_calib_path.format(meta['cam_calib']))
            self._scene_camera[sc_name] = {}
            for cam in self._cameras:
                serial = self._cam2serial[cam]
                K = self.get_intrinsic_from_dic(cam_calib[serial])
                extr = self.get_extrinsic_from_dic(cam_calib[serial])
                extrinsic = np.eye(4)
                R = extr[:3, :3]
                t = extr[:3, 3] / 1e3 # convert to meter
                extrinsic[:3, :3] = R
                extrinsic[:3, 3] = t
                self._scene_camera[sc_name][cam] = {
                    "intrinsic": K,
                    "extrinsics": extrinsic
                }
            self.H = cam_calib[serial]["height"]
            self.W = cam_calib[serial]["width"]
            
        self._total_scene.sort()
        super().__init__()

    def get_intrinsic_from_dic(self, dic):
        return np.array(dic['K']).reshape(3, 3)
    def get_extrinsic_from_dic(self, dic):
        return np.array(dic["H"]).reshape(4, 4)
    @staticmethod
    def load_point_cloud(pc_file):
        pcd = o3d.io.read_point_cloud(pc_file)
        if pcd is not None:
            print("[Info] Successfully read scene ")
            if not pcd.has_normals():
                pcd.estimate_normals()
            pcd.normalize_normals()
        else:
            print("[WARNING] Failed to read points")
        return pcd

def load_dataset_from_file(file_path):
    data_dir = os.path.dirname(file_path)
    camera_dir = os.path.dirname(data_dir)
    scene_dir = os.path.dirname(camera_dir)
    dataset_dir = os.path.dirname(scene_dir)
    
    return OurDataset(dataset_dir)
    
class Scene:
    
    class Frame:
        class SingleView:
            def __init__(self, rgb, depth, pcd, frame_id):
                self.rgb = rgb
                self.depth = depth
                self.pcd = pcd
                self.frame_id = frame_id

        def __init__(self, scene_dir, frame_id, hands, objs, cams, load_pcd):
            self.scene_dir = scene_dir
            
            self.id = frame_id
            
            self.hands = hands
            self.objects = objs
            self.cameras = cams
            
            self.rgb_format = os.path.join(self.scene_dir, "{}", "rgb", "{:06d}.png".format(self.id)) 
            self.depth_format = os.path.join(self.scene_dir, "{}", "depth", "{:06d}.png".format(self.id))
            self.pcd_format = os.path.join(self.scene_dir, "{}", "pcd", "{:06d}.pcd".format(self.id))

            self._load_pcd_fn = load_pcd

            self.single_views = {}
            for cam_name, cam in cams.items():
                self.single_views[cam_name] = self.SingleView(rgb=self._load_rgb(cam_name),
                                                              depth=self._load_depth(cam_name),
                                                              pcd=self._load_pcd(cam_name),
                                                              frame_id=self.id)
            self.scene_pcd = self.get_pcd(list(self.cameras.keys()))
            self.active_cam = 'merge' # merge

        def _load_rgb(self, cam_name):
            folder = self.cameras[cam_name].folder
            rgb_path = self.rgb_format.format(folder)
            return cv2.imread(rgb_path)
        def _load_depth(self, cam_name):
            folder = self.cameras[cam_name].folder
            depth_path = self.depth_format.format(folder)
            depth_img = cv2.imread(depth_path, -1)
            depth_img = np.float32(depth_img) # mm
            return depth_img
        def _load_pcd(self, cam_name):
            folder = self.cameras[cam_name].folder
            pcd_path = self.pcd_format.format(folder)
            return self._load_pcd_fn(pcd_path)
        
        def get_rgb(self, cam_name):
            return self.single_views[cam_name].rgb
        
        def get_depth(self, cam_name):
            return self.single_views[cam_name].depth
            
        def get_pcd(self, cam_list):
            """get target points by cam list
            Args:
                cam_list (list): list of cam_name
            Returns:
                pcd
            """
            pcd = o3d.geometry.PointCloud()
            for cam_name in cam_list:
                pcd += self.single_views[cam_name].pcd
            return pcd

    def __init__(self, scene_dir, hands, objects, cameras, frame_list, current_frame, load_pcd):
        self._scene_dir = scene_dir
        sc_name = os.path.basename(scene_dir)
        self._label_dir = os.path.join(scene_dir, f'{sc_name}_labels')
        os.makedirs(self._label_dir, exist_ok=True)
        
        self._hands = hands
        self._objects = objects
        self._cameras = cameras
        
        self.total_frame = len(frame_list) # for master camera
        self._frame_list = frame_list
        self.frame_id = current_frame
        self._frame_idx = self._frame_list.index(self.frame_id)
        
        self._hand_label_format = os.path.join(self._label_dir, "hands_{:06d}.npz")
        self._object_label_format = os.path.join(self._label_dir, "objs_{:06d}.npz")

        self._load_point_cloud = load_pcd

        self._label = None
        self._previous_label = None
        self._obj_label = None
        self._obj_previous_label = None

    def _load_frame(self):
        try:
            self.current_frame = Scene.Frame(scene_dir=self._scene_dir,
                                             frame_id=self.frame_id,
                                             hands=self._hands,
                                             objs=self._objects,
                                             cams=self._cameras,
                                             load_pcd=self._load_point_cloud)
        except Exception as e:
            print("Fail to load Frame", e)
            self.current_frame = None
    
    def get_current_frame(self):
        try:
            self._load_frame()
        except:
            print("Fail to get Frame")
        return self.current_frame
    
    def moveto_next_frame(self):
        frame_idx = self._frame_idx + 1
        if frame_idx > self.total_frame - 1:
            return False
        else:
            self._frame_idx = frame_idx
            self.frame_id = self._frame_list[frame_idx]
            return True
    def moveto_previous_frame(self):
        frame_idx = self._frame_idx - 1
        if frame_idx < 0:
            return False
        else:
            self._frame_idx = frame_idx
            self.frame_id = self._frame_list[frame_idx]
            return True
    def get_progress(self):
        return "현재 진행률: {} [{}/{}]".format(self.frame_id, self._frame_idx+1, self.total_frame)
    
    def save_label(self):
        # object label
        obj_label = {}
        for obj_id, obj in self._objects.items():
            obj_label[str(obj_id)] = obj.get_transform()
        np.savez(self._obj_label_path, **obj_label)
        self._obj_label = obj_label
        # get current hand label
        label = {}
        for side, hand_model in self._hands.items():
            h_state = hand_model.get_state()
            for k, v in h_state.items():
                label['{}_{}'.format(side, k)] = v
        np.savez(self._hand_label_path, **label)
        self._label = label

    def load_label(self):
        # object label
        if self._obj_label is not None:
            self._obj_previous_label = self._obj_label.copy()
        try:
            self._obj_label = dict(np.load(self._obj_label_path))
            
            for obj_id, label in self._obj_label.items():
                self._objects[int(obj_id)].set_transform(label)
        except:
            print("Fail to load object Label -> Load previous Label")
            try:
                for obj_id, label in self._obj_previous_label.items():
                    self._objects[int(obj_id)].set_transform(label)
            except:
                print("Fail to load previous Label -> Reset Label")
                for obj in self._objects.values():
                    obj.reset()
        #----- hand label
        # default is previous frame
        if self._label is not None:
            self._previous_label = self._label.copy()
        # first previous saved label
        try:
            self._label = dict(np.load(self._hand_label_path))
            hand_states = {}
            for k, v in self._label.items():
                side = k.split('_')[0]
                hand_states.setdefault(side, {})
                param = k.replace(side + "_", "")
                hand_states[side][param] = v
            for side, hand_model in self._hands.items():
                hand_model.set_state(hand_states[side])
            return True
        except:
            print("Fail to load previous Label -> Try to load AI label")
        
        # second ai label
        # for cam_name, cam in self._cameras.items():
        try:
            ai_label_path = os.path.join(self._scene_dir, "hand_mocap", "{:06d}.json".format(self.frame_id))
            # all_pred_path = os.path.join(self._scene_dir, cam.folder, "hand_mocap", "{:06d}_prediction_result.pkl".format(self.frame_id))
            ai_label = Utils.load_json_to_dic(ai_label_path)
            # all_pred = Utils.load_pickle(all_pred_path)['pred_output_list'][0]
            # extr = cam.extrinsics
            # R = extr[:3, :3]
            # R_inv = np.linalg.inv(R)
            for side, hand_model in self._hands.items():
                val = ai_label[side]
                mano_pose = np.array(val['mano_pose']).reshape(-1)
                wrist_pos = np.array(val['wrist_pos'])
                wrist_ori = np.array(val['wrist_ori'])
                mano_pose = np.concatenate((wrist_ori, mano_pose))
                
                hand_model.set_state(
                    {'pose_param': mano_pose,
                     'root_trans': wrist_pos})
                
                hand_model.set_root_position(wrist_pos)
            print("Success to Load AI Label")
            return True
        except:
            print("Fail to load AI Label")
        hand_states = {}
        if self._previous_label is not None:
            self._label = self._previous_label.copy()
            for k, v in self._label.items():
                side = k.split('_')[0]
                hand_states.setdefault(side, {})
                param = k.replace(side + "_", "")
                hand_states[side][param] = v
            for side, hand_model in self._hands.items():
                hand_model.set_state(hand_states[side])
            return True
        else:
            for hand_model in self._hands.values():
                hand_model.reset()
            return False
    
    def _load_hand_label(self, npz_file):
        try:
            label = dict(np.load(npz_file))
            hand_states = {}
            for k, v in label.items():
                side = k.split('_')[0]
                hand_states.setdefault(side, {})
                param = k.replace(side + "_", "")
                hand_states[side][param] = v
            for side, hand_model in self._hands.items():
                hand_model.set_state(hand_states[side])
            return True
        except:
            return False
    def _load_obj_label(self, npz_file):
        try:
            obj_label = dict(np.load(npz_file))
            for obj_id, label in obj_label.items():
                self._objects[int(obj_id)].load_label(label)
            return True
        except:
            return False
    def load_previous_label(self):
        if (self._previous_label is None) or (self._obj_previous_label is None):
            return False
        try:
            hand_states = {}
            for k, v in self._previous_label.items():
                side = k.split('_')[0]
                hand_states.setdefault(side, {})
                param = k.replace(side + "_", "")
                hand_states[side][param] = v
            for side, hand_model in self._hands.items():
                hand_model.set_state(hand_states[side], only_pose=True)
            for obj_id, label in self._obj_previous_label.items():
                self._objects[int(obj_id)].load_label(label)
            return True
        except:
            return False

    def save_json(self):
        json_path = os.path.join(self._scene_dir, self._json_format.format(self.frame_id))
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
            json_label['hand_mask_info'][side] ={
                "hand_type": str(side),
                "hand_position": hand_model.get_hand_pose(),
                "hand_shape": hand_model.get_hand_shape()
            }
        with open(json_path, 'w') as f:
            json.dump(json_label, f, sort_keys=True, indent=4)

    @property
    def _frame_path(self):
        return os.path.join(self._scene_dir, self._data_format.format(self.frame_id))
    @property
    def _hand_label_path(self):
        return os.path.join(self._scene_dir, self._hand_label_format.format(self.frame_id))
    @property
    def _obj_label_path(self):
        return os.path.join(self._scene_dir, self._object_label_format.format(self.frame_id))
    
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
        self.obj_material.base_color = [1, 1, 1, 1-self.obj_transparency]
        self.obj_material.shader = Settings.SHADER_LIT_TRANS

        self.active_obj_material = rendering.MaterialRecord()
        self.active_obj_material.base_color = [0.3, 0.9, 0.3, 1-self.obj_transparency]
        self.active_obj_material.shader = Settings.SHADER_LIT_TRANS

        # mesh
        self.nonactive_mesh_material = rendering.MaterialRecord()
        self.nonactive_mesh_material.base_color = [0.5, 0.5, 0.5, 1.0-self.hand_transparency]
        self.nonactive_mesh_material.shader = Settings.SHADER_LIT_TRANS
        
        # activate material
        self.active_mesh_material = rendering.MaterialRecord()
        self.active_mesh_material.base_color = [1.0, 0, 0, 1.0-self.hand_transparency]
        self.active_mesh_material.shader = self.SHADER_LIT_TRANS

        self.active_pcd_material = rendering.MaterialRecord()
        self.active_pcd_material.base_color = [0.3, 0.9, 0.3, 1]
        self.active_pcd_material.shader = Settings.SHADER_UNLIT
        self.active_pcd_material.point_size = 2*self.scene_material.point_size

        self.inlier_sphere_material = rendering.MaterialRecord()
        self.inlier_sphere_material.base_color = [0.6, 0.1, 0.4, 0.5]
        self.inlier_sphere_material.shader = Settings.SHADER_UNLIT

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
            geo = obj.get_mesh()
            geo = copy.deepcopy(geo)
            geo.paint_uniform_color(color)
            self.render.scene.add_geometry(obj_name, geo, self.obj_mtl)
    
    def add_hands(self, hands):
        for side, hand in hands.items():
            hand_geo = hand.get_geometry()
            geo = hand_geo['mesh']
            tmp_geo = copy.deepcopy(geo)
            if side=='right':
                tmp_geo.paint_uniform_color([1, 1, 0])
            else:    
                tmp_geo.paint_uniform_color([0, 0, 1])
            self.render.scene.add_geometry(side, tmp_geo, self.hand_mtl)

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
    def __init__(self, width, height, logger):
        self.logger = logger
        self.logger.info("Intialize AppWindow")
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
        self._control_joint_name = "control_joint"
        self._hand_geometry_names = [
            self._right_hand_mesh_name,
            self._right_hand_joint_name,
            self._right_hand_link_name,
            self._left_hand_mesh_name,
            self._left_hand_joint_name,
            self._left_hand_link_name,
            self._target_joint_name,
            self._target_link_name,
            self._active_joint_name,
            self._control_joint_name
        ]

        
        self._hand_mesh_name = "{}_hand_mesh"
        self._active_mesh_name = "{}_active_mesh"
        self._active_pcd_name = "active_scene"
        self._nonactive_pcd_name = "annotation_scene"

        self._inlier_sphere_name = 'inlier_sphere'

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
        self.hand_coord_labels = []
        
        self._view_rock = False
        self._depth_image = None
        self.prev_ms_x, self.prev_ms_y = None, None
        self._inlier_points = []
        self._inlier_radius = 0.01

        self._objects = None
        self._active_object = None
        self._object_names = []
        self._active_object_idx = -1
        self._cur_diff_mask = None

        self._camera_idx = -1
        self._cam_name_list = []
        self._depth_diff_list = []
        self.scale_factor = None
        self.reset_flat = False
        self.joint_back = False
        self._move_root = False
        self._guide_changed = False
        self._active_type = "hand"
        self.obj_coord_labels = []
        
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
        self._init_joint_mask_layout()
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
        self._on_scene_point_size(1) # set default size to 1
        # self._on_point_transparency(0)
        self._on_object_transparency(0)
        self._on_hand_transparency(0.2)
        self._on_hand_point_size(10) # set default size to 10
        self._on_hand_line_size(2) # set default size to 2
        self._on_responsiveness(5) # set default responsiveness to 5
        # self._on_inlier_radius(1)
        
        self._scene.set_on_mouse(self._on_mouse)
        self._scene.set_on_key(self._on_key)
        
        self.window.set_on_tick_event(self._on_tick)
        self._log.text = "\t라벨링 대상 파일을 선택하세요."
        self.window.set_needs_layout()
        self.logger.info("End Intialize AppWindow")

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
        center = np.array([0, 0, 1])
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
        if self._active_type=='hand' or self._active_type=='mesh':
            eye_on = self._active_hand.get_control_position()
        else:
            eye_on = self._active_object.get_transform()[:3, 3]
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
        self.logger.debug('_init_fileeidt_layout')
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
        self.logger.debug('_on_filedlg_button')
        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "파일 선택",
                                self.window.theme)
        filedlg.add_filter(".pcd", "포인트 클라우드")
        filedlg.set_on_cancel(self._on_filedlg_cancel)
        filedlg.set_on_done(self._on_filedlg_done)
        #TODO:
        if os.name=='nt' and os.getlogin()=='GIST':
            filedlg.set_path('D:\OccludedObjectDataset\data4')
        self.window.show_dialog(filedlg)
    def _on_filedlg_cancel(self):
        self.logger.debug('_on_filedlg_cancel')
        self.window.close_dialog()
    def _on_filedlg_done(self, file_path):
        self.logger.debug('_on_filedlg_done')
        """Load file -> Scene

        Args:
            file_path (_type_): _description_
        """
        self._fileedit.text_value = file_path
        try:
            # if already initialized
            if self.dataset is None:
                self.dataset = load_dataset_from_file(file_path)
                self.H, self.W = self.dataset.H, self.dataset.W
                self.hl_renderer = HeadlessRenderer(self.W, self.H)
            else:
                if self.dataset.check_same_data(file_path):
                    pass
                else:
                    del self.dataset
                    
                    self.dataset = load_dataset_from_file(file_path)
            if self.annotation_scene is None:
                self.annotation_scene = self.dataset.get_scene_from_file(file_path)
            else:
                del self.annotation_scene
                self.annotation_scene = self.dataset.get_scene_from_file(file_path)
            self._init_cam_name()
            self._load_scene()
            self.window.close_dialog()
            self._log.text = "\t 라벨링 대상 파일을 불러왔습니다."
        except Exception as e:
            print(e)
            self._on_error("잘못된 경로가 입력되었습니다. (error at _on_filedlg_done)")
            self._log.text = "\t 올바른 파일 경로를 선택하세요."
    
    # scene edit 편의 기능
    def _init_viewctrl_layout(self):
        self.logger.debug('_init_viewctrl_layout')
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

        self._error_box = gui.Checkbox("자동 에러율 계산")
        viewctrl_layout.add_child(self._error_box)
        self._error_box.checked = False

        self._down_sample_pcd = gui.Checkbox("손 중심 랜더링")
        viewctrl_layout.add_child(self._down_sample_pcd)
        self._down_sample_pcd.set_on_checked(self._on_down_sample_pcd)
        self._down_sample_pcd.checked = False

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
        self._optimize_rate.double_value = 5
        self._optimize_rate.set_on_value_changed(self._on_optimize_rate)
        grid.add_child(gui.Label("최적화 속도"))
        grid.add_child(self._optimize_rate)
        
        self._auto_save_interval = gui.Slider(gui.Slider.INT)
        self._auto_save_interval.set_limits(1, 20) # 1-> 1e-3
        self._auto_save_interval.double_value = 5
        self._auto_save_interval.set_on_value_changed(self._on_auto_save_interval)
        grid.add_child(gui.Label("자동 저장 간격"))
        grid.add_child(self._auto_save_interval)

        # self._inlier_radius_ui = gui.Slider(gui.Slider.INT)
        # self._inlier_radius_ui.set_limits(1, 10)
        # self._inlier_radius_ui.set_on_value_changed(self._on_inlier_radius)
        # grid.add_child(gui.Label("선택 반지름 크기"))
        # grid.add_child(self._inlier_radius_ui)
        
        viewctrl_layout.add_child(grid)
        
        self._settings_panel.add_child(viewctrl_layout)
    def _on_show_axes(self, show):
        self.logger.debug('_on_show_axes')
        self.settings.show_axes = show
        self._scene.scene.show_axes(self.settings.show_axes)
    def _on_show_hand(self, show):
        self.logger.debug('_on_show_hand')
        if self._active_hand is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_show_hand)")
            self._show_hands.checked = not show
            return
        self.settings.show_hand = show
        self._update_hand_layer()
    def _on_show_object(self, show):
        self.logger.debug('_on_show_object')
        if self._objects is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_show_object)")
            self._show_objects.checked = not show
            return
        self.settings.show_objects = show
        self._update_object_layer()
    def _on_show_pcd(self, show):
        self.logger.debug('_on_show_pcd')
        if self._pcd is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_show_pcd)")
            self._show_objects.checked = not show
            return
        self.settings.show_pcd = show
        self._update_pcd_layer()
    def _on_show_coord_frame(self, show):
        self.logger.debug('_on_show_coord_frame')
        self.settings.show_coord_frame = show
        if show:
            self._add_coord_frame("world_coord_frame")
        else:
            self._scene.scene.remove_geometry("world_coord_frame")
            for label in self.coord_labels:
                self._scene.remove_3d_label(label)
            self.coord_labels = []
    def _on_scene_point_size(self, size):
        self.logger.debug('_on_scene_point_size')
        self._log.text = "\t 포인트 사이즈 값을 변경합니다."
        self.window.set_needs_layout()
        self._last_change = time.time()
        self.settings.scene_material.point_size = int(size)
        self.settings.active_mesh_material.point_size = int(size)*2
        if self._check_geometry(self._scene_name):
            self._set_geometry_material(self._scene_name, self.settings.scene_material)
        self._scene_point_size.double_value = size
    def _on_point_transparency(self, transparency):
        self.logger.debug('_on_point_transparency')
        self._log.text = "\t 투명도 값을 변경합니다."
        self.window.set_needs_layout()
        self._last_change = time.time()
        self.settings.point_transparency = transparency
        self.settings.scene_material.base_color = [1.0, 1.0, 1.0, 1.0-transparency]
        self._point_transparency.double_value = transparency
        if self._check_geometry(self._scene_name):
            self._set_geometry_material(self._scene_name, self.settings.scene_material)
    def _on_object_transparency(self, transparency):
        self.logger.debug('_on_object_transparency')
        self._log.text = "\t 투명도 값을 변경합니다."
        self.window.set_needs_layout()
        self._last_change = time.time()
        self.settings.obj_transparency = transparency
        self.settings.obj_material.base_color = [1, 1, 1, 1 - transparency]
        self.settings.active_obj_material.base_color = [0.3, 0.9, 0.3, 1 - transparency]
        self._object_transparency.double_value = transparency
        if self._objects is not None:
            for obj_id, _ in self._objects.items():
                obj_name = "obj_{}".format(obj_id)
                if self._active_type=='object':
                    self._set_geometry_material(obj_name, self.settings.active_obj_material)
                else:
                    self._set_geometry_material(obj_name, self.settings.obj_material)
    def _on_inlier_radius(self, radius):
        self.logger.debug('_on_inlier_radius')
        self._log.text = "\t 매쉬 선택 반지름 변경."
        self.window.set_needs_layout()
        self._last_change = time.time()
        self._inlier_radius = radius*0.01
        self._inlier_radius_ui.double_value = radius
        self._inlier_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self._inlier_radius)
        
    def _on_hand_transparency(self, transparency):
        self.logger.debug('_on_hand_transparency')
        self._log.text = "\t 투명도 값을 변경합니다."
        self.window.set_needs_layout()
        self._last_change = time.time()
        self.settings.hand_transparency = transparency
        self.settings.hand_mesh_material.base_color = [0.8, 0.8, 0.8, 1.0-transparency]
        self.settings.active_hand_mesh_material.base_color = [0.0, 1.0, 0.0, 1.0-transparency]
        self.settings.nonactive_mesh_material.base_color = [0.5, 0.5, 0.5, 1.0-transparency]
        self.settings.active_mesh_material.base_color = [1.0, 0, 0, 1.0-transparency]
        self._hand_transparency.double_value = transparency
        if self._active_hand is not None:
            if self._active_type=='mesh':
                for side in ['right', 'left']:
                    self._set_geometry_material(self._hand_mesh_name.format(side), self.settings.nonactive_mesh_material)
                    self._set_geometry_material(self._active_mesh_name.format(side), self.settings.nonactive_mesh_material)
            else:
                active_side = self._active_hand.side
                if active_side == 'right':
                    self._set_geometry_material(self._right_hand_mesh_name, self.settings.active_hand_mesh_material)
                    self._set_geometry_material(self._left_hand_mesh_name, self.settings.hand_mesh_material)
                else:
                    self._set_geometry_material(self._right_hand_mesh_name, self.settings.hand_mesh_material)
                    self._set_geometry_material(self._left_hand_mesh_name, self.settings.active_hand_mesh_material)
    def _on_hand_point_size(self, size):
        self.logger.debug('_on_hand_point_size')
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
        self.logger.debug('_on_hand_line_size')
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
        self.logger.debug('_on_responsiveness')
        self._log.text = "\t 라벨링 민감도 값을 변경합니다."
        self.window.set_needs_layout()
        self._last_change = time.time()
        
        self.dist = 0.0004 * responsiveness
        self.deg = 0.2 * responsiveness
        self._responsiveness.double_value = responsiveness
    def _on_optimize_rate(self, optimize_rate):
        self.logger.debug('_on_optimize_rate')
        if self._active_hand is None: # shsh
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_optimize_rate)")
            return
        self._log.text = "\t 자동 정렬 민감도 값을 변경합니다."
        self._active_hand.set_learning_rate(optimize_rate*1e-3)
        self.window.set_needs_layout()
        self._last_change = time.time()
        self._optimize_rate.double_value = optimize_rate
    def _on_auto_save_interval(self, interval):
        self.logger.debug('_on_auto_save_interval')
        self._log.text = "\t 자동 저장 간격을 변경합니다."
        self.window.set_needs_layout()
        self._auto_save_interval.double_value = interval
    def _on_down_sample_pcd(self, checked):
        if checked:
            if self._active_hand is None:
                self._on_error("손이 없습니다.")
                self._down_sample_pcd.checked = False
                return
        self._update_pcd_layer()
    # labeling stage edit
    def _init_stageedit_layout(self):
        self.logger.debug('_init_stageedit_layout')
        em = self.window.theme.font_size
        stageedit_layout = gui.CollapsableVert("라벨링 단계 선택 (F1, F2, F3, F4)", 0.33*em,
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

        button = gui.Button(LabelingMode.OBJECT)
        button.set_on_clicked(self._on_object_mode)
        stageedit_layout.add_child(button)

        # button = gui.Button(LabelingMode.MESH)
        # button.set_on_clicked(self._on_object_mode)
        # stageedit_layout.add_child(button)

        self._settings_panel.add_child(stageedit_layout)
    def _on_static_mode(self):
        self.logger.debug('_on_static_mode')
        self._convert_mode(LabelingMode.STATIC)
    def _on_optimize_mode(self):
        self.logger.debug('_on_optimize_mode')
        self._convert_mode(LabelingMode.OPTIMIZE)
    def _on_object_mode(self):
        self.logger.debug('_on_object_mode')
        self._convert_mode(LabelingMode.OBJECT)
    def _on_mesh_mode(self):
        self.logger.debug('_on_mesh_mode')
        self._convert_mode(LabelingMode.MESH)
    def _convert_mode(self, labeling_mode):
        self.logger.debug('_convert_mode_{}'.format(labeling_mode))
        if not self._check_annotation_scene():
            return
        self._labeling_mode = labeling_mode
        self._current_stage_str.text = "현재 상태: {}".format(self._labeling_mode)
        if self._labeling_mode==LabelingMode.OBJECT:
            self._active_type = 'object'
            self._active_object_idx = 0
            self._deactivate_hand()
        # elif self._labeling_mode==LabelingMode.MESH:
        #     self._active_type = 'mesh'
        #     self._active_object_idx = -1
        #     self._active_object = None
        #     self._deactivate_hand()
        #     self._active_hand.set_optimize_state(labeling_mode)
        else:
            self._active_type = 'hand'
            self._active_object_idx = -1
            self._active_object = None
            self._active_hand.set_optimize_state(labeling_mode)
        self._inlier_points = []
        for side, hm in self._hands.items():
            hm.active_faces = []
        self._view_rock = False
        for side in self._hands.keys():
            self._remove_geometry(self._active_mesh_name.format(side))

        self._update_pcd_layer()
        self._update_object_layer()
        self._update_hand_layer()
        
            
    # labeling hand edit
    def _init_handedit_layout(self):
        self.logger.debug('_init_handedit_layout')
        em = self.window.theme.font_size
        self._active_hand_idx = 0
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
        
        button = gui.Button("이전 관절 (PgDn)")
        button.horizontal_padding_em = 0.3
        button.vertical_padding_em = 0.3
        button.set_on_clicked(self._control_joint_down)
        grid.add_child(button)
        button = gui.Button("다음 관절 (PgUp)")
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
        self.logger.debug('_reset_current_hand')
        if self._active_hand is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _reset_current_hand)")
            return 
        self._active_hand.reset_pose()
        self._update_hand_layer()
    def _deactivate_hand(self):
        self.logger.debug('_deactivate_hand')
        if self._hands is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _convert_hand)")
            return 
        if self._check_geometry(self._right_hand_mesh_name):
            self._set_geometry_material(self._right_hand_mesh_name, self.settings.hand_mesh_material)
        if self._check_geometry(self._left_hand_mesh_name):
            self._set_geometry_material(self._left_hand_mesh_name, self.settings.hand_mesh_material)
        
        if self._check_geometry(self._target_joint_name):
            self._remove_geometry(self._target_joint_name)
        if self._check_geometry(self._target_link_name):
            self._remove_geometry(self._target_link_name)
        if self._check_geometry(self._active_joint_name):
            self._remove_geometry(self._active_joint_name)
        if self._check_geometry(self._control_joint_name):
            self._remove_geometry(self._control_joint_name)
    def _convert_hand(self):
        self.logger.debug('_convert_hand')
        if self._hands is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _convert_hand)")
            return 

        self._active_hand_idx = (self._active_hand_idx+1) % len(self._hands.keys())
        active_side = self._hand_names[self._active_hand_idx]
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
        if self._down_sample_pcd.checked:
            self._update_pcd_layer()

    # convert finger
    def _convert_to_root(self):
        self.logger.debug('_convert_to_root')
        self._convert_finger('root')
    def _convert_to_thumb(self):
        self.logger.debug('_convert_to_thumb')
        self._convert_finger('thumb')
    def _convert_to_fore(self):
        self.logger.debug('_convert_to_fore')
        self._convert_finger('fore')
    def _convert_to_middle(self):
        self.logger.debug('_convert_to_middle')
        self._convert_finger('middle')
    def _convert_to_ring(self):
        self.logger.debug('_convert_to_ring')
        self._convert_finger('ring')
    def _convert_to_little(self):
        self.logger.debug('_convert_to_little')
        self._convert_finger('little')
    def _convert_finger(self, name):
        self.logger.debug('_convert_finger')
        if self._active_hand is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _convert_finger)")
            return
        self._active_hand.set_optimize_target(name)
        self._update_target_hand()
        self._update_current_hand_str()
    def _control_joint_up(self):
        self.logger.debug('_control_joint_up')
        if self._active_hand is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _control_joint_up)")
            return
        ctrl_idx = self._active_hand.control_idx + 1
        self._active_hand.set_control_joint(ctrl_idx)
        self._update_joint_mask()
        self._update_target_hand()
        self._update_current_hand_str()
    def _control_joint_down(self):
        self.logger.debug('_control_joint_down')
        if self._active_hand is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _control_joint_down)")
            return
        ctrl_idx = self._active_hand.control_idx - 1
        self._active_hand.set_control_joint(ctrl_idx)
        self._update_joint_mask()
        self._update_target_hand()
        self._update_current_hand_str()
    def _update_current_hand_str(self):
        self.logger.debug('_update_current_hand_str')
        self._current_hand_str.text = "현재 대상: {}".format(self._active_hand.get_control_joint_name())

    # scene control
    def _init_scene_control_layout(self):
        self.logger.debug('_init_scene_control_layout')
        em = self.window.theme.font_size
        scene_control_layout = gui.CollapsableVert("작업 파일 리스트", 0.33 * em,
                                                   gui.Margins(0.25 * em, 0, 0, 0))
        scene_control_layout.set_is_open(True)
        
        v = gui.Vert(0.4 * em)
        self._current_scene_pg = gui.Label("작업 폴더: 준비중 [00/00]")
        v.add_child(self._current_scene_pg)
        h = gui.Horiz(0.4 * em)
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
        v.add_child(h)
        scene_control_layout.add_child(v)
        
        v = gui.Vert(0.4 * em)
        self._current_file_pg = gui.Label("현재 파일: 준비중 [00/00]")
        v.add_child(self._current_file_pg)
        h = gui.Horiz(0.4 * em)
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
        v.add_child(h)
        scene_control_layout.add_child(v)
        self._settings_panel.add_child(scene_control_layout)
    def _check_changes(self):
        self.logger.debug('_check_changes')
        if self._annotation_changed:
            self._on_error("라벨링 결과를 저장하지 않았습니다. 저장하지 않고 넘어가려면 버튼을 다시 눌러주세요.")
            self._annotation_changed = False
            return True
        else:
            return False
    def _load_scene(self):
        self.logger.debug('_load_scene')
        self._frame = self.annotation_scene.get_current_frame()
        try:
            ret = self.annotation_scene.load_label()
        except:
            self._log.text = "\t 저장된 라벨이 없습니다."
            pass
        if not ret:
            hands = self._frame.hands
            objs = self._frame.objects
            pcd = self._frame.scene_pcd
            center = pcd.get_center()
            for s, hand_model in hands.items():
                hand_model.set_root_position(center)
            for obj_name, obj in objs.items():
                obj.set_position(center)

        self._update_progress_str()
        self._init_pcd_layer()
        self._init_obj_layer()
        self._init_hand_layer()
        self._on_change_camera_merge()
        self._update_valid_error(calculate=True)
        self._update_pc_error()
    def _update_progress_str(self):
        self.logger.debug('_update_progress_str')
        self._current_file_pg.text = self.annotation_scene.get_progress()
        self._current_scene_pg.text = self.dataset.get_progress()
    def _on_previous_frame(self):
        self.logger.debug('_on_previous_frame')
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
        self.logger.debug('_on_next_frame')
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
        self.logger.debug('_on_previous_scene')
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
        self.logger.debug('_on_next_scene')
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
        self.logger.debug('_init_label_control_layout')
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
        self.logger.debug('_on_save_label')
        self._log.text = "\t라벨링 결과를 저장 중입니다."
        self.window.set_needs_layout()
        
        if not self._check_annotation_scene():
            return
        
        self.annotation_scene.save_label()
        self._update_diff_viewer()
        self._last_saved = time.time()
        self._log.text = "\t라벨링 결과를 저장했습니다."
        self._annotation_changed = False
    def _on_load_previous_label(self):
        self.logger.debug('_on_load_previous_label')
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
        self.logger.debug('_init_preset_layout')
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
        self._validation_panel.add_child(preset_layout)
    
    def _on_load_preset(self):
        self.logger.debug('_on_load_preset')
        if not self._check_annotation_scene():
            return
        name = self.preset_name.text_value
        try:
            pose = self._activate_template.get_template2pose(name)
            self._active_hand.set_joint_pose(pose)
            self._update_hand_layer()
        except:
            self._on_error("프리셋 이름을 확인하세요. (error at _on_load_preset)")
    def _on_save_preset(self):
        self.logger.debug('_on_save_preset')
        if not self._check_annotation_scene():
            return
        name = self.preset_name.text_value
        pose = self._active_hand.get_joint_pose()
        self._activate_template.save_pose2template(name, pose)
        self.preset_list.set_items(self._activate_template.get_template_list())
    def _on_change_preset_select(self, preset_name, double):
        self.logger.debug('_on_change_preset_select')
        self.preset_name.text_value = preset_name
        if double:
            self._on_load_preset()

    def _init_joint_mask_layout(self):
        self.logger.debug('_init_joint_mask_layout')
        em = self.window.theme.font_size
        joint_mask_layout = gui.CollapsableVert("활성화된 관절 시각화", 0.33 * em,
                                                gui.Margins(0.25 * em, 0, 0, 0))
        self._joint_mask_proxy = gui.WidgetProxy()
        self._joint_mask_proxy.set_widget(gui.ImageWidget())
        joint_mask_layout.add_child(self._joint_mask_proxy)
        self._validation_panel.add_child(joint_mask_layout)

    def _update_joint_mask(self):
        img = self._active_hand.get_joint_mask()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = o3d.geometry.Image(img)
        self._joint_mask_proxy.set_widget(gui.ImageWidget(img))

    # image viewer
    def _init_image_view_layout(self):
        em = self.window.theme.font_size
        self.logger.debug('_init_image_view_layout')
        self._rgb_proxy = gui.WidgetProxy()
        self._rgb_proxy.set_widget(gui.ImageWidget())
        self._images_panel.add_child(self._rgb_proxy)
        self._images_panel.set_is_open(False)
        self._diff_proxy = gui.WidgetProxy()
        self._diff_proxy.set_widget(gui.ImageWidget())
        self._images_panel.add_child(self._diff_proxy)
        self._images_panel.set_is_open(False)
        h = gui.Horiz(0.4 * em)
        h.add_child(gui.Label("마스크 이미지 선택:"))        
        self._mask_mode = gui.Combobox()
        
        self._mask_mode.add_item(MaskMode.RGB_ALL)
        self._mask_mode.add_item(MaskMode.RGB_RIGHT)
        self._mask_mode.add_item(MaskMode.RGB_LEFT)
        self._mask_mode.add_item(MaskMode.RGB_OBJECT)
        self._mask_mode.add_item(MaskMode.MASK_ALL)
        self._mask_mode.add_item(MaskMode.MASK_RIGHT)
        self._mask_mode.add_item(MaskMode.MASK_LEFT)
        self._mask_mode.add_item(MaskMode.MASK_OBJECT)
        self._mask_mode.set_on_selection_changed(self._on_update_mode)
        h.add_child(self._mask_mode)
        
        self._rgb_transparency = gui.Slider(gui.Slider.DOUBLE)
        self._rgb_transparency.set_limits(0, 1)
        self._rgb_transparency.double_value = 0.5
        self._last_rgb_transparency = 0.5
        self._rgb_transparency.set_on_value_changed(self._on_update_rgb_transparency)
        
        self._images_panel.add_child(h)
        self._images_panel.add_child(self._rgb_transparency)

    
    def _init_show_error_layout(self):
        self.logger.debug('_init_show_error_layout')
        em = self.window.theme.font_size
        show_error_layout = gui.CollapsableVert("카메라 시점 조정", 0.33 * em,
                                                   gui.Margins(0.25 * em, 0, 0, 0))
        show_error_layout.set_is_open(True)
        
        self._view_error_layout_list = []
        
        for i in range(7):
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
        h.add_child(gui.Label("|\n|\n|"))
        
        v = gui.Vert(0)
        right_error_txt = gui.Label("준비 안됨")
        v.add_child(right_error_txt)
        left_error_txt = gui.Label("준비 안됨")
        v.add_child(left_error_txt)
        obj_error_txt = gui.Label("준비 안됨")
        v.add_child(obj_error_txt)
        h.add_child(v)
        
        button = gui.Button("에러 업데이트")
        button.vertical_padding_em = 0.1
        button.set_on_clicked(self._on_update_error)
        h.add_child(button)

        show_error_layout.add_child(h)
        self._total_error_txt = (right_error_txt, left_error_txt, obj_error_txt)
        h = gui.Horiz(0.4 * em)
        self._total_pc_error_txt = gui.Label("포인트 에러: 준비 안됨")
        h.add_child(self._total_pc_error_txt)
        
        button = gui.Button("에러 업데이트")
        button.vertical_padding_em = 0.1
        button.set_on_clicked(self._on_update_pc_error)
        h.add_child(button)
        show_error_layout.add_child(h)
        
        self._activate_cam_txt = gui.Label("현재 활성화된 카메라: 없음")
        show_error_layout.add_child(self._activate_cam_txt)

        self._validation_panel.add_child(show_error_layout)
    def _on_change_camera_0(self):
        self.logger.debug('_on_change_camera_0')
        if not self._check_annotation_scene():
            return
        self._camera_idx = 0
        self._frame.active_cam=self._view_error_layout_list[self._camera_idx][0].text
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_error_layout_list[self._camera_idx][0].text)
    def _on_change_camera_1(self):
        self.logger.debug('_on_change_camera_1')
        if not self._check_annotation_scene():
            return
        self._camera_idx = 1
        self._frame.active_cam=self._view_error_layout_list[self._camera_idx][0].text
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_error_layout_list[self._camera_idx][0].text)
    def _on_change_camera_2(self):
        self.logger.debug('_on_change_camera_2')
        if not self._check_annotation_scene():
            return
        self._camera_idx = 2
        self._frame.active_cam=self._view_error_layout_list[self._camera_idx][0].text
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_error_layout_list[self._camera_idx][0].text)
    def _on_change_camera_3(self):
        self.logger.debug('_on_change_camera_3')
        if not self._check_annotation_scene():
            return
        self._camera_idx = 3
        self._frame.active_cam=self._view_error_layout_list[self._camera_idx][0].text
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_error_layout_list[self._camera_idx][0].text)
    def _on_change_camera_4(self):
        self.logger.debug('_on_change_camera_4')
        if not self._check_annotation_scene():
            return
        self._camera_idx = 4
        self._frame.active_cam=self._view_error_layout_list[self._camera_idx][0].text
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_error_layout_list[self._camera_idx][0].text)
    def _on_change_camera_5(self):
        self.logger.debug('_on_change_camera_5')
        if not self._check_annotation_scene():
            return
        self._camera_idx = 5
        self._frame.active_cam=self._view_error_layout_list[self._camera_idx][0].text
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_error_layout_list[self._camera_idx][0].text)
    def _on_change_camera_6(self):
        self.logger.debug('_on_change_camera_6')
        if not self._check_annotation_scene():
            return
        self._camera_idx = 6
        self._frame.active_cam=self._view_error_layout_list[self._camera_idx][0].text
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_error_layout_list[self._camera_idx][0].text)
    def _on_change_camera_7(self):
        self.logger.debug('_on_change_camera_7')
        if not self._check_annotation_scene():
            return
        self._camera_idx = 7
        self._frame.active_cam=self._view_error_layout_list[self._camera_idx][0].text
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_error_layout_list[self._camera_idx][0].text)
    def _on_change_camera_merge(self):
        self.logger.debug('_on_change_camera_merge')
        if not self._check_annotation_scene():
            return
        self._camera_idx = -1
        self._frame.active_cam="merge"
        for but, _, _, bbox in self._view_error_layout_list:
            bbox.checked = True
        self._update_pcd_layer()
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: 합쳐진 뷰"
    def _on_change_camera(self):
        self._reset_image_viewer()
        self._update_image_viewer()
        self._update_diff_viewer()
        self._on_active_camera_viewpoint()
    def _on_change_bbox(self, visible):
        self.logger.debug('_on_change_bbox')
        if not self._check_annotation_scene():
            return
        self._update_pcd_layer()
    def _get_activate_cam(self):
        self.logger.debug('_get_activate_cam')
        cam_list = []
        for but, _, _, bbox in self._view_error_layout_list:
            if bbox.checked:
                cam_list.append(but.text)
        return cam_list
    def _on_click_focus_0(self):
        self.logger.debug('_on_click_focus_0')
        self._on_focus(0)
    def _on_click_focus_1(self):
        self.logger.debug('_on_click_focus_1')
        self._on_focus(1)
    def _on_click_focus_2(self):
        self.logger.debug('_on_click_focus_2')
        self._on_focus(2)
    def _on_click_focus_3(self):
        self.logger.debug('_on_click_focus_3')
        self._on_focus(3)
    def _on_click_focus_4(self):
        self.logger.debug('_on_click_focus_4')
        self._on_focus(4)
    def _on_click_focus_5(self):
        self.logger.debug('_on_click_focus_5')
        self._on_focus(5)
    def _on_click_focus_6(self):
        self.logger.debug('_on_click_focus_6')
        self._on_focus(6)
    def _on_click_focus_7(self):
        self.logger.debug('_on_click_focus_7')
        self._on_focus(7)
    def _on_focus(self, idx):
        self.logger.debug('_on_focus')
        for i, (_, _, _, bbox) in enumerate(self._view_error_layout_list):
            if i == idx:
                bbox.checked = True
            else:
                bbox.checked = False
        self._update_pcd_layer()
    def _on_update_error(self):
        if not self._check_annotation_scene():
            return 
        self._update_valid_error(calculate=True)
        self._update_diff_viewer()
    def _on_update_pc_error(self):
        self._update_pc_error()
        
    def _init_cam_name(self):
        self.logger.debug('_init_cam_name')
        self._cam_name_list = list(self.annotation_scene._cameras.keys())
        for idx, (cam_button, _, _, _) in enumerate(self._view_error_layout_list):
            cam_button.text = self._cam_name_list[idx]
        self._diff_images = {cam_name: None for cam_name in self._cam_name_list}
    def _update_image_viewer(self):
        self.logger.debug('_update_image_viewer')
        if self._camera_idx == -1:
            self._rgb_proxy.set_widget(gui.ImageWidget())
            return
        current_cam = self._cam_name_list[self._camera_idx]
        rgb_img = self._frame.get_rgb(current_cam)
        self.rgb_img = rgb_img
        self.H, self.W, _ = rgb_img.shape
        self._rgb_proxy.set_widget(gui.ImageWidget(self._img_wrapper(self.rgb_img)))
    def _update_diff_viewer(self, only_transparency=False):
        self.logger.debug('_update_diff_viewer')
        if self._camera_idx == -1:
            self._diff_proxy.set_widget(gui.ImageWidget())
            return
        current_cam = self._cam_name_list[self._camera_idx]
        try:
            diff_img = self._generate_valid_image(current_cam, only_transparency)
        except:
            self._on_error('이미지 생성 실패')
            return
        self.diff_img = diff_img
        if diff_img is not None:
            self._diff_proxy.set_widget(gui.ImageWidget(self._img_wrapper(diff_img)))
        else:
            self._diff_proxy.set_widget(gui.ImageWidget())
    def _on_update_mode(self, text, idx):
        if not self._check_annotation_scene():
            return
        self._update_diff_viewer()
    def _on_update_rgb_transparency(self, value):
        if not self._check_annotation_scene():
            return
        if abs(value - self._last_rgb_transparency) > 0.01:
            self._last_rgb_transparency = value
            self._update_diff_viewer(only_transparency=True)
        
    def _update_valid_error(self, calculate=False):
        self.logger.debug('Start _update_valid_error')
        self._log.text = "\t라벨링 검증용 이미지를 생성 중입니다."
        self.window.set_needs_layout()   

        if calculate or (self._depth_diff_list==[]) or self._error_box.checked:
            self.logger.debug('\tset hl renderer')
            self.hl_renderer.reset()    
            self.hl_renderer.add_objects(self._objects, color=[1, 0, 0])
            self.hl_renderer.add_hands(self._hands) # right [0, 1, 0] left [0, 0, 1]


            # rendering depth for each camera
            self.logger.debug('\trendering depth for each camera')
            self._depth_diff_list = []
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
                # cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'depth_rendered.png'), depth_rendered)
                
                rgb_rendered = self.hl_renderer.render_rgb()
                rgb_rendered = np.array(rgb_rendered)
                # cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rgb_rendered.png'), rgb_rendered)

                # object mask
                object_mask = np.bitwise_and(rgb_rendered[:, :, 0]>10, np.bitwise_and(rgb_rendered[:, :, 1]<2, rgb_rendered[:, :, 2]<2))

                # only hand mask
                right_hand_mask = np.bitwise_and(rgb_rendered[:, :, 0]>10, np.bitwise_and(rgb_rendered[:, :, 1]>10, rgb_rendered[:, :, 2]>2))
                # right_hand_mask_vis = np.zeros_like(rgb_rendered)
                # right_hand_mask_vis[right_hand_mask] = 255
                # cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'right_hand_mask_vis.png'), right_hand_mask_vis)

                left_hand_mask = np.bitwise_and(rgb_rendered[:, :, 0]<2, np.bitwise_and(rgb_rendered[:, :, 1]<2, rgb_rendered[:, :, 2]>2))
                # left_hand_mask_vis = np.zeros_like(rgb_rendered)
                # left_hand_mask_vis[left_hand_mask] = 255
                # cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'left_hand_mask_vis.png'), left_hand_mask_vis)


                # set mask as rendered depth
                valid_mask = depth_rendered > 0

                # get captured image
                rgb_captured = self._frame.get_rgb(cam_name)
                depth_captured = self._frame.get_depth(cam_name)

                diff_vis = np.zeros_like(rgb_captured)
                diff_vis[right_hand_mask] = [255, 0, 0] #BGR 
                diff_vis[left_hand_mask] = [0, 255, 0] # BGR

                diff_vis[object_mask] = [0, 180, 180]

                hand_mask = np.bitwise_or(right_hand_mask, left_hand_mask)
                hand_object_mask = np.bitwise_or(object_mask, hand_mask)

                # calculate diff
                depth_diff = depth_captured - depth_rendered
                depth_diff = np.clip(depth_diff, -300, 300)
                depth_diff_abs = np.abs(np.copy(depth_diff))
                inlier_mask = depth_diff_abs < 300 # consider only 300mm under error

                # object
                object_valid_mask = valid_mask * object_mask * inlier_mask
                if np.sum(object_valid_mask) > 0:
                    
                    depth_diff_mean = np.sum(depth_diff_abs[object_valid_mask]) / np.sum(object_valid_mask)
                else:
                    depth_diff_mean = -1
                obj_diff_mean = copy.deepcopy(depth_diff_mean)
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
                
                self._depth_diff_list.append([r_diff_mean, l_diff_mean, obj_diff_mean])
                error_layout[1].text = "오른손: {:.2f}".format(r_diff_mean)
                error_layout[2].text = "왼손: {:.2f}".format(l_diff_mean)
                
                # diff_vis[depth_rendered > 0] = [255, 0, 0]
                diff_vis = cv2.addWeighted(rgb_captured, 0.8, diff_vis, 0.3, 0)
                self._diff_images[cam_name] = {
                    "rgb_captured": rgb_captured,
                    "right_hand_mask": right_hand_mask,
                    "left_hand_mask": left_hand_mask,
                    "object_mask": object_mask,
                    "depth_diff": depth_diff,
                }

        self.logger.debug('\tupdate error')
        total_mean = [0, 0, 0]
        count = [0, 0, 0]
        max_v = [-np.inf, -np.inf, -np.inf]
        max_idx = [None, None, None]
        min_v = [np.inf, np.inf, np.inf]
        min_idx = [None, None, None]
        for idx, diff in enumerate(self._depth_diff_list):
            for s_idx, dif in enumerate(diff):
                if dif==-1:
                    continue
                if dif > max_v[s_idx]:
                    max_v[s_idx] = dif
                    max_idx[s_idx] = idx
                if dif < min_v[s_idx]:
                    min_v[s_idx] = dif
                    min_idx[s_idx] = idx
                total_mean[s_idx] += dif
                count[s_idx] += 1
        if self._active_hand.side=='right':
            target_idx = 0 
        else:
            target_idx = 1
        for idx, error_layout in enumerate(self._view_error_layout_list):
            if idx==max_idx[target_idx]:
                error_layout[target_idx+1].text_color = gui.Color(1, 0, 0)
                error_layout[2-target_idx].text_color = gui.Color(1, 1, 1)
            elif idx==min_idx[target_idx]:
                error_layout[target_idx+1].text_color = gui.Color(0, 1, 0)
                error_layout[2-target_idx].text_color = gui.Color(1, 1, 1)
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
        try:
            total_mean[2] /= count[2]
        except:
            total_mean[2] = -1
        self._total_error_txt[0].text = "오른손: {:.2f}".format(total_mean[0])
        self._total_error_txt[1].text = "왼손: {:.2f}".format(total_mean[1])
        self._total_error_txt[2].text = "물체: {:.2f}".format(total_mean[2])
        # clear geometry
        self._log.text = "\t라벨링 검증용 이미지를 생성했습니다."
        self.window.set_needs_layout()
        self.logger.debug('End _update_valid_error')
    def _generate_valid_image(self, cam_name, only_transparency):
        def color_map(mask_type):
            if mask_type == 'right_hand':
                c = mpl.colormaps['Set1'](0)[:3][::-1]
            elif mask_type == 'left_hand':
                c = mpl.colormaps['Set1'](1)[:3][::-1]
            elif mask_type == 'object':
                c = mpl.colormaps['Set1'](2)[:3][::-1]
            else:
                c = [0, 0, 0]
            return np.uint8(np.array(c) * 255)
                
        mode = self._mask_mode.selected_text
        cam_mask_dict = self._diff_images[cam_name]
        rgb_captured = cam_mask_dict['rgb_captured']
        right_hand_mask = cam_mask_dict['right_hand_mask']
        left_hand_mask = cam_mask_dict['left_hand_mask']
        object_mask = cam_mask_dict['object_mask']
        depth_diff = cam_mask_dict['depth_diff'].copy()
        abs_depth_diff = np.abs(depth_diff)
        
        if not only_transparency or (self._cur_diff_mask is None):
            diff_vis = np.zeros_like(rgb_captured)
            high_error_mask = np.bitwise_and(abs_depth_diff>5, abs_depth_diff<300) # 5 ~ 300 mm error
            if mode == MaskMode.RGB_ALL or mode == MaskMode.MASK_ALL:
                diff_vis[right_hand_mask] = color_map('right_hand')
                diff_vis[left_hand_mask] = color_map('left_hand')
                diff_vis[object_mask] = color_map('object')
            elif mode == MaskMode.RGB_RIGHT or mode == MaskMode.MASK_RIGHT:
                diff_vis[right_hand_mask] = color_map('right_hand')
                high_error_mask = np.bitwise_and(high_error_mask, right_hand_mask)
            elif mode == MaskMode.RGB_LEFT or mode == MaskMode.MASK_LEFT:
                diff_vis[left_hand_mask] = color_map('left_hand')
                high_error_mask = np.bitwise_and(high_error_mask, left_hand_mask)
            else:
                diff_vis[object_mask] = color_map('object')
                high_error_mask = np.bitwise_and(high_error_mask, object_mask)
            
            depth_diff_vis = np.zeros_like(rgb_captured)
            depth_diff_vis[depth_diff>0] = [0, 0, 255]
            depth_diff_vis[depth_diff<0] = [255, 0, 0]
            diff_vis[high_error_mask] = depth_diff_vis[high_error_mask]
            self._cur_diff_mask = diff_vis.copy()
        else:
            diff_vis = self._cur_diff_mask.copy()

        
        if "RGB" in mode:
            val = self._rgb_transparency.double_value
            diff_vis = cv2.addWeighted(rgb_captured, val, diff_vis, 1-val, 0)
        return diff_vis
            
    def _update_pc_error(self):
        # update point cloud error
        points = []
        for side in self._hand_names:
            points.append(self._hands[side].get_mesh_points())
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(points, axis=0)))
        for obj_id, obj_model in self._objects.items():
            pcd += obj_model.get_geometry()
        target_pcd = self._pcd.crop(pcd.get_oriented_bounding_box())
        if len(target_pcd.points) > 0:
            target_pcd = target_pcd.voxel_down_sample(voxel_size=0.003)
            targets = np.asarray(target_pcd.points)
            points = np.asarray(pcd.voxel_down_sample(voxel_size=0.003).points)
            error = Utils.calc_chamfer_distance(points, targets)*10e3
            self._total_pc_error_txt.text = "포인트 에러: {:.6f}".format(error)
            if error < 1:
                self._total_pc_error_txt.text_color = gui.Color(0, 1, 0)
            else:
                self._total_pc_error_txt.text_color = gui.Color(1, 0, 0)
        else:
            self._total_pc_error_txt.text = "포인트 에러: 근처 포인트가 없음"
    
    def _img_wrapper(self, img):
        self.logger.debug('_img_wrapper')
        ratio = 640 / self.W
        img = cv2.resize(img.copy(), (640, int(self.H*ratio)))
        return o3d.geometry.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    def _reset_image_viewer(self):
        self.logger.debug('_reset_image_viewer')
        self.icx, self.icy = self.W / 2, self.H / 2
        self.scale_factor = 1
        self._move_viewer()
    @staticmethod
    def _viewer_translate(tx=0, ty=0):
                T = np.eye(3)
                T[0:2,2] = [tx, ty]
                return T
    @staticmethod
    def _viewer_scale(s=1, sx=1, sy=1):
        T = np.diag([s*sx, s*sy, 1])
        return T
    @staticmethod
    def _viewer_rotate(degrees):
        T = np.eye(3)
        # just involves some sin() and cos()
        T[0:2] = cv2.getRotationMatrix2D(center=(0,0), angle=-degrees, scale=1.0)
        return T
    def _move_viewer(self):
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
        H = self._viewer_translate(+ocx, +ocy) @ self._viewer_rotate(degrees=0) @ self._viewer_scale(self.scale_factor) @ self._viewer_translate(-self.icx, -self.icy)
        M = H[0:2]
        def img_wrapper(img):
            out = cv2.warpAffine(img.copy(), dsize=(ow,oh), M=M, flags=cv2.INTER_NEAREST)
            ratio = 640 / self.W
            img = cv2.resize(out, (640, int(self.H*ratio)))
            return o3d.geometry.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self._img_wrapper = img_wrapper
        self._update_image_viewer()
        self._update_diff_viewer()

    #endregion
    
    #region ----- Open3DScene 
    #----- geometry
    def _check_geometry(self, name):
        return self._scene.scene.has_geometry(name)
    
    def _remove_geometry(self, name):
        if self._check_geometry(name):
            self._scene.scene.remove_geometry(name)
    
    def _add_geometry(self, name, geo, mat):
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
        if self._check_geometry(name):
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
        
        self.hand_coord_labels = []
        size = size * 0.8
        self.hand_coord_labels.append(self._scene.add_3d_label(np.matmul(transform, np.array([size, 0, 0, 1]))[:3], "W, S"))
        self.hand_coord_labels.append(self._scene.add_3d_label(np.matmul(transform, np.array([0, size, 0, 1]))[:3], "A, D"))
        self.hand_coord_labels.append(self._scene.add_3d_label(np.matmul(transform, np.array([0, 0, size, 1]))[:3], "Q, E"))
        self._add_geometry("hand_frame", coord_frame, self.settings.coord_material)
    def _remove_hand_frame(self):
        self._remove_geometry("hand_frame")
        for label in self.hand_coord_labels:
            self._scene.remove_3d_label(label)
        self.hand_coord_labels = []
    def _add_obj_frame(self, size=0.1, origin=[0, 0, 0]):
        if not self._check_annotation_scene():
            return
        self._remove_obj_frame()
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
        
        # transform = np.eye(4)
        # transform[:3, 3] = self._active_object.get_transform()[:3, 3]
        transform = self._active_object.get_transform()
        coord_frame.transform(transform)

        self.obj_coord_labels = []
        size = size * 0.8
        self.obj_coord_labels.append(self._scene.add_3d_label(np.matmul(transform, np.array([size, 0, 0, 1]))[:3], "W, S"))
        self.obj_coord_labels.append(self._scene.add_3d_label(np.matmul(transform, np.array([0, size, 0, 1]))[:3], "A, D"))
        self.obj_coord_labels.append(self._scene.add_3d_label(np.matmul(transform, np.array([0, 0, size, 1]))[:3], "Q, E"))
        self._add_geometry("obj_frame", coord_frame, self.settings.coord_material)
    def _remove_obj_frame(self):
        self._remove_geometry("obj_frame")
        for label in self.obj_coord_labels:
            self._scene.remove_3d_label(label)
        self.obj_coord_labels = []
    def _on_mouse(self, event):
        if self._view_rock:
            if event.type == gui.MouseEvent.Type.BUTTON_UP:
                self._remove_geometry(self._inlier_sphere_name)

            if event.is_modifier_down(gui.KeyModifier.ALT):
                if event.is_button_down(gui.MouseButton.LEFT):
                    if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
                        self._last_xyz = None
                    world_xyz = self._mouse_to_world_xyz(event)
                    if world_xyz is not None:
                        if self._check_xyz_movement(world_xyz):
                            self._on_select_pcd(world_xyz)
                elif event.is_button_down(gui.MouseButton.RIGHT):
                    if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
                        self._last_xyz = None
                    world_xyz = self._mouse_to_world_xyz(event)
                    if world_xyz is not None:
                        if self._check_xyz_movement(world_xyz):
                            self._on_select_pcd(world_xyz, invert=True)
            else:
                if event.is_button_down(gui.MouseButton.LEFT):
                    if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
                        self._last_xyz = None
                    world_xyz = self._mouse_to_world_xyz(event)
                    if world_xyz is not None:
                        if self._check_xyz_movement(world_xyz):
                            self._on_select_mesh(world_xyz)
                elif event.is_button_down(gui.MouseButton.RIGHT):
                    if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
                        self._last_xyz = None
                    world_xyz = self._mouse_to_world_xyz(event)
                    if world_xyz is not None:
                        if self._check_xyz_movement(world_xyz):
                            self._on_select_mesh(world_xyz, invert=True)
            return gui.Widget.EventCallbackResult.CONSUMED
        else:
            # We could override BUTTON_DOWN without a modifier, but that would
            # interfere with manipulating the scene.
            if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                    gui.KeyModifier.ALT) and event.is_button_down(gui.MouseButton.LEFT):
                if not self._check_annotation_scene():
                    return gui.Widget.EventCallbackResult.IGNORED
                def depth_callback(depth_image):
                    self.logger.debug('depth_callback')
                    # Coordinates are expressed in absolute coordinates of the
                    # window, but to dereference the image correctly we need them
                    # relative to the origin of the widget. Note that even if the
                    # scene widget is the only thing in the window, if a menubar
                    # exists it also takes up space in the window (except on macOS).
                    x = event.x - self._scene.frame.x
                    y = event.y - self._scene.frame.y
                    # Note that np.asarray() reverses the axes.
                    depth_area = np.asarray(depth_image)[y-10:y+10, x-10:x+10]
                    if depth_area.min() == 1.0: # clicked on nothing (i.e. the far plane)
                        pass
                    
                    else:
                        depth = np.mean(depth_area[depth_area!=1.0])
                    
                        world_xyz = self._scene.scene.camera.unproject(
                            event.x, event.y, depth, self._scene.frame.width,
                            self._scene.frame.height)
                        def move_joint():
                            if self._active_type=='hand':
                                self._move_hand_translation(world_xyz)
                            elif self._active_type=='object':
                                self._move_object_translation(world_xyz)
                        gui.Application.instance.post_to_main_thread(
                            self.window, move_joint)
                    
                self._scene.scene.scene.render_to_depth_image(depth_callback)
                return gui.Widget.EventCallbackResult.CONSUMED
            
            if not self._labeling_mode==LabelingMode.OBJECT:
                if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                        gui.KeyModifier.SHIFT) and event.is_button_down(gui.MouseButton.RIGHT):
                    ctrl_idx = self._active_hand.control_idx + 1
                    self._active_hand.set_control_joint(ctrl_idx)
                    self._update_target_hand()
                    self._update_joint_mask()
                    self.logger.debug("convert joint {}".format(self._active_hand.get_control_joint_name()))
                if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                        gui.KeyModifier.CTRL) and event.is_button_down(gui.MouseButton.RIGHT):
                    ctrl_idx = self._active_hand.control_idx - 1
                    self._active_hand.set_control_joint(ctrl_idx)
                    self._update_target_hand()
                    self._update_joint_mask()
                    self.logger.debug("convert joint {}".format(self._active_hand.get_control_joint_name()))
                
        return gui.Widget.EventCallbackResult.IGNORED
    
    def _check_xyz_movement(self, xyz):
        try:
            movement = np.linalg.norm(xyz - self._last_xyz)
        except:
            movement = 10
        if movement < self._inlier_radius*0.5:
            return False
        else:
            self._last_xyz = xyz
            return True
    def _depth_callback(self, depth_image):
        self._depth_image = np.asarray(depth_image)
    def _mouse_to_world_xyz(self, event):
        if self._depth_image is None:
            return None
        x = event.x - self._scene.frame.x
        y = event.y - self._scene.frame.y
        if x < 0 or x >= self._depth_image.shape[1] or y < 0 or y >= self._depth_image.shape[0]:
            return None
        
        sx = min(max(0, x-10), self._depth_image.shape[1]-1) 
        ex = min(max(0, x+10), self._depth_image.shape[1]-1)
        sy = min(max(0, y-10), self._depth_image.shape[0]-1)
        ey = min(max(0, y+10), self._depth_image.shape[0]-1)
        
        depth_area = self._depth_image[sy:ey, sx:ex]
        if depth_area.min() == 1.0: # clicked on nothing (i.e. the far plane)
            return None
        else:
            depth = np.mean(depth_area[depth_area!=1.0])
            world_xyz = self._scene.scene.camera.unproject(
                        event.x, event.y, depth, self._scene.frame.width,
                        self._scene.frame.height)
            return world_xyz
    def _on_select_mesh(self, xyz, invert=False):
        self.logger.debug('_on_select_mesh')
        self._update_inlier_sphere(xyz)
        for side, hand_model in self._hands.items():
            hand_model.update_active_faces(xyz, self._inlier_radius, invert)
        self._update_hand_layer()
    def _on_select_pcd(self, xyz, invert=False):
        self.logger.debug('_on_select_pcd')
        self._update_inlier_sphere(xyz)
        points = np.asarray(self._pcd.points)
        dist = np.linalg.norm(points - xyz, axis=1)
        inlier_points = np.where(dist < 0.005)[0].tolist()
        if invert:
            inlier_points = list(set(self._inlier_points) - set(inlier_points))
        else:
            inlier_points = list(set(inlier_points) | set(self._inlier_points))
        self._inlier_points = inlier_points
        self._update_activate_pcd()
    def _update_activate_mesh(self):
        self.logger.debug('_update_activate_mesh')
        for side, hand_model in self._hands.items():
            active_mesh, nonactive = hand_model.get_active_mesh(return_inactive=True)
            if self._show_hands.checked:
                if self._active_hand==hand_model:
                    self._add_geometry(self._active_mesh_name.format(side), active_mesh, mat=self.settings.active_mesh_material)    
                    self._add_geometry(self._hand_mesh_name.format(side), nonactive, mat=self.settings.active_hand_mesh_material)
                else:
                    self._add_geometry(self._hand_mesh_name.format(side), nonactive, mat=self.settings.nonactive_mesh_material)
            else:
                self._remove_geometry(self._hand_mesh_name.format(side))
    def _update_activate_pcd(self):
        self.logger.debug('_update_activate_pcd')
        pcd = self._pcd
        activate = pcd.select_by_index(self._inlier_points)
        nonactivate = pcd.select_by_index(self._inlier_points, invert=True)
        self._add_geometry(self._active_pcd_name, activate, mat=self.settings.active_pcd_material)
        self._add_geometry(self._nonactive_pcd_name, nonactivate, mat=self.settings.scene_material)
    def _update_inlier_sphere(self, xyz):
        self.logger.debug('_update_inlier_sphere')
        if self._view_rock:
            center = self._inlier_sphere.get_center()
            self._inlier_sphere.translate(xyz - center)
            self._add_geometry(self._inlier_sphere_name, self._inlier_sphere, mat=self.settings.inlier_sphere_material)
        else:
            self._remove_geometry(self._inlier_sphere_name)
        
    def _toggle_view_rock(self):
        self._view_rock = not self._view_rock
        if self._view_rock:
            self._scene.scene.scene.render_to_depth_image(self._depth_callback)
        else:
            self._remove_geometry(self._inlier_sphere_name)
        self._update_hand_layer()
    
    def move(self, x, y, z, rx, ry, rz):
        self.logger.debug('move_{}'.format(self._active_type))
        if self._active_type=='hand':

            self._log.text = "{} 라벨 이동 중입니다.".format(self._active_hand.get_control_joint_name())
            self.window.set_needs_layout()
            self._last_change = time.time()
            if x != 0 or y != 0 or z != 0: # translation
                current_xyz = self._active_hand.get_control_position()
                # convert x, y, z cam to world
                R = self._scene.scene.camera.get_view_matrix()[:3,:3]
                R_inv = np.linalg.inv(R)
                xyz = np.dot(R_inv, np.array([x, y, z]))
                xyz = current_xyz + xyz
                
                self._move_hand_translation(xyz)
            else: # rotation
                current_xyz = self._active_hand.get_control_rotation()
                r = Rot.from_rotvec(current_xyz)
                current_rot_mat = r.as_matrix()
                rot_mat = o3d.geometry.get_rotation_matrix_from_xyz((rx, ry, rz))
                r = Rot.from_matrix(np.matmul(current_rot_mat, rot_mat))
                xyz = r.as_rotvec()

                self._move_hand_rotation(xyz)
        else:
            self._log.text = "물체 라벨 이동 중입니다."
            self.window.set_needs_layout()
            self._last_change = time.time()
            if x != 0 or y != 0 or z != 0: # translation
                # convert x, y, z cam to world
                R = self._scene.scene.camera.get_view_matrix()[:3,:3]
                R_inv = np.linalg.inv(R)
                xyz = np.dot(R_inv, np.array([x, y, z]))
                h_transform = np.eye(4)
                h_transform[:3, 3] = xyz
                
            else: # rotation
                center = self._active_object.obj_geo.get_center()
                current_rot = self._active_object.get_transform()[:3, :3]
                rot_vec = rx*current_rot[:, 0] + ry*current_rot[:, 1] + rz*current_rot[:, 2]
                r = Rot.from_rotvec(rot_vec)
                rot_mat = r.as_matrix()
                T_neg = np.vstack((np.hstack((np.identity(3), -center.reshape(3, 1))), [0, 0, 0, 1]))
                R = np.vstack((np.hstack((rot_mat, [[0], [0], [0]])), [0, 0, 0, 1]))
                T_pos = np.vstack((np.hstack((np.identity(3), center.reshape(3, 1))), [0, 0, 0, 1]))
                h_transform = T_pos @ R @ T_neg
            self._active_object.transform(h_transform)
            self._annotation_changed = True
            self._update_object_layer()
    # move hand
    def _move_hand_translation(self, xyz):
        self._active_hand.set_control_position(xyz)
        if self._active_hand.get_optimize_target()=='root':
            self._active_hand.save_undo(forced=True)
            self._annotation_changed = True
            self._move_root = True
        else:
            self._guide_changed = True
        self._update_hand_layer()
    def _move_hand_rotation(self, xyz):
        self._active_hand.set_control_rotation(xyz)
        self._active_hand.save_undo()
        self._annotation_changed = True
        self._update_hand_layer()
    def _move_object_translation(self, xyz):
        self._active_object.set_position(xyz)
        self._annotation_changed = True
        self._update_object_layer()

    def _init_pcd_layer(self):
        self.logger.debug('_init_pcd_layer')
        if self.settings.show_pcd:
            self._pcd = self._frame.scene_pcd
            self.bounds = self._pcd.get_axis_aligned_bounding_box()
            self._add_geometry(self._scene_name, self._pcd, self.settings.scene_material)
        else:
            self._remove_geometry(self._scene_name)
    def _update_pcd_layer(self):
        self.logger.debug('_update_pcd_layer')
        if self.settings.show_pcd:
            cam_name_list = []
            for cam_name in self._get_activate_cam():
                cam_name_list.append(cam_name)
            if cam_name_list == []:
                self._remove_geometry(self._scene_name)
                return
            self._pcd = self._frame.get_pcd(cam_name_list)
            self.bounds = self._pcd.get_axis_aligned_bounding_box()
            if self._down_sample_pcd.checked:
                pcd = self._get_active_pcd(self._pcd)
            else:
                pcd = self._pcd
            
            self._add_geometry(self._scene_name, pcd, self.settings.scene_material)
            self._inlier_points = []
        else:
            self._remove_geometry(self._scene_name)
    def _get_active_pcd(self, pcd):
        hand = self._active_hand.get_geometry()['mesh']
        bbox = hand.get_axis_aligned_bounding_box()
        bbox = bbox.scale(3, bbox.get_center())
        
        active_pcd = pcd.crop(bbox)
        nactive_pcd = pcd.voxel_down_sample(0.02)
        return active_pcd+nactive_pcd

    def _toggle_pcd_visible(self):
        self.logger.debug('_toggle_pcd_visible')
        show = self._show_pcd.checked
        self._show_pcd.checked = not show
        self._on_show_pcd(not show)
    
    def _init_obj_layer(self):
        self.logger.debug('_init_obj_layer')
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
    def _update_object_layer(self):
        if self.settings.show_objects:
            for obj_id, obj in self._objects.items():
                obj_name = "obj_{}".format(obj_id)
                if obj_name == self._active_object_name:
                    mat = self.settings.active_obj_material
                    self._active_object = obj
                else:
                    mat = self.settings.obj_material
                self._add_geometry(obj_name, obj.get_geometry(), mat)
        else:
            for obj_id, obj in self._objects.items():
                obj_name = "obj_{}".format(obj_id)
                self._remove_geometry(obj_name)
    @property
    def _active_object_name(self):
        if self._active_object_idx==-1:
            return ""
        else:
            return self._object_names[self._active_object_idx]

    def _init_hand_layer(self):
        self.logger.debug('_init_hand_layer')
        # remove all hands
        for name in self._hand_geometry_names:
            self._remove_geometry(name)
        # visualize hand
        hands = self._frame.hands
        self._hand_names = list(hands.keys())
        self._hand_names.sort()
        active_side = self._hand_names[self._active_hand_idx]
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
    def _update_hand_layer(self):
        if self._active_type=='hand':
            self._update_activate_hand()
            self._update_target_hand()
            self._update_joint_mask()
        elif self._active_type=='mesh':
            self._update_activate_mesh()
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
        self.logger.debug('_toggle_hand_visible')
        show = self._show_hands.checked
        self._show_hands.checked = not show
        self._on_show_hand(not show)
    def _on_optimize(self):
        self._log.text = "\t {} 자동 정렬 중입니다.".format(self._active_hand.get_control_joint_name())
        self.window.set_needs_layout()
        self._last_change = time.time()
        self._annotation_changed = self._active_hand.optimize_to_target()
        if self._annotation_changed:
            self._update_hand_layer()
            self._active_hand.save_undo()
    def _on_icp(self):
        self.logger.debug('_on_icp')
        if not self._check_annotation_scene():
            return
        # TODO: Selected mesh and (option) selected points
        if len(self._inlier_points) == 0:
            self._on_error("No inlier points selected")
            return
        target_pcd = self._pcd.select_by_index(self._inlier_points)
        if len(target_pcd.points) > 10000:
            # downsampling
            sample = min(10000, len(target_pcd.points))
            target_pcd = target_pcd.uniform_down_sample(sample)
        target_points = np.asarray(target_pcd.points)
        ret = self._active_hand.optimize_to_points(target_points)
        if not ret:
            self._on_error("최적화 실패")
        self._update_hand_layer()
        self._active_hand.save_undo()
            
    def _undo(self):
        self.logger.debug('_undo')
        self._auto_optimize.checked = False
        ret = self._active_hand.undo()
        if not ret:
            self._on_error("이전 상태가 없습니다. (error at _undo)")
        else:
            self._log.text = "이전 상태를 불러옵니다."
            self.window.set_needs_layout()
            self._update_hand_layer()
    def _redo(self):
        self.logger.debug('_redo')
        self._auto_optimize.checked = False
        ret = self._active_hand.redo()
        if not ret:
            self._on_error("이후 상태가 없습니다. (error at _redo)")
        else:
            self._log.text = "이후 상태를 불러옵니다."
            self.window.set_needs_layout()
            self._update_hand_layer()
        
    def _on_key(self, event):
        if self._active_hand is None:
            return gui.Widget.EventCallbackResult.IGNORED

        # mode change
        if event.key == gui.KeyName.F1 and event.type == gui.KeyEvent.DOWN:
            self._convert_mode(LabelingMode.STATIC)
            return gui.Widget.EventCallbackResult.CONSUMED
        elif event.key == gui.KeyName.F2 and event.type == gui.KeyEvent.DOWN:
            self._convert_mode(LabelingMode.OPTIMIZE)
            return gui.Widget.EventCallbackResult.CONSUMED
        elif event.key == gui.KeyName.F3 and event.type == gui.KeyEvent.DOWN:
            self._convert_mode(LabelingMode.OBJECT)
            return gui.Widget.EventCallbackResult.CONSUMED
        # elif event.key == gui.KeyName.F4 and event.type == gui.KeyEvent.DOWN:
        #     self._convert_mode(LabelingMode.MESH)
        #     return gui.Widget.EventCallbackResult.CONSUMED
        # if ctrl is pressed then  increase translation / reset finger to flat
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

        # edit view point
        if event.key == gui.KeyName.T and event.type == gui.KeyEvent.DOWN:
            self._on_initial_viewpoint()
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key == gui.KeyName.Y and event.type == gui.KeyEvent.DOWN:
            self._on_active_camera_viewpoint()
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key == gui.KeyName.G and event.type == gui.KeyEvent.DOWN:
            self._on_active_viewpoint()
            return gui.Widget.EventCallbackResult.HANDLED
        
        # update error
        if event.key == gui.KeyName.SLASH and (event.type==gui.KeyEvent.DOWN):
            self._update_valid_error(calculate=True)
            self._update_pc_error()
            return gui.Widget.EventCallbackResult.HANDLED
        
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
            self._move_viewer()
            
            # out = cv2.warpAffine(self.rgb_img.copy(), dsize=(ow,oh), M=M, flags=cv2.INTER_NEAREST)
            # ratio = 640 / self.W
            # _rgb_img = cv2.resize(out, (640, int(self.H*ratio)))
            # _rgb_img = o3d.geometry.Image(cv2.cvtColor(_rgb_img, cv2.COLOR_BGR2RGB))
            
            
            # out = cv2.warpAffine(self.diff_img.copy(), dsize=(ow,oh), M=M, flags=cv2.INTER_NEAREST)
            # ratio = 640 / self.W
            # _diff_img = cv2.resize(out, (640, int(self.H*ratio)))
            # _diff_img = o3d.geometry.Image(cv2.cvtColor(_diff_img, cv2.COLOR_BGR2RGB))
            
            
            return gui.Widget.EventCallbackResult.HANDLED
        
        # save label
        if event.key==gui.KeyName.F and event.type==gui.KeyEvent.DOWN:
            self._on_save_label()
            return gui.Widget.EventCallbackResult.HANDLED
        


        if self._labeling_mode==LabelingMode.OBJECT:
            if (event.key == gui.KeyName.LEFT_SHIFT or event.key == gui.KeyName.RIGHT_SHIFT):
                if event.type == gui.KeyEvent.DOWN:
                    self._left_shift_modifier = True
                    self._add_obj_frame()
                elif event.type == gui.KeyEvent.UP:
                    self._left_shift_modifier = False
                    self._remove_obj_frame()
            if event.key == gui.KeyName.R and event.type==gui.KeyEvent.DOWN:
                self._active_object.reset()
                self._update_object_layer()
                return gui.Widget.EventCallbackResult.CONSUMED
        # elif self._labeling_mode==LabelingMode.MESH:
        #     if event.key == gui.KeyName.B and (event.type==gui.KeyEvent.DOWN):
        #         self._toggle_view_rock()
        #         return gui.Widget.EventCallbackResult.HANDLED
        #     if event.key == gui.KeyName.R and event.type==gui.KeyEvent.DOWN:
        #         for side, hand_model in self._hands.items():
        #             hand_model.active_faces = []
        #         self._inlier_points = []
        #         self._update_hand_layer()
        #         self._update_activate_pcd()
        #         return gui.Widget.EventCallbackResult.HANDLED
        #     if event.key == gui.KeyName.SPACE:
        #         self._on_icp()
        #     if (event.key == gui.KeyName.TAB) and (event.type==gui.KeyEvent.DOWN):
        #         self._convert_hand()
        #         return gui.Widget.EventCallbackResult.CONSUMED
        else:
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
            
        
            # convert hand
            if (event.key == gui.KeyName.TAB) and (event.type==gui.KeyEvent.DOWN):
                self._convert_hand()
                return gui.Widget.EventCallbackResult.CONSUMED
            
            # reset hand pose
            if event.key == gui.KeyName.R and event.type==gui.KeyEvent.DOWN:
                if self.reset_flat:
                    self._active_hand.reset_pose(flat_hand=True)
                else:
                    self._active_hand.reset_pose()
                self._update_hand_layer()
                return gui.Widget.EventCallbackResult.CONSUMED

            if (event.key==gui.KeyName.COMMA and event.type==gui.KeyEvent.DOWN):
                self._undo()
                return gui.Widget.EventCallbackResult.CONSUMED
            elif (event.key==gui.KeyName.PERIOD and event.type==gui.KeyEvent.DOWN):
                self._redo()
                return gui.Widget.EventCallbackResult.CONSUMED
            
            if event.key == gui.KeyName.ZERO and (event.type==gui.KeyEvent.DOWN):
                self._active_hand.toggle_current_joint_lock()
                self._update_joint_mask()
                return gui.Widget.EventCallbackResult.HANDLED
            
            # convert finger
            is_converted_finger = True
            if event.key == gui.KeyName.BACKTICK:
                self._active_hand.set_optimize_target('root')
            elif event.key == gui.KeyName.ONE and (event.type==gui.KeyEvent.DOWN):
                self._active_hand.set_optimize_target('thumb')
            elif event.key == gui.KeyName.TWO and (event.type==gui.KeyEvent.DOWN):
                self._active_hand.set_optimize_target('fore')
            elif event.key == gui.KeyName.THREE and (event.type==gui.KeyEvent.DOWN):
                self._active_hand.set_optimize_target('middle')
            elif event.key == gui.KeyName.FOUR and (event.type==gui.KeyEvent.DOWN):
                self._active_hand.set_optimize_target('ring')
            elif event.key == gui.KeyName.FIVE and (event.type==gui.KeyEvent.DOWN):
                self._active_hand.set_optimize_target('little')
            else:
                is_converted_finger = False
            
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
                self.logger.debug("convert joint {}".format(self._active_hand.get_control_joint_name()))
                self._update_target_hand()
                self._update_joint_mask()
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
                    self._guide_changed = False
                    self._active_hand.reset_target()
                    self._update_target_hand()
                    return gui.Widget.EventCallbackResult.CONSUMED
            
            
        # Translation
        if event.type!=gui.KeyEvent.UP:
            if not self._left_shift_modifier and \
                (self._labeling_mode==LabelingMode.OPTIMIZE or self._active_hand.get_optimize_target()=='root' or self._labeling_mode==LabelingMode.OBJECT):
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
                (self._labeling_mode==LabelingMode.STATIC or self._active_hand.get_optimize_target()=='root' or self._labeling_mode==LabelingMode.OBJECT):
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
                if self._active_type=='hand':
                    self._add_hand_frame()
                elif self._active_type=='object':
                    self._add_obj_frame()
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
            if self._labeling_mode==LabelingMode.OPTIMIZE and self._guide_changed:
                if not self._active_hand.get_optimize_target()=='root':
                    self._on_optimize()
        
        if self._auto_save.checked and self.annotation_scene is not None:
            if (time.time()-self._last_saved) > self._auto_save_interval.double_value and self._annotation_changed:
                self.logger.debug('auto saving')
                self._annotation_changed = False
                self.annotation_scene.save_label()
                self._last_saved = time.time()
                self._log.text = "라벨 결과 자동 저장중입니다."
                self.window.set_needs_layout()
                if self._error_box.checked:
                    self._update_valid_error()
                    self._update_diff_viewer()
        
        if self._down_sample_pcd.checked and self._move_root:
            self._update_pcd_layer()
            self._move_root = False
        self.logger.memory_usage()
        self._init_view_control()

def main(logger):
    gui.Application.instance.initialize()
    
    font = gui.FontDescription(hangeul)
    font.add_typeface_for_language(hangeul, "ko")
    gui.Application.instance.set_font(gui.Application.DEFAULT_FONT_ID, font)

    w = AppWindow(1920, 1080, logger)
    gui.Application.instance.run()

if __name__ == "__main__":
    logger = Logger('hand-pose-annotator')
    atexit.register(logger.handle_exit)
    main(logger)