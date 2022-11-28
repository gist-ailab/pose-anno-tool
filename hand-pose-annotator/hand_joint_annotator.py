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

from typing import List, NamedTuple, Optional, TYPE_CHECKING, Union

import torch
from manopth.manolayer import ManoLayer

import numpy as np
import cv2
import pickle

import yaml
import time
import json
import copy

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
        
class MaskMode:
    RGB_ALL     = "RGB 전체"
    RGB_FINGER  = "RGB 손가락"
    MASK_ALL    = "MASK 전체"
    MASK_FINGER  = "MASK 손가락"
    
class MultiViewJointModel:
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
    _IDX_OF_JOINT = {
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

    _COLOR_OF_JOINT = [
        [255, 255, 255], # white
        [255, 0, 0], # red
        [255, 0, 0], # red
        [255, 0, 0], # red
        [255, 0, 0], # red
        [255, 255, 0], # yellow
        [255, 255, 0], # yellow
        [255, 255, 0], # yellow
        [255, 255, 0], # yellow
        [0, 255, 0], # green
        [0, 255, 0], # green
        [0, 255, 0], # green
        [0, 255, 0], # green
        [0, 255, 255], # cyan
        [0, 255, 255], # cyan
        [0, 255, 255], # cyan
        [0, 255, 255], # cyan
        [0, 0, 255], # blue
        [0, 0, 255], # blue
        [0, 0, 255], # blue
        [0, 0, 255], # blue
    ]
    
    def __init__(self, side, serials, shape_param=None):
        self.side = side
        self.serials = serials
        
        self.joint_state = {
            serial: {
                'pose': np.zeros((21, 3), dtype=np.float32),
                'visible': [True for _ in range(21)],
            } for serial in self.serials
        }
        self.mano_layer = ManoLayer(mano_root=MANO_PATH, side=side,
                            use_pca=False, flat_hand_mean=True, joint_rot_mode='axisang')
        self.faces = self.mano_layer.th_faces
        
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
    
    def reset(self, shape_param=None, flat_hand=False):
        """reset hand model

        - shape

        Args:
            shape_param (_type_, optional): _description_. Defaults to None.
            flat_hand (bool, optional): _description_. Defaults to False.
        """
        # initialize shape param
        if shape_param is None:
            pass
        else:
            self.shape_param = torch.Tensor(shape_param).unsqueeze(0)
        root_trans = torch.zeros(1, 3)
        if flat_hand:
            pose_param = torch.zeros(1, 48)
        else:
            pose_param = torch.concat((torch.zeros(3), torch.tensor(self.mano_layer.smpl_data['hands_mean'], dtype=torch.float32))).unsqueeze(0)
        verts, joints = self.mano_layer(th_pose_coeffs=pose_param,
                                        th_betas=self.shape_param,
                                        th_trans=root_trans)
        verts = verts / 1000
        joints = joints / 1000
        self.root_delta = joints.cpu().detach()[0, 0]
        self.optimize_idx = self._IDX_OF_JOINT
        self.optimize_target = 'root'
        self.active_joints = self.optimize_idx[self.optimize_target]
        self.control_idx = 0 # idx of joint to control (in active_joints)
        self.control_joint = 0 # idx of joint to control (in joints)
        
        # init label as all camera
        self.ai_label = None
        self.ai_joints = None
        
        # specific view label
        self.current_serial = self.serials[0]
        self.current_joints = None
        self.current_visible = None

    def load_ai_label(self, val=None):
        if val is None:
            self.ai_label = None
            pose_param = torch.concat((torch.zeros(3), torch.tensor(self.mano_layer.smpl_data['hands_mean'], dtype=torch.float32))).unsqueeze(0)
            root_trans = torch.zeros(1, 3)
        else:    
            self.ai_label = val
            mano_pose = np.array(val['mano_pose']).reshape(-1)
            wrist_pos = np.array(val['wrist_pos'])
            wrist_ori = np.array(val['wrist_ori'])
            mano_pose = np.concatenate((wrist_ori, mano_pose))
            pose_param = torch.tensor(mano_pose, dtype=torch.float32).unsqueeze(0) # 1, 48
            root_trans = torch.tensor(wrist_pos, dtype=torch.float32).unsqueeze(0) # 1, 3
        verts, joints = self.mano_layer(th_pose_coeffs=pose_param,
                                        th_betas=self.shape_param,
                                        th_trans=root_trans)
        self.ai_joints = np.array(joints.cpu().detach()[0] / 1000)
    def get_ai_joints(self):
        return self.ai_joints
    def reset_to_ai_label(self):
        for serial in self.joint_state.keys():
            self.joint_state[serial]['pose'] = copy.deepcopy(self.ai_joints)
            self.joint_state[serial]['visible'] = [True for _ in range(21)]
        self.current_serial = "merge"
        self.current_visible = [True for _ in range(21)]
        self.current_joints = copy.deepcopy(self.ai_joints)
    def set_joint_state(self, joint_state):
        # check joint_state and self.joint_state key are same
        for serial in self.joint_state.keys():
            self.joint_state[serial]['pose'] = np.array(joint_state[serial]['pose'])
            self.joint_state[serial]['visible'] = joint_state[serial]['visible']
    def get_joint_state(self):
        temp = {}
        for serial, state in self.joint_state.items():
            temp[serial] = {
                "pose": state["pose"].tolist(),
                "visible": state["visible"]
            }
        return temp        
    def get_current_joints(self):
        return self.current_joints
    def convert_camera(self, cam):
        if cam=="merge":
            self.current_serial = "merge"
            self.current_visible = [True for _ in range(21)]
            self.current_joints = copy.deepcopy(self.ai_joints)
        else:
            self.current_serial = cam.serial
            self.current_visible = self.joint_state[self.current_serial]['visible']
            self.current_joints = self.joint_state[self.current_serial]['pose']
    
    def get_visible_state(self, cam):
        return self.joint_state[cam.serial]['visible']
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
    def get_control_position(self):
        return self.current_joints[self.control_joint]
    def set_control_position(self, xyz):
        if self.optimize_target == 'root':
            self.current_joints[0] = xyz
        else:
            self.current_joints[self.control_joint] = xyz
        return True
    def get_optimize_target(self):
        return self.optimize_target
    def set_optimize_target(self, target):
        self.optimize_target = target
        self.active_joints = self.optimize_idx[self.optimize_target]
        self.control_idx = 0
        self.control_joint = self.active_joints[self.control_idx]
    def set_control_joint(self, idx):
        assert len(self.active_joints) > 0, "set_control_joint error"
        idx = np.clip(idx, 0, len(self.active_joints)-1) 
        self.control_idx = idx
        self.control_joint = self.active_joints[self.control_idx]
    def get_hand_shape(self):
        shape = np.array(self.shape_param.cpu().detach()[0, :])
        return shape.tolist()
    def get_geometry(self):
        return {
            "joint": self._get_joints(),
            "link": self._get_links()
        }
    def _get_joints(self, idx=None):
        if idx is None:
            joints = self.current_joints
        else:
            joints = self.current_joints[idx]
        joints = o3d.utility.Vector3dVector(joints)
        pcd = o3d.geometry.PointCloud(points=joints)
        return pcd
    def get_links(self):
        joints = self.current_joints
        joints = o3d.utility.Vector3dVector(joints)
        lines = o3d.utility.Vector2iVector(np.array(self.LINK))
        lineset = o3d.geometry.LineSet(lines=lines, points=joints)
        return lineset

    def get_active_geometry(self):
        return {
            "joint": self._get_joints(self.active_joints),
            "control": self._get_joints([self.control_joint])
        }
        
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
        mask[self.joint_mask['active'][self.control_joint]] = [255, 0, 0]
        # draw lock state
        for i in range(21):
            is_visible = self.current_visible[i]
            cnt = self._active_img[i]

            if not is_visible:
                s1, e1 = [cnt[0]-7, cnt[1]-7], [cnt[0]+7, cnt[1]+7]
                s2, e2 = [cnt[0]-7, cnt[1]+7], [cnt[0]+7, cnt[1]-7]
                mask = cv2.line(mask, s1, e1, [255, 0, 0], 2)
                mask = cv2.line(mask, s2, e2, [255, 0, 0], 2)
            else:
                mask = cv2.circle(mask, cnt, 7, [0, 255, 0], 2)
        return mask
    
    def toggle_current_joint_visible(self):
        self.current_visible[self.control_joint] = not self.current_visible[self.control_joint]
    
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
        self.mesh_path = self.model_path.replace("object", "mesh")
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
        mesh = o3d.io.read_triangle_mesh(self.mesh_path)
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
            "right": MultiViewJointModel(side='right', serials=self._SERIALS),
            "left": MultiViewJointModel(side='left', serials=self._SERIALS)
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

        def convert_camera(self, camera_name):
            self.active_cam = camera_name
            if self.active_cam=='merge':
                for hand in self.hands.values():
                    hand.convert_camera("merge")
            else:
                for hand in self.hands.values():
                    hand.convert_camera(self.cameras[camera_name])
        
        def get_visible_state(self, camera_name):
            temp = {}
            for side, hand in self.hands.items():
                temp[side] = hand.get_visible_state(self.cameras[camera_name])
            return temp
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
        
        self._joint_label_format = os.path.join(self._label_dir, "joints_{:06d}.json")
        self._hand_label_format = os.path.join(self._label_dir, "hands_{:06d}.npz")
        self._object_label_format = os.path.join(self._label_dir, "objs_{:06d}.npz")

        self._load_point_cloud = load_pcd

        self._joint_label = None
        self._previous_joint_label = None
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
        with open(self._joint_label_path, 'w') as f:
            json.dump(self._joint_label, f)
    def load_label(self):
        #----- object label
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
        #----- hand label (from ai label)
        ai_label_path = os.path.join(self._scene_dir, "hand_mocap", "{:06d}.json".format(self.frame_id))
        if os.path.exists(ai_label_path):
            ai_label = Utils.load_json_to_dic(ai_label_path)
            for side, hand_model in self._hands.items():
                try:
                    val = ai_label[side]
                    hand_model.load_ai_label(val)
                except:
                    hand_model.load_ai_label()
        else:
            for side, hand_model in self._hands.items():
                hand_model.load_ai_label()
        #----- joint label
        #1. load saved label
        try:
            with open(self._joint_label_path, 'r') as f:
                self._joint_label = json.load(f)
            for side, hand_model in self._hands.items():
                val = self._joint_label[side]
                hand_model.set_joint_state(val)
        except:
            print("Fail to load exist label -> try load previous label")
            #2. if not exist, load previous label
            if self._previous_joint_label is not None:
                self._joint_label = copy.deepcopy(self._previous_joint_label)
                for side, hand_model in self._hands.items():
                    val = self._joint_label[side]
                    hand_model.set_joint_state(val)
            else:
                print("Fail to load previous label -> Initialize label with AI label")
            #3. if not exist, initialize joint label with ai label
            for side, hand_model in self._hands.items():
                hand_model.reset_to_ai_label()
        self._joint_label = {
            side: hand_model.get_joint_state() for side, hand_model in self._hands.items()
        }
    def _load_joint_label(self, npz_file):
        try:
            label = dict(np.load(npz_file))
            self._joint_label = label
            for side, hand_model in self._hands.items():
                val = label[side]
                hand_model.set_joint_state(val)
            return True
        except:
            return False
    @property
    def _frame_path(self):
        return os.path.join(self._scene_dir, self._data_format.format(self.frame_id))
    @property
    def _joint_label_path(self):
        return os.path.join(self._scene_dir, self._joint_label_format.format(self.frame_id))
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

        self.hand_joint_material = []
        for i in range(21):
            mat = rendering.MaterialRecord()
            mat.base_color = list(np.array(MultiViewJointModel._COLOR_OF_JOINT[i])/255) + [1]
            mat.shader = Settings.SHADER_UNLIT
            mat.point_size = 5.0
            self.hand_joint_material.append(mat)
        
        self.hand_link_material = rendering.MaterialRecord()
        self.hand_link_material.base_color = [1.0, 0.0, 0.0, 1.0]
        self.hand_link_material.shader = Settings.SHADER_LINE
        self.hand_link_material.line_width = 2.0
        
        self.active_link_material = rendering.MaterialRecord()
        self.active_link_material.base_color = [0.0, 0.0, 1.0, 1.0]
        self.active_link_material.shader = Settings.SHADER_LINE
        self.active_link_material.line_width = 3.0
        
        self.active_target_joint_material = rendering.MaterialRecord()
        self.active_target_joint_material.base_color = [1.0, 0.75, 0.75, 1.0]
        self.active_target_joint_material.shader = Settings.SHADER_UNLIT
        self.active_target_joint_material.point_size = 20.0
        
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
            mesh = hand_geo['mesh']
            joint = hand_geo['joint']
            link = hand_geo['link']
            tmp_geo = copy.deepcopy(joint)
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

    def get_projection_matrix(self):
        return self.render.scene.camera.get_projection_matrix()

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
        self._window_name = "3D Hand Joint Annotator by GIST AILAB"
        self._scene_name = "annotation_scene"
        
        #----- hand model geometry name
        self._hand_joint_name = "{}_hand_joint_{}"
        self._hand_link_name = "{}_hand_link"
        #----- active/control geometry name
        self._active_joint_name = "active_joint"
        self._control_joint_name = "control_joint"
        
        self._hand_geometry_names = [
            self._hand_link_name.format("right"),
            self._hand_link_name.format("left"),
            self._active_joint_name,
            self._control_joint_name
        ]
        for i in range(21):
            self._hand_geometry_names += [self._hand_joint_name.format(side, i) for side in ['right', 'left']]
        
        self._active_pcd_name = "active_scene"

        #----- intialize values
        self.dataset = None
        self.annotation_scene = None
        
        self._pcd = None
        self._hands = None
        self._active_hand = None
        self.upscale_responsiveness = False
        self._left_shift_modifier = False
        self._annotation_changed = False
        self._last_change = time.time()
        self._last_saved = time.time()
        self.hand_coord_labels = []
        
        self._depth_image = None
        self.prev_ms_x, self.prev_ms_y = None, None
        
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
        self.obj_coord_labels = []
        
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
        self._on_object_transparency(0)
        self._on_hand_point_size(5) # set default size to 10
        self._on_hand_line_size(2) # set default size to 2
        self._on_responsiveness(5) # set default responsiveness to 5
        
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
        if os.name=='nt' and os.getlogin()=='raeyo':
            filedlg.set_path('C:\data4')
        elif os.name=='posix' and os.getlogin()=='raeyo':
            filedlg.set_path('/media/raeyo/T7/Workspace/data4-source')
            
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
        
        self._object_transparency = gui.Slider(gui.Slider.DOUBLE)
        self._object_transparency.set_limits(0, 1)
        self._object_transparency.set_on_value_changed(self._on_object_transparency)
        grid.add_child(gui.Label("물체 투명도"))
        grid.add_child(self._object_transparency)
        
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
        
        self._auto_save_interval = gui.Slider(gui.Slider.INT)
        self._auto_save_interval.set_limits(1, 20) # 1-> 1e-3
        self._auto_save_interval.double_value = 5
        self._auto_save_interval.set_on_value_changed(self._on_auto_save_interval)
        grid.add_child(gui.Label("자동 저장 간격"))
        grid.add_child(self._auto_save_interval)

        viewctrl_layout.add_child(grid)
        
        self._settings_panel.add_child(viewctrl_layout)
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
        if self._check_geometry(self._scene_name):
            self._set_geometry_material(self._scene_name, self.settings.scene_material)
        self._scene_point_size.double_value = size
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
        
    def _on_hand_point_size(self, size):
        self.logger.debug('_on_hand_point_size')
        self._log.text = "\t 손 관절 사이즈 값을 변경합니다."
        self.window.set_needs_layout()
        self._last_change = time.time()

        for i in range(21):
            mat = self.settings.hand_joint_material[i]
            mat.point_size = int(size)
            for side in ['right', 'left']:
                if self._check_geometry(self._hand_joint_name.format(side, i)):
                    self._set_geometry_material(self._hand_joint_name.format(side, i), mat)
        self._hand_point_size.double_value = size
        if self._active_hand is not None:
            self._update_hand_mask()
            self._update_diff_viewer()
    def _on_hand_line_size(self, size):
        self.logger.debug('_on_hand_line_size')
        self._log.text = "\t 손 연결선 두께 값을 변경합니다."
        self.window.set_needs_layout()
        self._last_change = time.time()
        mat = self.settings.hand_link_material
        mat.line_width = int(size)
        for side in ['right', 'left']:
            if self._check_geometry(self._hand_link_name.format(side)):
                self._set_geometry_material(self._right_hand_link_name, mat)
        if self._active_hand is not None:
            mat = self.settings.active_link_material
            mat.line_width = int(size)
            active_side = self._active_hand.side
            if active_side == 'right':
                self._set_geometry_material(self._right_hand_link_name, mat)
            else:
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
    def _on_auto_save_interval(self, interval):
        self.logger.debug('_on_auto_save_interval')
        self._log.text = "\t 자동 저장 간격을 변경합니다."
        self.window.set_needs_layout()
        self._auto_save_interval.double_value = interval
    
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
    def _convert_hand(self):
        self.logger.debug('_convert_hand')
        if self._hands is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _convert_hand)")
            return 
        # convert active hand
        self._active_hand_idx = (self._active_hand_idx+1) % len(self._hands.keys())
        active_side = self._hand_names[self._active_hand_idx]
            
        self._active_hand = self._hands[active_side]
        self._convert_to_root()
        self._update_joint_mask()
        self._update_current_hand_visible()
        self._update_hand_mask()
        self._update_diff_viewer()
        self._update_current_hand_str()
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
        self._update_hand_mask()
        self._update_hand_layer()
        self._update_diff_viewer()
        self._update_current_hand_str()
    def _control_joint_up(self):
        self.logger.debug('_control_joint_up')
        if self._active_hand is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _control_joint_up)")
            return
        ctrl_idx = self._active_hand.control_idx + 1
        self._active_hand.set_control_joint(ctrl_idx)
        self._update_joint_mask()
        self._update_hand_layer()
        self._update_hand_mask()
        self._update_diff_viewer()
        self._update_current_hand_str()
    def _control_joint_down(self):
        self.logger.debug('_control_joint_down')
        if self._active_hand is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _control_joint_down)")
            return
        ctrl_idx = self._active_hand.control_idx - 1
        self._active_hand.set_control_joint(ctrl_idx)
        self._update_joint_mask()
        self._update_hand_layer()
        self._update_hand_mask()
        self._update_diff_viewer()
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
        self._update_progress_str()
        self._init_pcd_layer()
        self._init_obj_layer()
        self._init_hand_layer()
        self._on_change_camera_merge()
        
        
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
        button.set_on_clicked(self._on_load_label_button)
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
    def _on_load_label_button(self):
        self.logger.debug('_on_load_label_button')
        if not self._check_annotation_scene():
            return
        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "파일 선택",
                                self.window.theme)
        filedlg.add_filter(".npz", "라벨")
        filedlg.set_on_cancel(self._on_load_label_cancel)
        filedlg.set_on_done(self._on_load_label_done)
        self.window.show_dialog(filedlg)
    def _on_load_label_cancel(self):
        self.logger.debug('_on_load_label_cancel')
        self.window.close_dialog()
    def _on_load_label_done(self, file_path):
        self.logger.debug('_on_load_label_done')
        self._log.text = "\t라벨링 결과를 불러오는 중입니다."
        self.window.set_needs_layout()
        if 'hand' in file_path:
            ret = self.annotation_scene._load_hand_label(file_path)
        elif 'obj' in file_path:
            ret = self.annotation_scene._load_obj_label(file_path)
        else:
            ret = False
        if ret:
            self._log.text = "\t이전 라벨링 결과를 불러왔습니다."
            self._init_hand_layer()
            self._init_obj_layer()
            self._on_save_label()
            self.window.close_dialog()
            self._log.text = "\t 라벨링 대상 파일을 불러왔습니다."
        else:
            self._on_error("저장된 라벨이 없습니다. (error at _on_load_label_done)")
            self._log.text = "\t 올바른 파일 경로를 선택하세요."

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
        try:
            img = self._active_hand.get_joint_mask()
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = o3d.geometry.Image(img)
            self._joint_mask_proxy.set_widget(gui.ImageWidget(img))
        except:
            pass

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
        self._mask_mode.add_item(MaskMode.RGB_FINGER)
        self._mask_mode.add_item(MaskMode.MASK_ALL)
        self._mask_mode.add_item(MaskMode.MASK_FINGER)
        self._mask_mode.set_on_selection_changed(self._on_update_mode)
        h.add_child(self._mask_mode)
        self._images_panel.add_child(h)
        
        h = gui.Horiz(0.4 * em)
        self._mask_transparency = gui.Slider(gui.Slider.DOUBLE)
        self._mask_transparency.set_limits(0, 1)
        self._mask_transparency.double_value = 0.5
        self._last_mask_transparency = 0.5
        self._mask_transparency.set_on_value_changed(self._on_update_mask_transparency)
        h.add_child(gui.Label("마스크 투명도:"))
        h.add_child(self._mask_transparency)
        self._images_panel.add_child(h)


    def _init_show_error_layout(self):
        self.logger.debug('_init_show_error_layout')
        em = self.window.theme.font_size
        show_error_layout = gui.CollapsableVert("카메라 시점 조정", 0.33 * em,
                                                   gui.Margins(0.25 * em, 0, 0, 0))
        show_error_layout.set_is_open(True)
        
        self._view_label_layout_list = []
        
        for i in range(7):
            h = gui.Horiz(0)
            
            button = gui.Button("카메라 {}".format(i+1))
            button.set_on_clicked(self.__getattribute__("_on_change_camera_{}".format(i)))
            button.vertical_padding_em = 0.1
            h.add_child(button)
            h.add_child(gui.Label(" | \n | "))
            
            v = gui.Vert(0)
            left_vis_joint_num = gui.Label("왼손 : 00/21")
            v.add_child(left_vis_joint_num)
            right_vis_joint_num = gui.Label("오른손 : 00/21")
            v.add_child(right_vis_joint_num)
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
            self._view_label_layout_list.append((button, right_vis_joint_num, left_vis_joint_num,  box))

        show_error_layout.add_child(gui.Label("-"*60))

        h = gui.Horiz(0.4 * em)
        button = gui.Button("합친 상태")
        button.vertical_padding_em = 0.1
        button.set_on_clicked(self._on_change_camera_merge)
        h.add_child(button)

        self._activate_cam_txt = gui.Label("현재 활성화된 카메라: 없음")
        show_error_layout.add_child(self._activate_cam_txt)
        self._validation_panel.add_child(show_error_layout)
    def _on_change_camera_0(self):
        self.logger.debug('_on_change_camera_0')
        if not self._check_annotation_scene():
            return
        self._camera_idx = 0
        self._frame.convert_camera(self._view_label_layout_list[self._camera_idx][0].text)
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_label_layout_list[self._camera_idx][0].text)
    def _on_change_camera_1(self):
        self.logger.debug('_on_change_camera_1')
        if not self._check_annotation_scene():
            return
        self._camera_idx = 1
        self._frame.convert_camera(self._view_label_layout_list[self._camera_idx][0].text)
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_label_layout_list[self._camera_idx][0].text)
    def _on_change_camera_2(self):
        self.logger.debug('_on_change_camera_2')
        if not self._check_annotation_scene():
            return
        self._camera_idx = 2
        self._frame.convert_camera(self._view_label_layout_list[self._camera_idx][0].text)
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_label_layout_list[self._camera_idx][0].text)
    def _on_change_camera_3(self):
        self.logger.debug('_on_change_camera_3')
        if not self._check_annotation_scene():
            return
        self._camera_idx = 3
        self._frame.convert_camera(self._view_label_layout_list[self._camera_idx][0].text)
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_label_layout_list[self._camera_idx][0].text)
    def _on_change_camera_4(self):
        self.logger.debug('_on_change_camera_4')
        if not self._check_annotation_scene():
            return
        self._camera_idx = 4
        self._frame.convert_camera(self._view_label_layout_list[self._camera_idx][0].text)
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_label_layout_list[self._camera_idx][0].text)
    def _on_change_camera_5(self):
        self.logger.debug('_on_change_camera_5')
        if not self._check_annotation_scene():
            return
        self._camera_idx = 5
        self._frame.convert_camera(self._view_label_layout_list[self._camera_idx][0].text)
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_label_layout_list[self._camera_idx][0].text)
    def _on_change_camera_6(self):
        self.logger.debug('_on_change_camera_6')
        if not self._check_annotation_scene():
            return
        self._camera_idx = 6
        self._frame.convert_camera(self._view_label_layout_list[self._camera_idx][0].text)
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_label_layout_list[self._camera_idx][0].text)
    def _on_change_camera_7(self):
        self.logger.debug('_on_change_camera_7')
        if not self._check_annotation_scene():
            return
        self._camera_idx = 7
        self._frame.convert_camera(self._view_label_layout_list[self._camera_idx][0].text)
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: {}".format(self._view_label_layout_list[self._camera_idx][0].text)
    def _on_change_camera_merge(self):
        self.logger.debug('_on_change_camera_merge')
        if not self._check_annotation_scene():
            return
        self._camera_idx = -1
        self._frame.convert_camera("merge")
        for but, _, _, bbox in self._view_label_layout_list:
            bbox.checked = True
        self._update_pcd_layer()
        self._on_change_camera()
        self._activate_cam_txt.text = "현재 활성화된 카메라: 합쳐진 뷰"
    def _on_change_camera(self):
        self._update_hand_mask()
        self._reset_image_viewer()
        self._update_image_viewer()
        self._update_diff_viewer()
        self._update_current_hand_visible()
        self._update_joint_mask()
        self._update_hand_layer()
        self._on_active_camera_viewpoint()
    def _on_change_bbox(self, visible):
        self.logger.debug('_on_change_bbox')
        if not self._check_annotation_scene():
            return
        self._update_pcd_layer()
    def _get_activate_cam(self):
        self.logger.debug('_get_activate_cam')
        cam_list = []
        for but, _, _, bbox in self._view_label_layout_list:
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
        for i, (_, _, _, bbox) in enumerate(self._view_label_layout_list):
            if i == idx:
                bbox.checked = True
            else:
                bbox.checked = False
        self._update_pcd_layer()
    def _init_cam_name(self):
        self.logger.debug('_init_cam_name')
        self._cam_name_list = list(self.annotation_scene._cameras.keys())
        for idx, (cam_button, _, _, _) in enumerate(self._view_label_layout_list):
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
    def _update_diff_viewer(self):
        self.logger.debug('_update_diff_viewer')
        if self._camera_idx == -1:
            self._diff_proxy.set_widget(gui.ImageWidget())
            return
        current_cam = self._cam_name_list[self._camera_idx]
        try:
            diff_img = self._generate_valid_image(current_cam)
        except Exception as e:
            print(e)
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
    def _on_update_mask_transparency(self, value):
        if not self._check_annotation_scene():
            return
        if abs(value - self._last_mask_transparency) > 0.01:
            self._last_mask_transparency = value
            self._update_diff_viewer()
        
    def _update_obj_mask(self):
        self.logger.debug('Start _update_obj_mask')
        self.logger.debug('\tset hl renderer')
        
        self.hl_renderer.reset()    
        self.hl_renderer.add_objects(self._objects, color=[1, 0, 0])

        # rendering depth for each camera
        self.logger.debug('\trendering depth for each camera')
        self._obj_mask = {}
        
        for error_layout in self._view_label_layout_list:
            cam_name = error_layout[0].text
            intrinsic = self._frame.cameras[cam_name].intrinsic
            extrinsic = self._frame.cameras[cam_name].extrinsics
            
            self.hl_renderer.set_camera(intrinsic, extrinsic, self.W, self.H)
            
            rgb_rendered = self.hl_renderer.render_rgb()
            rgb_rendered = np.array(rgb_rendered)
            depth_rendered = self.hl_renderer.render_depth()
            depth_rendered = np.array(depth_rendered, dtype=np.float32)
            depth_rendered[np.isposinf(depth_rendered)] = 0
            
            # object mask
            object_mask = np.bitwise_and(rgb_rendered[:, :, 0]>10, np.bitwise_and(rgb_rendered[:, :, 1]<2, rgb_rendered[:, :, 2]<2))
            rgb_rendered[~object_mask] = [0,0,0]
            
            rgb_captured = self._frame.get_rgb(cam_name)
            
            self._obj_mask[cam_name] = {
                "rgb_captured": rgb_captured,
                "object_mask": rgb_rendered,
                "depth_rendered": depth_rendered
            }
        # clear geometry
        self._log.text = "\t라벨링 검증용 이미지를 생성했습니다."
        self.window.set_needs_layout()
        self.logger.debug('End _update_valid_error')
    def _update_hand_mask(self):
        self.logger.debug('Start _update_hand_mask')
        self._joint_mask = {}
        for error_layout in self._view_label_layout_list:
            cam_name = error_layout[0].text
            if cam_name != self._frame.active_cam:
                continue
            intrinsic = self._frame.cameras[cam_name].intrinsic
            extrinsic = self._frame.cameras[cam_name].extrinsics
            # add hand joint mask
            joints = self._active_hand.get_current_joints() # unit is m
            
            # convert 3d world xyz to 2d image xy
            inv_SE = np.linalg.inv(extrinsic)
            joint_z = (inv_SE[:3] @ np.concatenate((joints, np.ones((joints.shape[0], 1))), axis=1).T)[2, :]
            joint_2d = intrinsic @ inv_SE[:3] @ np.concatenate((joints, np.ones((joints.shape[0], 1))), axis=1).T
            joint_2d = joint_2d[:2, :] / joint_2d[2, :]
            joint_2d = joint_2d.astype(np.int32)
            joint_2d = joint_2d.T
            valid_joint_2d = np.bitwise_and(joint_2d[:, 0]>=0, np.bitwise_and(joint_2d[:, 0]<self.W, np.bitwise_and(joint_2d[:, 1]>=0, joint_2d[:, 1]<self.H)))
            
            self._joint_mask[cam_name] = {
                "joint_2d": joint_2d,
                "valid_joint_2d": valid_joint_2d,
                "joint_z": joint_z
            }
    
    def _generate_valid_image(self, cam_name):
        self.logger.debug('Start _generate_valid_image')
        mode = self._mask_mode.selected_text

        rgb_captured = self._obj_mask[cam_name]['rgb_captured']
        object_mask  = self._obj_mask[cam_name]['object_mask']
        depth_rendered = self._obj_mask[cam_name]['depth_rendered']
        joint_2d  = self._joint_mask[cam_name]['joint_2d']
        valid_joint_2d  = self._joint_mask[cam_name]['valid_joint_2d']
        joint_z = self._joint_mask[cam_name]['joint_z']

        target_idx = [0]
        target = self._active_hand.optimize_target
        if "전체" in mode:
            target_idx += list(range(1, 21))
        elif "손가락" in mode:
            if target=="root":
                pass
            elif target=="thumb":
                target_idx += list(range(1, 5))
            elif target=="fore":
                target_idx += list(range(5, 9))
            elif target=="middle":
                target_idx += list(range(9, 13))
            elif target=="ring":
                target_idx += list(range(13, 17))
            elif target=="little":
                target_idx += list(range(17, 21))

        joint_mask = copy.deepcopy(object_mask)
        for idx in target_idx:
            if not valid_joint_2d[idx]: continue
            
            if not self._active_hand.current_visible[idx]: continue
            
            # check depth with padding 5 consider H, W
            x, y = joint_2d[idx]
            x_min = max(0, x-5)
            x_max = min(self.W, x+5)
            y_min = max(0, y-5)
            y_max = min(self.H, y+5)
            depth_mask = depth_rendered[y_min:y_max, x_min:x_max]
            try:
                mean_depth = np.sum(depth_mask)/np.count_nonzero(depth_mask)
            except:
                mean_depth = np.inf
            joint_depth = joint_z[idx]
            if joint_depth > mean_depth:
                continue
            joint_mask = cv2.circle(joint_mask, tuple(joint_2d[idx]), int(self._hand_point_size.double_value), self._active_hand._COLOR_OF_JOINT[idx], -1)
            # control idx
            if idx == self._active_hand.control_joint:
                joint_mask = cv2.circle(joint_mask, tuple(joint_2d[idx]), int(self._hand_point_size.double_value*1.5), [0, 255, 0], 5)
        joint_mask = cv2.cvtColor(joint_mask, cv2.COLOR_BGR2RGB)
        image = joint_mask
        if "RGB" in mode:
            val = self._mask_transparency.double_value
            bool_mask = np.bitwise_or(joint_mask[:,:,0]>0, np.bitwise_or(joint_mask[:,:,1]>0, joint_mask[:,:,2]>0))
            
            
            image = copy.deepcopy(rgb_captured)
            image[bool_mask] = image[bool_mask]*(val) + joint_mask[bool_mask]*(1-val)
            
        return image
            
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
    def _on_mouse(self, event):
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
                        self._move_hand_translation(world_xyz)
                        
                    gui.Application.instance.post_to_main_thread(
                        self.window, move_joint)
                
            self._scene.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.CONSUMED
        
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.SHIFT) and event.is_button_down(gui.MouseButton.RIGHT):
            self._control_joint_up()
            self.logger.debug("convert joint {}".format(self._active_hand.get_control_joint_name()))
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.CTRL) and event.is_button_down(gui.MouseButton.RIGHT):
            self._control_joint_down()
            self.logger.debug("convert joint {}".format(self._active_hand.get_control_joint_name()))
        
        return gui.Widget.EventCallbackResult.IGNORED
    
    def _depth_callback(self, depth_image):
        self._depth_image = np.asarray(depth_image)
    
    def move(self, x, y, z, rx, ry, rz):
        self.logger.debug('move')
        self._log.text = "{} 라벨 이동 중입니다.".format(self._active_hand.get_control_joint_name())
        self.window.set_needs_layout()
        self._last_change = time.time()
        
        current_xyz = self._active_hand.get_control_position()
        # convert x, y, z cam to world
        R = self._scene.scene.camera.get_view_matrix()[:3,:3]
        R_inv = np.linalg.inv(R)
        xyz = np.dot(R_inv, np.array([x, y, z]))
        xyz = current_xyz + xyz
        
        self._move_hand_translation(xyz)
        self._update_hand_mask()
        self._update_diff_viewer()
            
    
    # move hand
    def _move_hand_translation(self, xyz):
        self._active_hand.set_control_position(xyz)
        self._annotation_changed = True
        self._update_hand_layer()
    
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
            self._add_geometry(self._scene_name, self._pcd, self.settings.scene_material)
            self._inlier_points = []
        else:
            self._remove_geometry(self._scene_name)
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
        # update object mask
        self._update_obj_mask()
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
        self._active_hand.set_optimize_target('root')
        self._on_change_camera_merge()
        
    def _update_hand_layer(self):
        if not self._check_annotation_scene():
            return
        
        active_side = self._active_hand.side
        
        for side, hand in self._hands.items():
            joints = hand.get_current_joints()
            # add joints
            for i, xyz in enumerate(joints):
                geo = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([xyz]))
                mat = self.settings.hand_joint_material[i]
                self._add_geometry(self._hand_joint_name.format(side, i), geo, mat)

            
            if side==active_side:
                # add link
                link_geo = hand.get_links()
                mat = self.settings.active_link_material
                self._add_geometry(self._hand_link_name.format(side), link_geo, mat)
                
                active_geo = hand.get_active_geometry()
                
                mat = self.settings.active_target_joint_material
                self._add_geometry(self._active_joint_name, active_geo['joint'], mat)
                mat = self.settings.control_target_joint_material
                self._add_geometry(self._control_joint_name, active_geo['control'], mat)
                
            else:
                # add link
                link_geo = hand.get_links()
                mat = self.settings.hand_link_material
                self._add_geometry(self._hand_link_name.format(side), link_geo, mat)
    
        self._update_hand_mask()
    def _update_current_hand_visible(self):
        if not self._check_annotation_scene():
            return
        for but, r, l, _ in self._view_label_layout_list:
            state = self._frame.get_visible_state(but.text)
            for side, vis_state in state.items():
                if side=='left':
                    l.text = "왼손: {:02d}/21".format(sum(vis_state))
                elif side=='right':
                    r.text = "오른손: {:02d}/21".format(sum(vis_state))
    
    def _toggle_hand_visible(self):
        self.logger.debug('_toggle_hand_visible')
        show = self._show_hands.checked
        self._show_hands.checked = not show
        self._on_show_hand(not show)

    def _on_key(self, event):
        if self._active_hand is None:
            return gui.Widget.EventCallbackResult.IGNORED

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
        
        
        # activate autosave
        if event.key==gui.KeyName.Z and event.type==gui.KeyEvent.DOWN:
            self._toggle_hand_visible()
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
            return gui.Widget.EventCallbackResult.HANDLED
        
        # save label
        if event.key==gui.KeyName.F and event.type==gui.KeyEvent.DOWN:
            self._on_save_label()
            return gui.Widget.EventCallbackResult.HANDLED
        
        # convert hand
        if (event.key == gui.KeyName.TAB) and (event.type==gui.KeyEvent.DOWN):
            self._convert_hand()
            return gui.Widget.EventCallbackResult.CONSUMED
        
        # reset hand pose
        if event.key == gui.KeyName.R and event.type==gui.KeyEvent.DOWN:
            return gui.Widget.EventCallbackResult.CONSUMED
            #TODO
            if self.reset_flat:
                self._active_hand.reset_pose(flat_hand=True)
            else:
                self._active_hand.reset_pose()
            self._update_hand_layer()
            return gui.Widget.EventCallbackResult.CONSUMED
        
        if event.key == gui.KeyName.X and (event.type==gui.KeyEvent.DOWN):
            self._active_hand.toggle_current_joint_visible()
            self._annotation_changed = True
            self._update_current_hand_visible()
            self._update_hand_mask()
            self._update_diff_viewer()
            self._update_joint_mask()
            
            return gui.Widget.EventCallbackResult.HANDLED
        
        # convert finger
        is_converted_finger = True
        if event.key == gui.KeyName.BACKTICK:
            self._convert_to_root()
        elif event.key == gui.KeyName.ONE and (event.type==gui.KeyEvent.DOWN):
            self._convert_to_thumb()
        elif event.key == gui.KeyName.TWO and (event.type==gui.KeyEvent.DOWN):
            self._convert_to_fore()
        elif event.key == gui.KeyName.THREE and (event.type==gui.KeyEvent.DOWN):
            self._convert_to_middle()
        elif event.key == gui.KeyName.FOUR and (event.type==gui.KeyEvent.DOWN):
            self._convert_to_ring()
        elif event.key == gui.KeyName.FIVE and (event.type==gui.KeyEvent.DOWN):
            self._convert_to_little()
        else:
            is_converted_finger = False
        
        is_convert_joint = True
        if event.key == gui.KeyName.PAGE_UP and (event.type==gui.KeyEvent.DOWN):
            self._control_joint_up()
        elif event.key == gui.KeyName.PAGE_DOWN and (event.type==gui.KeyEvent.DOWN):
            self._control_joint_down()
        else:
            is_convert_joint = False
        
        if is_converted_finger or is_convert_joint:
            self.logger.debug("convert joint {}".format(self._active_hand.get_control_joint_name()))
            self._update_hand_layer()
            self._update_joint_mask()
            return gui.Widget.EventCallbackResult.CONSUMED

        # Translation
        if event.type!=gui.KeyEvent.UP:
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
        return gui.Widget.EventCallbackResult.IGNORED
        
 
    def _on_tick(self):
        if (time.time()-self._last_change) > 1:
            if self._active_hand is None:
                self._log.text = "\t라벨링 대상 파일을 선택하세요."
                self.window.set_needs_layout()
            else:
                self._log.text = "{} 라벨링 중입니다.".format(self._active_hand.get_control_joint_name())
                self.window.set_needs_layout()
        
        if self._auto_save.checked and self.annotation_scene is not None:
            if (time.time()-self._last_saved) > self._auto_save_interval.double_value and self._annotation_changed:
                self.logger.debug('auto saving')
                self._annotation_changed = False
                self.annotation_scene.save_label()
                self._last_saved = time.time()
                self._log.text = "라벨 결과 자동 저장중입니다."
                self.window.set_needs_layout()
        
        
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