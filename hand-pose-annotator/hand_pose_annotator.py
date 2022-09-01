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

import torch
from manopth.manolayer import ManoLayer
from torch import optim

import numpy as np


import yaml
import time
import json
import datetime
import shutil
from scipy.spatial.transform import Rotation

MANO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "mano")
hangeul = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "NanumGothic.ttf")


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
        
    def reset(self, shape_param=None):
        #1. shape
        if shape_param is None:
            pass
        else:
            self.shape_param = torch.Tensor(shape_param).unsqueeze(0)

        #2. root translation
        self.root_trans = torch.zeros(1, 3)
        
        #3. root, 15 joints
        self.joint_rot = [torch.zeros(1, 3) for _ in range(16)]
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
                    self.joint_rot[target_idx] = torch.zeros(1, 3)
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
            'pose_param': np.array(pose_param.cpu().detach()[0, :]), # 16, 3, 3
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
        lineset = o3d.geometry.LineSet.create_from_triangle_mesh(tri_mesh)
        
        return lineset
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
    
class Camera:
    # DexYCB Toolkit
    # Copyright (C) 2021 NVIDIA Corporation
    # Licensed under the GNU General Public License v3.0 [see LICENSE for details]
    
    def __init__(self, name, intrinsic, extrinsic, device="cpu"):
        self.name = name
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self._device = device
        
        x = self.intrinsic
        self._K = torch.tensor([[x['fx'], 0.0, x['ppx']], [0.0, x['fy'], x['ppy']], [0.0, 0.0, 1.0]],
                               dtype=torch.float32,
                               device=self._device)
        self._K_inv = torch.inverse(self._K)
        
        self._T = torch.tensor(self.extrinsic, 
                               dtype=torch.float32,
                               device=self._device).view(3, 4)
        self._R = self._T[:, :3]
        self._t = self._T[:, 3]
        self._R_inv = torch.inverse(self._R)
        self._t_inv = torch.mv(self._R_inv, self._t)
              
class Dataset:
    def print_dataset_info(self):
        total_sub = 0
        total_scene = 0
        total_frame = 0
        for sub, sub_data in self._scene_meta.items():
            total_sub += 1
            for scene, meta in sub_data.items():
                total_scene += 1
                frame_num = meta['num_frames']
                total_frame += frame_num
                print("Subject: {}| Scene: {}| Frame: {}".format(sub, scene, frame_num))
        
        print("Total\nSubject: {}\nScene: {}\nFrame: {}".format(total_sub, total_scene, total_frame)) 
        
    def get_extrinsic(self, extrinsic_id, camera_id):
        extr_file = os.path.join(self._calib_dir, "extrinsics_{}".format(extrinsic_id), "extrinsics.yml")
        with open(extr_file, 'r') as f:
            extr = yaml.load(f, Loader=yaml.FullLoader)
        return extr['extrinsics'][camera_id]

    def get_mano_calib(self, mano_id):
        mano_file = os.path.join(self._calib_dir, "mano_{}".format(mano_id), "mano.yml")
        with open(mano_file, 'r') as f:
            mano_calib = yaml.load(f, Loader=yaml.FullLoader)
        return mano_calib['betas']
    
    def get_scene_from_file(self, file_path):
        subject_id, scene_id, camera_id, frame_id = self.path_to_info(file_path)
        scene_meta = self._scene_meta[subject_id][scene_id]
        self.current_scene_idx = self._total_scene.index((subject_id, scene_id))
        self.current_scene_file = scene_id
        self.current_frame_file = frame_id
        scene_dir = os.path.join(self._data_dir, subject_id, scene_id, camera_id)
        if camera_id=="merge":
            camera = None
        else:
            assert False, "error on get_scene_from_file"
            camera = Camera(camera_id, self._intrinsics[camera_id],self.get_extrinsic(scene_meta['extrinsics'], camera_id))
        
        calib = scene_meta['mano_calib'][0]
        for hand_model in self.hand_models.values():
            hand_model.reset(self.get_mano_calib(calib))
            
        return Scene(scene_dir=scene_dir, 
                     camera=camera, 
                     hands=self.hand_models, 
                     total_frame=scene_meta['num_frames'], 
                     current_frame=frame_id)
    
    def get_current_file(self):
        return "작업 폴더: {}\n현재 파일: {:06}".format(self.current_scene_file, self.current_frame_file)
        
    def get_progress(self):
        return "작업 폴더: [{}/{}]".format(self.current_scene_idx, len(self._total_scene))
    
    def get_next_scene(self):
        scene_idx = self.current_scene_idx + 1
        if scene_idx > len(self._total_scene) -1:
            return None
        else:
            self.current_scene_idx = scene_idx    
            file_path = self.get_scene_first_file()
            return self.get_scene_from_file(file_path)
    
    def get_previous_scene(self):
        scene_idx = self.current_scene_idx - 1
        if scene_idx < 0:
            return None
        else:
            self.current_scene_idx = scene_idx    
            file_path = self.get_scene_first_file()
            return self.get_scene_from_file(file_path)
    
    def get_scene_first_file(self):
        sub_id, scene_id = self._total_scene[self.current_scene_idx]
        scene_dir = os.path.join(self._data_dir, sub_id, scene_id, 'merge')
        assert len(scene_dir) > 0
        scene_files = os.listdir(scene_dir)
        scene_files.sort()
        frame_file = scene_files[0]
        
        return os.path.join(scene_dir, frame_file)
        
    def check_same_data(self, file_path):
        camera_dir = os.path.dirname(file_path)
        scene_dir = os.path.dirname(camera_dir)
        subject_dir = os.path.dirname(scene_dir)
        dataset_dir = os.path.dirname(subject_dir)
        return self._data_dir == dataset_dir

    @staticmethod
    def path_to_info(file_path):
        camera_dir = os.path.dirname(file_path)
        scene_dir = os.path.dirname(camera_dir)
        subject_dir = os.path.dirname(scene_dir)
        
        frame_id = int(os.path.splitext(os.path.basename(file_path))[0].split("_")[-1])
        camera_id = os.path.basename(camera_dir)
        scene_id = os.path.basename(scene_dir)
        subject_id = os.path.basename(subject_dir)

        return subject_id, scene_id, camera_id, frame_id

class DexYCBSample(Dataset):
    """Dex-YCB"""
    _SUBJECTS = [
        '20200709-subject-01',
        '20200813-subject-02',
        '20200820-subject-03',
        '20200903-subject-04',
        '20200908-subject-05',
    ]
    _SERIALS = [
        '836212060125',
        '839512060362',
        '840412060917',
        '841412060263',
        '932122060857',
        '932122060861',
        '932122061900',
        '932122062010',
    ]
    def __init__(self, data_root):
        # Total Data Statics
        self._data_dir = data_root
        self._calib_dir = os.path.join(self._data_dir, "calibration")
    
        self._h = 480
        self._w = 640
        self._subjects = self._SUBJECTS
        self._total_scene = []
        self._scene_meta = {}
        for sub in self._subjects:
            # for each subject 100 sequence for dex-ycb
            seq = sorted(os.listdir(os.path.join(self._data_dir, sub)))
            self._scene_meta[sub] = {s: {} for s in seq}
            # for each seq
            for s in seq:
                meta_file = os.path.join(self._data_dir, sub, s, "meta.yml")
                with open(meta_file, 'r') as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)
                self._scene_meta[sub][s] = meta
                self._total_scene.append((sub, s))
        
        self.print_dataset_info()
        self.hand_models = {
            "right": HandModel(side='right'),
            "left": HandModel(side='left')
        }
        self.current_scene_idx = -1

def load_dataset_from_file(file_path):
    # initialize dataset directory # same as dex-ycb
    camera_dir = os.path.dirname(file_path)
    scene_dir = os.path.dirname(camera_dir)
    subject_dir = os.path.dirname(scene_dir)
    dataset_dir = os.path.dirname(subject_dir)
    
    if os.path.basename(dataset_dir)=='dex-ycb-sample':
        return DexYCBSample(dataset_dir)
    else:
        raise NotImplementedError



class Scene:
    
    class Frame:
        def __init__(self, frame_idx, scene_pcd, hands, objs):
            self.id = frame_idx
            self.scene_pcd = scene_pcd
            
            self.hands = hands
            self.objs = objs
    
    def __init__(self, scene_dir, camera, hands, total_frame, current_frame):
        self._scene_dir = scene_dir
        self.camera = camera
        self._hands = hands
        self.total_frame = total_frame
        self.frame_id = current_frame
        
        self._data_format = "points_{:06d}.pcd"
        self._label_format = "hand_labels_{:06d}.npz"
        
        self._json_format = "labels_{:06d}.json"
        self._label = None
        self._previous_label = None

    def _load_frame(self):
        try:
            pcd = self._load_point_cloud(self._frame_path)
            self.current_frame = Scene.Frame(frame_idx=self.frame_id,
                                             scene_pcd=pcd,
                                             hands=self._hands,
                                             objs=[])
        except:
            print("Fail to load point cloud")
            self.current_frame = None
    
    def get_current_frame(self):
        try:
            self._load_frame()
        except:
            print("Fail to get Frame")
        return self.current_frame
    
    def moveto_next_frame(self):
        frame_id = self.frame_id + 1
        if frame_id > self.total_frame - 1:
            return False
        else:
            self.frame_id = frame_id
            if not os.path.isfile(self._frame_path):
                self.frame_id = frame_id - 1
                return False
            return True
    
    def moveto_previous_frame(self):
        frame_id = self.frame_id - 1
        if frame_id < 0:
            return False
        else:
            self.frame_id = frame_id
            if not os.path.isfile(self._frame_path):
                self.frame_id = frame_id + 1
                return False
            return True

    def get_progress(self):
        return "현재 파일: [{}/{}]".format(self.frame_id, self.total_frame)

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
        if self._label is not None:
            self._previous_label = self._label.copy()
        try:
            self._label = dict(np.load(self._label_path))
        except:
            print("Fail to load Label")
            for hand_model in self._hands.values():
                hand_model.reset()
            return False
        hand_states = {}
        for k, v in self._label.items():
            side = k.split('_')[0]
            hand_states.setdefault(side, {})
            param = k.replace(side + "_", "")
            hand_states[side][param] = v
        for side, state in hand_states.items():
            self._hands[side].set_state(state)
        return True

    def load_previous_label(self):
        if self._previous_label is None:
            return False
        try:
            hand_states = {}
            for k, v in self._previous_label.items():
                side = k.split('_')[0]
                hand_states.setdefault(side, {})
                param = k.replace(side + "_", "")
                hand_states[side][param] = v
            for side, state in hand_states.items():
                self._hands[side].set_state(state, only_pose=True)
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
    def _label_path(self):
        return os.path.join(self._scene_dir, self._label_format.format(self.frame_id))

class Settings:
    SHADER_POINT = "defaultUnlit"
    SHADER_LINE = "unlitLine"

    def __init__(self):
        self.bg_color = gui.Color(1, 1, 1)
        self.show_axes = False
        self.show_coord_frame = False
        self.show_hand = True
        self.transparency = 0.5

        # ----- Material Settings -----
        self.apply_material = True  # clear to False after processing

        # ----- scene material
        self.scene_material = rendering.MaterialRecord()
        self.scene_material.base_color = [1.0, 1.0, 1.0, 1.0]
        self.scene_material.shader = Settings.SHADER_POINT
        self.scene_material.point_size = 1.0

        # ----- hand model setting
        self.hand_mesh_material = rendering.MaterialRecord()
        self.hand_mesh_material.base_color = [0.8, 0.8, 0.8, 1.0]
        self.hand_mesh_material.shader = Settings.SHADER_LINE
        self.hand_mesh_material.line_width = 2.0
        
        self.hand_joint_material = rendering.MaterialRecord()
        self.hand_joint_material.base_color = [1.0, 0.0, 0.0, 1.0]
        self.hand_joint_material.shader = Settings.SHADER_POINT
        self.hand_joint_material.point_size = 5.0
        
        self.hand_link_material = rendering.MaterialRecord()
        self.hand_link_material.base_color = [1.0, 0.0, 0.0, 1.0]
        self.hand_link_material.shader = Settings.SHADER_LINE
        self.hand_link_material.line_width = 2.0
        
        self.active_hand_mesh_material = rendering.MaterialRecord()
        self.active_hand_mesh_material.base_color = [0.0, 1.0, 0.75, 0.5]
        self.active_hand_mesh_material.shader = Settings.SHADER_LINE
        self.active_hand_mesh_material.line_width = 2.0

        # ----- hand label setting
        self.target_joint_material = rendering.MaterialRecord()
        self.target_joint_material.base_color = [0.0, 0.0, 1.0, 1.0]
        self.target_joint_material.shader = Settings.SHADER_POINT
        self.target_joint_material.point_size = 10.0
        
        self.target_link_material = rendering.MaterialRecord()
        self.target_link_material.base_color = [0.0, 0.0, 1.0, 1.0]
        self.target_link_material.shader = Settings.SHADER_LINE
        self.target_link_material.line_width = 3.0
        
        self.active_target_joint_material = rendering.MaterialRecord()
        self.active_target_joint_material.base_color = [1.0, 0.75, 0.75, 1.0]
        self.active_target_joint_material.shader = Settings.SHADER_POINT
        self.active_target_joint_material.point_size = 20.0
        
        self.active_target_link_material = rendering.MaterialRecord()
        self.active_target_link_material.base_color = [0.0, 0.7, 0.0, 1.0]
        self.active_target_link_material.shader = Settings.SHADER_LINE
        self.active_target_link_material.line_width = 5.0
        
        self.control_target_joint_material = rendering.MaterialRecord()
        self.control_target_joint_material.base_color = [0.0, 1.0, 0.0, 1.0]
        self.control_target_joint_material.shader = Settings.SHADER_POINT
        self.control_target_joint_material.point_size = 30.0
        
        self.coord_material = rendering.MaterialRecord()
        self.coord_material.base_color = [1.0, 1.0, 1.0, 1.0]
        self.coord_material.shader = Settings.SHADER_LINE
        self.coord_material.point_size = 2.0
        
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
        self._hands = None
        self._active_hand = None
        self.upscale_responsiveness = False
        self._left_shift_modifier = False
        self._annotation_changed = False
        self._last_change = time.time()
        self._last_saved = time.time()
        self.coord_labels = []
        
        self.window = gui.Application.instance.create_window(self._window_name, width, height)
        w = self.window
        
        self.settings = Settings()
        
        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        
        #region ---- Settings panel
        em = w.theme.font_size
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        
        self._init_fileeidt_layout()
        self._init_viewctrl_layout()
        self._init_handedit_layout()
        self._init_stageedit_layout()
        self._init_scene_control_layout()
        
        
        # ---- log panel
        self._log_panel = gui.VGrid(1, em)
        self._log = gui.Label("\t 라벨링 대상 파일을 선택하세요. ")
        self._log_panel.add_child(self._log)
        
        
        # 3D Annotation tool options
        w.add_child(self._scene)
        
        w.add_child(self._settings_panel)
        w.add_child(self._log_panel)
        
        w.set_on_layout(self._on_layout)
        
        # ---- annotation tool settings ----
        self._initialize_background()
        self._on_scene_point_size(5) # set default size to 1
        self._on_scene_transparency(0)
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

        width_obj = 1.5 * width_set
        height_obj = 1.5 * layout_context.theme.font_size
        self._log_panel.frame = gui.Rect(r.get_right() - width_set - width_obj, r.y, width_obj, height_obj) 
    
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
        center = np.array([0, 0, 0])
        eye = center + np.array([0, 0, -0.5])
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
        filedlg.add_filter(".pcd", "포인트 클라우드")
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
            # if already initialized
            if self.dataset is None:
                self.dataset = load_dataset_from_file(file_path)
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
            self._load_scene()
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
        
        # self._show_axes = gui.Checkbox("카메라 좌표계 보기")
        # self._show_axes.set_on_checked(self._on_show_axes)
        # viewctrl_layout.add_child(self._show_axes)

        # self._show_coord_frame = gui.Checkbox("조작 중인 조인트 좌표계 보기")
        # self._show_coord_frame.set_on_checked(self._on_show_coord_frame)
        # viewctrl_layout.add_child(self._show_coord_frame)

        self._show_hands = gui.Checkbox("손 라벨 보기 (Z)")
        self._show_hands.set_on_checked(self._on_show_hand)
        viewctrl_layout.add_child(self._show_hands)
        self._show_hands.checked = True

        self._auto_optimize = gui.Checkbox("자동 정렬 활성화 (X)")
        viewctrl_layout.add_child(self._auto_optimize)
        self._auto_optimize.checked = False

        self._auto_save = gui.Checkbox("자동 저장 활성화")
        viewctrl_layout.add_child(self._auto_save)
        self._auto_save.checked = True
        
        grid = gui.VGrid(2, 0.25 * em)
        self._scene_point_size = gui.Slider(gui.Slider.INT)
        self._scene_point_size.set_limits(1, 20)
        self._scene_point_size.set_on_value_changed(self._on_scene_point_size)
        grid.add_child(gui.Label("포인트 크기"))
        grid.add_child(self._scene_point_size)
        
        # self._scene_transparency = gui.Slider(gui.Slider.DOUBLE)
        # self._scene_transparency.set_limits(0, 1)
        # self._scene_transparency.set_on_value_changed(self._on_scene_transparency)
        # grid.add_child(gui.Label("포인트 투명도"))
        # grid.add_child(self._scene_transparency)
        
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
    def _on_scene_transparency(self, transparency):
        self._log.text = "\t 투명도 값을 변경합니다."
        self.window.set_needs_layout()
        self._last_change = time.time()
        self.settings.transparency = transparency
        if self.annotation_scene is None:
            return
        mat = self.settings.scene_material
        mat.base_color = [1.0, 1.0, 1.0, 1.0]
        self._set_geometry_material(self._scene_name, mat)
        self._scene_transparency.double_value = transparency
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
        label = gui.Label("관절이동:")
        h.add_child(label)
        button = gui.Button("이전 (PgDn)")
        button.horizontal_padding_em = 0.3
        button.vertical_padding_em = 0
        button.set_on_clicked(self._control_joint_down)
        h.add_child(button)
        button = gui.Button("다음 (PgUp)")
        button.horizontal_padding_em = 0.3
        button.vertical_padding_em = 0
        button.set_on_clicked(self._control_joint_up)
        h.add_child(button)
        handedit_layout.add_child(h)

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

        button = gui.Button("손 바꾸기 (Tab)")
        button.set_on_clicked(self._convert_hand)
        handedit_layout.add_child(button)
        
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
        
        button = gui.Button("라벨링 결과 저장하기 (F)")
        button.set_on_clicked(self._on_save_label)
        scene_control_layout.add_child(button)
        button = gui.Button("이전 라벨 불러오기")
        button.set_on_clicked(self._on_load_previous_label)
        scene_control_layout.add_child(button)
        
        
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
            self.annotation_scene.load_label()
        except:
            self._log.text = "\t 저장된 라벨이 없습니다."
            pass
        self._update_progress_str()
        # visualize scene pcd
        pcd = self._frame.scene_pcd
        self.bounds = pcd.get_axis_aligned_bounding_box()
        self._on_initial_viewpoint()
        self._add_geometry(self._scene_name, pcd, self.settings.scene_material)
        self._init_hand_layer()

    def _update_progress_str(self):
        self._current_progress_str.text = self.dataset.get_current_file()
        self._current_file_pg.text = self.annotation_scene.get_progress()
        self._current_scene_pg.text = self.dataset.get_progress()
    
    def _on_previous_frame(self):
        if self._check_changes():
            return
        if self.annotation_scene is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_previous_frame)")
            return
        if not self.annotation_scene.moveto_previous_frame():
            self._on_error("이전 포인트 클라우드가 존재하지 않습니다.")
            return
        self._log.text = "\t 이전 포인트 클라우드로 이동했습니다."
        self._load_scene()
    def _on_next_frame(self):
        if self._check_changes():
            return
        if self.annotation_scene is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_previous_frame)")
            return
        if not self.annotation_scene.moveto_next_frame():
            self._on_error("다음 포인트 클라우드가 존재하지 않습니다.")
            return
        self._log.text = "\t 다음 포인트 클라우드로 이동했습니다."
        self._load_scene()
    def _on_previous_scene(self):
        if self._check_changes():
            return
        if self.dataset is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_previous_scene)")
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
        if self.annotation_scene is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_next_scene)")
            return
        scene = self.dataset.get_next_scene()
        if scene is None:
            self._on_error("다음 작업 폴더가 존재하지 않습니다.")
            return
        self._log.text = "\t 다음 작업 폴더로 이동했습니다."
        self.annotation_scene = scene
        self._load_scene()
    def _on_save_label(self):
        self._log.text = "\t라벨링 결과를 저장 중입니다."
        self.window.set_needs_layout()
        
        if self.annotation_scene is None: # shsh
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_save_label)")
            return
        
        self.annotation_scene.save_label()
        self._last_saved = time.time()
        self._log.text = "\t라벨링 결과를 저장했습니다."
        self._annotation_changed = False
    def _on_load_previous_label(self):
        if self.annotation_scene is None: # shsh
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_save_label)")
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
        if self._active_hand is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _add_hand_frame)")
        self._remove_hand_frame()
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
        transform = np.eye(4)
        current_xyz = self._active_hand.get_control_rotation()
        transform[:3, :3] = Rotation.from_rotvec(current_xyz).as_matrix()
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
            r = Rotation.from_rotvec(current_xyz)
            current_rot_mat = r.as_matrix()
            rot_mat = o3d.geometry.get_rotation_matrix_from_xyz((rx, ry, rz))
            r = Rotation.from_matrix(np.matmul(current_rot_mat, rot_mat))
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
        hands = self._frame.hands
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
        self._log.text = "\t {} 자동 정렬 중입니다.".format(self._active_hand.get_control_joint_name())
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
            self._on_initial_viewpoint()
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
        
        
        # save label
        if event.key==gui.KeyName.F and event.type==gui.KeyEvent.DOWN:
            self._on_save_label()
            return gui.Widget.EventCallbackResult.CONSUMED
        
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
            # Rotation - keystrokes are not in same order as translation to make movement more human intuitive
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
            if (time.time()-self._last_saved) > self._auto_save_interval.double_value:
                self._annotation_changed = False
                self.annotation_scene.save_label()
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