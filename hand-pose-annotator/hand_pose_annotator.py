# Author: Raeyoung Kang (raeyo@gm.gist.ac.kr)
# GIST AILAB, Republic of Korea
# Modified from the codes of Anas Gouda (anas.gouda@tu-dortmund.de)
# FLW, TU Dortmund, Germany

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.absolute()))

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import torch
from manopth.manolayer import ManoLayer
from torch import optim

import numpy as np

from os.path import basename 
import yaml

from utils.file_utils import *


MANO_PATH = os.path.join(str(Path(__file__).parent), 'models/mano')

class LabelingStage:
    LOADING = "준비중"
    ROOT = "1. 손 이동 및 회전"
    HAND_TIP =    "2. 손가락 끝 위치 조정(ZXCVB)"
    HAND_DETAIL = "3. 손가락 세부 위치 조정(ZXCV)"
    HAND_WHOLE = "4. 손 전체 최적화"

class DexYCB:
    """Dex-YCB"""
    _SUBJECTS = [
        '20200709-subject-01',
        '20200813-subject-02',
        '20200820-subject-03',
        '20200903-subject-04',
        '20200908-subject-05',
        # '20200918-subject-06',
        # '20200928-subject-07',
        # '20201002-subject-08',
        # '20201015-subject-09',
        # '20201022-subject-10',
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
        'tips'   : [4,8,12,16,20],
        'thumb'  : [1,2,3,4],
        'fore'   : [5,6,7,8],
        'middle' : [9,10,11,12],
        'ring'   : [13,14,15,16],
        'little' : [17,18,19,20],
        'whole'  : list(range(21))
    }

    def __init__(self, side, shape_param=None):
        self.side = side
        self.mano_layer = ManoLayer(mano_root=MANO_PATH, side=side,
                            use_pca=False, flat_hand_mean=False)
        self.reset(shape_param)
    
    def reset(self, shape_param=None):
        if shape_param is None:
            self.shape_param = torch.zeros(1, 10)
        else:
            self.shape_param = torch.Tensor(shape_param).unsqueeze(0)

        self.pose_param = torch.zeros(1, 45)
        self.pose_param.requires_grad = True
        self.rot_param = torch.zeros(1, 3)+1e-9
        self.rot_param.requires_grad = True
        self.trans_param = torch.zeros(1, 3)+1e-9
        self.trans_param.requires_grad = True
        self.optimizer = optim.Adam([self.rot_param, self.trans_param, self.pose_param], lr=1e-3)

        self.update_mano()
        
        self.root_delta = self.joints.cpu().detach()[0, 0]
        self.reset_target()
        
        self.optimize_state = 'none'
        self.active_joints = None
        self.contorl_joint = None
    
    def reset_pose(self):
        self.pose_param = torch.zeros(1, 45)
        self.pose_param.requires_grad = True
        self.optimizer = optim.Adam([self.rot_param, self.trans_param, self.pose_param], lr=1e-3)
        self.update_mano()
    
    def reset_root_rot(self):
        self.rot_param = torch.zeros(1, 3)+1e-9
        self.rot_param.requires_grad = True
        if self.optimize_state == 'root':
            self.rot_param.requires_grad = False
        self.optimizer = optim.Adam([self.rot_param, self.trans_param, self.pose_param], lr=1e-3)
        self.update_mano()
        
    def reset_root_trans(self):
        self.trans_param = torch.zeros(1, 3)+1e-9
        self.trans_param.requires_grad = True
        if self.optimize_state == 'root':
            self.trans_param.requires_grad = False
        self.optimizer = optim.Adam([self.rot_param, self.trans_param, self.pose_param], lr=1e-3)
        self.update_mano()
    
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
    
    def update_mano(self):
        pose_param = torch.concat((self.rot_param, self.pose_param), dim=1)
        verts, joints = self.mano_layer(th_pose_coeffs=pose_param,
                                                  th_betas=self.shape_param,
                                                  th_trans=self.trans_param)
        self.verts = verts / 1000
        self.joints = joints / 1000
        self.faces = self.mano_layer.th_faces
    
    def get_target(self):
        return self.targets.cpu().detach()[0, :]
    
    def set_target(self, targets):
        self.targets = torch.Tensor(targets).unsqueeze(0)
        self.targets.requires_grad = True
        self._target_changed = True

    def _mse_loss(self):
        if self.optimize_state=='root':
            # target_idx = self._IDX_OF_HANDS['root'] + self._IDX_OF_HANDS['tips']
            target_idx = self._IDX_OF_HANDS['whole']
        else:
            target_idx = self._IDX_OF_HANDS[self.optimize_state]
        return torch.norm(self.targets[:, target_idx]-self.joints[:, target_idx])
        
    def get_root_position(self):
        return self.trans_param.cpu().detach()[0, :] + self.root_delta
    def set_root_position(self, xyz):
        self.trans_param = torch.Tensor(xyz).unsqueeze(0) - self.root_delta
        self.trans_param.requires_grad = False
        self.update_mano()
        self.reset_target()
    
    def get_root_rotation(self):
        return self.rot_param.cpu().detach()[0, :]
    def set_root_rotation(self, xyz):
        self.rot_param = torch.Tensor(xyz).unsqueeze(0)
        self.rot_param.requires_grad = False
        self.update_mano()
        self.reset_target()

    def get_optimize_state(self):
        return self.optimize_state
    
    def set_optimize_state(self, state):
        if state == 'root':
            self.trans_param.requires_grad = False
            self.rot_param.requires_grad = False
        else:
            self.trans_param.requires_grad = True
            self.rot_param.requires_grad = True
        self.optimize_state = state
        
        self.active_joints = self._IDX_OF_HANDS[self.optimize_state]
        self.contorl_joint = self.active_joints[0]
    
    def set_control_joint(self, idx):
        assert len(self.active_joints) > 0, "set_control_joint error"
        self.contorl_joint = self.active_joints[idx]
    
    def get_geometry(self):
        return {
            "mesh": self._get_mesh(),
            "joint": self._get_joints(),
            "link": self._get_links()
        }

    def move_control_joint(self, xyz):
        if self.contorl_joint is None:
            return False
        
        if self.optimize_state == 'root':
            self.set_root_position(xyz)
        else:
            joints = self.get_target()
            joints[self.contorl_joint] = torch.Tensor(xyz)
            self.set_target(joints)

        return True

    def get_control_joint(self):
        if self.optimize_state == 'root':
            return self.get_root_position()
        else:
            joints = self.get_target()
            return np.array(joints[self.contorl_joint])
        
    def get_target_geometry(self):
        return {
            "joint": self._get_target_joints(),
            "link": self._get_target_links(),
        }
    def get_active_geometry(self):
        return {
            "joint": self._get_target_joints(self.active_joints),
            "link": None,
            "control": self._get_target_joints([self.contorl_joint])
        }
    
    def _get_mesh(self):
        verts = self.verts.cpu().detach()[0, :]
        faces = self.faces.cpu().detach()
        verts = o3d.utility.Vector3dVector(verts)
        faces = o3d.utility.Vector3iVector(faces)
        tri_mesh = o3d.geometry.TriangleMesh(vertices=verts, triangles=faces)
        lineset = o3d.geometry.LineSet.create_from_triangle_mesh(tri_mesh)
        
        return lineset
    
    def _get_joints(self):
        joints = self.joints.cpu().detach()[0, :]
        joints = o3d.utility.Vector3dVector(joints)
        pcd = o3d.geometry.PointCloud(points=joints)
        return pcd
    
    def _get_links(self):
        joints = self.joints.cpu().detach()[0, :]
        joints = o3d.utility.Vector3dVector(joints)
        lines = o3d.utility.Vector2iVector(np.array(HandModel.LINK))
        lineset = o3d.geometry.LineSet(lines=lines, points=joints)
        
        return lineset

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


class SceneObject:
    pass

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
              
class DexYCBDataset:
    # DexYCB Toolkit
    # Copyright (C) 2021 NVIDIA Corporation
    # Licensed under the GNU General Public License v3.0 [see LICENSE for details]

    def __init__(self, data_root):
        # Total Data Statics
        self._data_dir = data_root
        self._calib_dir = os.path.join(self._data_dir, "calibration")
    
        self._h = 480
        self._w = 640

        self._subjects = DexYCB._SUBJECTS
        self._serials = DexYCB._SERIALS
        
        self._intrinsics = {}
        for s in self._serials:
            intr_file = os.path.join(self._calib_dir, "intrinsics",
                               "{}_{}x{}.yml".format(s, self._w, self._h))
            with open(intr_file, 'r') as f:
                intr = yaml.load(f, Loader=yaml.FullLoader)
            intr = intr['color']
            self._intrinsics[s] = intr
            
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
        
        self.print_dataset_info()
        self.hand_models = {
            "right": HandModel(side='right'),
            "left": HandModel(side='left')
        }
        

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
        extr_file = os.path.join(self._calib_dir, 'extrinsics_{}/extrinsics.yml'.format(extrinsic_id))
        with open(extr_file, 'r') as f:
            extr = yaml.load(f, Loader=yaml.FullLoader)
        return extr['extrinsics'][camera_id]

    def get_mano_calib(self, mano_id):
        mano_file = os.path.join(self._calib_dir, 'mano_{}/mano.yml'.format(mano_id))
        with open(mano_file, 'r') as f:
            mano_calib = yaml.load(f, Loader=yaml.FullLoader)
        return mano_calib['betas']
    
    def get_scene_from_file(self, file_path):
        subject_id, scene_id, camera_id, frame_id = self.path_to_info(file_path)
        scene_meta = self._scene_meta[subject_id][scene_id]

        scene_dir = os.path.join(self._data_dir, subject_id, scene_id, camera_id)
        if camera_id=="merge":
            camera = None
        else:
            camera = Camera(camera_id, self._intrinsics[camera_id],self.get_extrinsic(scene_meta['extrinsics'], camera_id))
        
        calib = scene_meta['mano_calib'][0]
        for hand_model in self.hand_models.values():
            hand_model.reset(self.get_mano_calib(calib))
            
        return Scene(scene_dir=scene_dir, 
                     camera=camera, 
                     hands=self.hand_models, 
                     total_frame=scene_meta['num_frames'], 
                     current_frame=frame_id)
                
    
    @staticmethod
    def load_dataset_from_file(file_path):
        # initialize dataset directory # same as dex-ycb
        camera_dir = str(Path(file_path).parent)
        scene_dir = str(Path(camera_dir).parent)
        subject_dir = str(Path(scene_dir).parent)
        dataset_dir = str(Path(subject_dir).parent)
        
        return DexYCBDataset(dataset_dir)

    @staticmethod
    def path_to_info(file_path):
        camera_dir = str(Path(file_path).parent)
        scene_dir = str(Path(camera_dir).parent)
        subject_dir = str(Path(scene_dir).parent)
        
        frame_id = int(os.path.splitext(basename(file_path))[0].split("_")[-1])
        camera_id = basename(camera_dir)
        scene_id = basename(scene_dir)
        subject_id = basename(subject_dir)

        return subject_id, scene_id, camera_id, frame_id

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
        self._label_format = "labels_{:06d}.npy"
        
    def get_current_frame(self):
        try:
            pcd = self._load_point_cloud(self._frame_path)
        except:
            print("Fail to load point cloud")
            return None
        return Scene.Frame(frame_idx=self.frame_id,
                           scene_pcd=pcd,
                           hands=self._hands,
                           objs=[])
        
    def moveto_next_frame(self):
        frame_id = self.frame_id + 1
        if frame_id > self.total_frame - 1:
            return False
        else:
            self.frame_id = frame_id
            return True
    
    def moveto_previous_frame(self):
        frame_id = self.frame_id - 1
        if frame_id < 0:
            return False
        else:
            self.frame_id = frame_id
            return True

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

    def _load_label(file_path):
        return np.load(file_path)

    def save_label(self):
        np.save(self._label, self._label_path)

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
        self._scene.setup_camera(60, self.bounds, self.bounds.get_center())
        center = np.array([0, 0, 0])
        eye = center + np.array([0, 0, -0.5])
        up = np.array([0, -1, 0])
        self._scene.look_at(center, eye, up)
        self._init_view_control()
        
    def _init_view_control(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
    
    
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
        self._labeling_stage = LabelingStage.LOADING
        self._hands = None
        self._active_hand = None
        self.upscale_responsiveness = False
        self._left_shift_modifier = False
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
        self._init_sceneedit_layout()
        self._init_handedit_layout()
        self._init_stageedit_layout()
        
        
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
        self._on_hand_point_size(5) # set default size to 5
        self._on_hand_line_size(2) # set default size to 2
        self._on_responsiveness(5) # set default responsiveness to 5
        
        self._scene.set_on_mouse(self._on_mouse)
        self._scene.set_on_key(self._on_key)
        
        self.window.set_on_tick_event(self._init_view_control)
        
        
    #region Layout and Callback
   
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
        filedlg.add_filter("", "모든 파일")
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
            self.dataset = DexYCBDataset.load_dataset_from_file(file_path)
            self.annotation_scene = self.dataset.get_scene_from_file(file_path)
            self._load_scene()
            self.window.close_dialog()
            self._log.text = "\t 라벨링 대상 파일을 불러왔습니다."
        except Exception as e:
            print(e)
            self._on_error("잘못된 경로가 입력되었습니다. (error at _on_filedlg_done)")
            self._log.text = "\t 올바른 파일 경로를 선택하세요."
    
    # scene edit
    def _init_sceneedit_layout(self):
        em = self.window.theme.font_size
        
        sceneedit_layout = gui.CollapsableVert("편의 기능", 0.33*em,
                                          gui.Margins(em, 0, 0, 0))
        sceneedit_layout.set_is_open(True)
        
        self._show_axes = gui.Checkbox("카메라 좌표계 보기")
        self._show_axes.set_on_checked(self._on_show_axes)
        sceneedit_layout.add_child(self._show_axes)

        self._show_coord_frame = gui.Checkbox("조작 중인 조인트 좌표계 보기")
        self._show_coord_frame.set_on_checked(self._on_show_coord_frame)
        sceneedit_layout.add_child(self._show_coord_frame)

        self._show_hands = gui.Checkbox("손 라벨 보기")
        self._show_hands.set_on_checked(self._on_show_hand)
        sceneedit_layout.add_child(self._show_hands)

        
        grid = gui.VGrid(2, 0.25 * em)
        self._scene_point_size = gui.Slider(gui.Slider.INT)
        self._scene_point_size.set_limits(1, 20)
        self._scene_point_size.set_on_value_changed(self._on_scene_point_size)
        grid.add_child(gui.Label("포인트 크기"))
        grid.add_child(self._scene_point_size)
        
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
        
        sceneedit_layout.add_child(grid)
        
        self._settings_panel.add_child(sceneedit_layout)
    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._scene.scene.show_axes(self.settings.show_axes)
    def _on_show_hand(self, show):
        if self._active_hand is None: # shsh
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_show_hand)")
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
    def _add_coord_frame(self, name="coord_frame", size=0.02, origin=[0, 0, 0]):
        if self._active_hand is None: # shsh
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _add_coord_frame)")
            return
        self._scene.scene.remove_geometry(name)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
        if "world" in name:
            transform = np.eye(4)
            transform[:3, 3] = self._active_hand.get_control_joint().T
            coord_frame.transform(transform)
            for label in self.coord_labels:
                self._scene.remove_3d_label(label)
            self.coord_labels = []
            # size = size * 0.06 
            # self.coord_labels.append(self._scene.add_3d_label(self._active_hand.get_control_joint().T + np.array([size, 0, 0]), "D (+)"))
            # self.coord_labels.append(self._scene.add_3d_label(self._active_hand.get_control_joint().T + np.array([-size, 0, 0]), "A (-)"))
            # self.coord_labels.append(self._scene.add_3d_label(self._active_hand.get_control_joint().T + np.array([0, size, 0]), "S (+)"))
            # self.coord_labels.append(self._scene.add_3d_label(self._active_hand.get_control_joint().T + np.array([0, -size, 0]), "W (-)"))
            # self.coord_labels.append(self._scene.add_3d_label(self._active_hand.get_control_joint().T + np.array([0, 0, size]), "Q (+)"))
            # self.coord_labels.append(self._scene.add_3d_label(self._active_hand.get_control_joint().T + np.array([0, 0, -size]), "E (-)"))

        else:
            transform = np.eye(4)
            transform[:3, 3] = self._active_hand.get_control_joint().T
            coord_frame.transform(transform)
            
        self._scene.scene.add_geometry(name, coord_frame, 
                                        self.settings.coord_material,
                                        add_downsampled_copy_for_fast_rendering=True) 
    def _on_scene_point_size(self, size):
        mat = self.settings.scene_material
        mat.point_size = int(size)
        if self._check_geometry(self._scene_name):
            self._set_geometry_material(self._scene_name, mat)
        self._scene_point_size.double_value = size
    def _on_hand_point_size(self, size):
        mat = self.settings.hand_joint_material
        mat.point_size = int(size)
        if self._check_geometry(self._right_hand_joint_name):
            self._set_geometry_material(self._right_hand_joint_name, mat)
        if self._check_geometry(self._left_hand_joint_name):
            self._set_geometry_material(self._left_hand_joint_name, mat)
        self._hand_point_size.double_value = size
    def _on_hand_line_size(self, size):
        # mesh
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
        self.dist = 0.0004 * responsiveness
        self.deg = 0.2 * responsiveness
        self._responsiveness.double_value = responsiveness
    
    # labeling stage edit
    def _init_stageedit_layout(self):
        em = self.window.theme.font_size
        stageedit_layout = gui.CollapsableVert("라벨링 단계 선택(숫자 1,2,3)", 0.33*em,
                                                  gui.Margins(0.25*em, 0, 0, 0))
        stageedit_layout.set_is_open(True)
        self._current_stage_str = gui.Label("현재 상태: 준비중")
        stageedit_layout.add_child(self._current_stage_str)
        
        button = gui.Button(LabelingStage.ROOT)
        button.set_on_clicked(self._on_translation_stage)
        stageedit_layout.add_child(button)
        
        button = gui.Button(LabelingStage.HAND_TIP)
        button.set_on_clicked(self._on_hand_tip_stage)
        stageedit_layout.add_child(button)
        
        button = gui.Button(LabelingStage.HAND_DETAIL)
        button.set_on_clicked(self._on_hand_detail_stage)
        stageedit_layout.add_child(button)
        self._settings_panel.add_child(stageedit_layout)

    def _on_translation_stage(self):
        self._convert_stage(LabelingStage.ROOT)
    def _on_hand_tip_stage(self):
        self._convert_stage(LabelingStage.HAND_TIP)
    def _on_hand_detail_stage(self):
        self._convert_stage(LabelingStage.HAND_DETAIL)
    def _convert_stage(self, labeling_stage):
        if self._hands is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _convert_stage)")
            return 
        self._labeling_stage = labeling_stage
        self._current_stage_str.text = "현재 상태: {}".format(self._labeling_stage)
        if labeling_stage==LabelingStage.ROOT:
            self._active_hand.set_optimize_state('root')
        elif labeling_stage==LabelingStage.HAND_TIP:
            self._active_hand.set_optimize_state('tips')
        elif labeling_stage==LabelingStage.HAND_DETAIL:
            self._active_hand.set_optimize_state('thumb')
        elif labeling_stage==LabelingStage.HAND_WHOLE:
            self._active_hand.set_optimize_state('whole')
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
        
        button = gui.Button("손 바꾸기")
        button.set_on_clicked(self._convert_hand)
        handedit_layout.add_child(button)
        
        self._settings_panel.add_child(handedit_layout)
    def _convert_hand(self):
        if self._hands is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _convert_hand)")
            return 

        self._is_right_hand = not self._is_right_hand
        if self._is_right_hand:
            self._current_hand_str.text = "현재 대상: 오른손"
            active_side = 'right'
        else:
            self._current_hand_str.text = "현재 대상: 왼손"
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
        
        self._active_hand = self._hands[active_side]
        self._convert_stage(self._labeling_stage)
        
        self._update_target_hand()
    
    #endregion
    def _load_scene(self):
        self._frame = self.annotation_scene.get_current_frame()
        # visualize scene pcd
        pcd = self._frame.scene_pcd
        self.bounds = pcd.get_axis_aligned_bounding_box()
        self._on_initial_viewpoint()
        self._add_geometry(self._scene_name, pcd, self.settings.scene_material)
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
    
        target_geo = self._active_hand.get_target_geometry()
        self._add_geometry(self._target_joint_name, 
                           target_geo['joint'], self.settings.target_joint_material)
        self._add_geometry(self._target_link_name, 
                           target_geo['link'], self.settings.target_link_material)

        self._convert_stage(LabelingStage.ROOT)
        
        active_geo = self._active_hand.get_active_geometry()
        self._add_geometry(self._active_joint_name, 
                           active_geo['joint'], self.settings.active_target_joint_material)
        self._add_geometry(self._control_joint_name, 
                           active_geo['control'], self.settings.control_target_joint_material)
        self.control_joint_geo = active_geo['control']
    def _next_scene(self):
        pcd = self.annotation_scene.get_next_frame()
    
    def _update_scene(self):
        pass
    
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
    
    def _on_mouse(self, event):
        # We could override BUTTON_DOWN without a modifier, but that would
        # interfere with manipulating the scene.
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.ALT):
            
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
                    self._move_control_joint(world_xyz)
                    
            self._scene.scene.scene.render_to_depth_image(depth_callback)
            
            
        elif event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.RIGHT):
            self._toggle_hand_visible()

        return gui.Widget.EventCallbackResult.IGNORED

    def move(self, x, y, z, rx, ry, rz):
        if x != 0 or y != 0 or z != 0:
            current_xyz = self._active_hand.get_control_joint()
            # convert x, y, z cam to world
            R = self._scene.scene.camera.get_view_matrix()[:3,:3]
            R_inv = np.linalg.inv(R)
            xyz = np.dot(R_inv, np.array([x, y, z]))
            # xyz[1] *= -1#TODO: ... why -y
            xyz = current_xyz + xyz
            self._active_hand.move_control_joint(xyz)
        else:
            current_xyz = self._active_hand.get_root_rotation()
            xyz = current_xyz + np.array([rx, ry, rz], dtype=np.float32)
            self._active_hand.set_root_rotation(xyz)
        
        self._update_activate_hand()
        self._update_target_hand()
        
    
    def _move_control_joint(self, xyz):
        self._active_hand.move_control_joint(xyz)
        self._update_activate_hand()
        self._update_target_hand()
        
    def _update_target_hand(self):
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
        pass
        
    def _on_key(self, event):
        if self._labeling_stage == LabelingStage.LOADING:
            return gui.Widget.EventCallbackResult.IGNORED
        
        if event.key == gui.KeyName.LEFT_SHIFT or event.key == gui.KeyName.RIGHT_SHIFT:
            if event.type == gui.KeyEvent.DOWN:
                self._left_shift_modifier = True
            elif event.type == gui.KeyEvent.UP:
                self._left_shift_modifier = False
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
        
        # optimze
        if event.key == gui.KeyName.SPACE:
            if self._active_hand is None:
                pass
            if not self._labeling_stage == LabelingStage.ROOT:
                self._active_hand.optimize_to_target()
                self._update_activate_hand()
            
        # reset guide pose
        elif event.key == gui.KeyName.HOME:
            self._active_hand.reset_target()
            self._update_target_hand()
        
        # convert hand
        elif (event.key == gui.KeyName.TAB) and (event.type==gui.KeyEvent.DOWN):
            self._convert_hand()
        
        # stage change
        elif event.key == gui.KeyName.ONE:
            self._convert_stage(LabelingStage.ROOT)
        elif event.key == gui.KeyName.TWO:
            self._convert_stage(LabelingStage.HAND_TIP)
        elif event.key == gui.KeyName.THREE:
            self._convert_stage(LabelingStage.HAND_DETAIL)
        elif event.key == gui.KeyName.FOUR:
            self._convert_stage(LabelingStage.HAND_WHOLE)
        


        # change control joint
        if self._labeling_stage==LabelingStage.HAND_TIP:
            if event.key == gui.KeyName.Z:
                self._active_hand.set_control_joint(0)
            elif event.key == gui.KeyName.X:
                self._active_hand.set_control_joint(1)
            elif event.key == gui.KeyName.C:
                self._active_hand.set_control_joint(2)
            elif event.key == gui.KeyName.V:
                self._active_hand.set_control_joint(3)
            elif event.key == gui.KeyName.B:
                self._active_hand.set_control_joint(4)
            self._update_target_hand()
        elif self._labeling_stage==LabelingStage.HAND_DETAIL:
            # convert finger
            if event.key == gui.KeyName.Z:
                self._active_hand.set_optimize_state('thumb')
            elif event.key == gui.KeyName.X:
                self._active_hand.set_optimize_state('fore')
            elif event.key == gui.KeyName.C:
                self._active_hand.set_optimize_state('middle')
            elif event.key == gui.KeyName.V:
                self._active_hand.set_optimize_state('ring')
            elif event.key == gui.KeyName.B:
                self._active_hand.set_optimize_state('little')
            
            # convert joint
            if event.key == gui.KeyName.U:
                self._active_hand.set_control_joint(0)
            elif event.key == gui.KeyName.I:
                self._active_hand.set_control_joint(1)
            elif event.key == gui.KeyName.O:
                self._active_hand.set_control_joint(2)
            elif event.key == gui.KeyName.P:
                self._active_hand.set_control_joint(3)
            
            self._update_target_hand()
        
        # Translation
        if not self._left_shift_modifier:
            self._log.text = "\t손목 위치를 조정 중 입니다."
            if event.key == gui.KeyName.D:
                self.move( self.dist, 0, 0, 0, 0, 0)
            elif event.key == gui.KeyName.A:
                self.move( -self.dist, 0, 0, 0, 0, 0)
            elif event.key == gui.KeyName.S:
                self.move( 0, -self.dist, 0, 0, 0, 0)
            elif event.key == gui.KeyName.W:
                self.move( 0, self.dist, 0, 0, 0, 0)
            elif event.key == gui.KeyName.Q:
                self.move( 0, 0, self.dist, 0, 0, 0)
            elif event.key == gui.KeyName.E:
                self.move( 0, 0, -self.dist, 0, 0, 0)
        # Rotation - keystrokes are not in same order as translation to make movement more human intuitive
        else:
            if self._labeling_stage==LabelingStage.ROOT:
                self._log.text = "\t손목 자세를 조정 중 입니다."
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

        return gui.Widget.EventCallbackResult.IGNORED

        
def main():
    gui.Application.instance.initialize()
    hangeul = os.path.join(str(Path(__file__).parent.absolute()), "lib/NanumGothic.ttf")
    font = gui.FontDescription(hangeul)
    font.add_typeface_for_language(hangeul, "ko")
    gui.Application.instance.set_font(gui.Application.DEFAULT_FONT_ID, font)

    w = AppWindow(1920, 1080)
    gui.Application.instance.run()

if __name__ == "__main__":
    # o3d.visualization.webrtc_server.enable_webrtc()
    main()
    # file_path = '/data/datasets/hope/dex-ycb-sample/20200709-subject-01/20200709_141754/merge/points_000000.pcd'
    # dataset = DexYCBDataset.load_dataset_from_file(file_path)
    # scene = dataset.get_scene_from_file(file_path)
    
    
    # DexYCBDataset('/data/datasets/hope/dex-ycb')
    