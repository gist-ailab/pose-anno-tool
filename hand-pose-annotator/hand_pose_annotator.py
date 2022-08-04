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

from os.path import basename, dirname
from typing import Optional
import yaml
import copy

from utils.file_utils import *


MANO_PATH = os.path.join(str(Path(__file__).parent), 'models/mano')

class LabelingStage:
    LOADING = "준비중"
    TRANSLATION = "1. 손 이동"
    HAND_TIP =    "2. 손가락 끝 위치 조정"
    HAND_DETAIL = "3. 손가락 세부 위치 조정"

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
    IDX_ROOT   = [0]
    IDX_TIPS   = [4,8,12,16,20]
    IDX_THUMB  = [1,2,3,4]
    IDX_FORE   = [5,6,7,8]
    IDX_MIDDLE = [9,10,11,12]
    IDX_RING   = [13,14,15,16]
    IDX_LITTLE = [17,18,19,20]
    
    # optimze state
    OPTIMIZE_NONE   = -1
    OPTIMIZE_TRANS  = 0
    OPTIMIZE_TIPS   = 1
    OPTIMIZE_THUMB  = 2
    OPTIMIZE_FORE   = 3
    OPTIMIZE_MIDDLE = 4
    OPTIMIZE_RING   = 5
    OPTIMIZE_LITTLE = 6
    OPTIMIZE_WHOLE  = 7

    def __init__(self, side, shape_param=None):
        self.side = side
        
        if shape_param is None:
            self.shape_param = torch.zeros(1, 10)
        else:
            self.shape_param = torch.Tensor(shape_param).unsqueeze(0)
        
        self.mano_layer = ManoLayer(mano_root=MANO_PATH, side=side,
                            use_pca=False, flat_hand_mean=False)
        
        self.pose_param = torch.zeros(1, 45 + 3)
        self.pose_param.requires_grad = True
        self.trans_param = torch.zeros(1, 3)+1e-3
        self.trans_param.requires_grad = True

        self.optimizer = {
            "trans": optim.Adam([self.trans_param], lr=1e-2),
            "pose": optim.Adam([self.pose_param], lr=1e-3)
        }
        
        self.verts, self.joints = self.mano_layer(self.pose_param)
        self.faces = self.mano_layer.th_faces
        
        self.targets = torch.empty_like(self.joints).copy_(self.joints)
        
        self.optimize_state = self.OPTIMIZE_NONE
    
    def reset(self, shape_param=None):
        if shape_param is None:
            self.shape_param.zero_()
        else:
            self.shape_param = torch.Tensor(shape_param).unsqueeze(0)

        self.pose_param = torch.zeros(1, 45 + 3)
        self.pose_param.requires_grad = True
        self.trans_param = torch.zeros(1, 3)+1e-3
        self.trans_param.requires_grad = True
        self.optimizer = {
            "trans": optim.Adam([self.trans_param], lr=1e-2),
            "pose": optim.Adam([self.pose_param], lr=1e-3)
        }

        self.verts, self.joints = self.mano_layer(self.pose_param)
        self.faces = self.mano_layer.th_faces
        self.targets = torch.empty_like(self.joints).copy_(self.joints)
        self.optimize_state = self.OPTIMIZE_NONE
    
    def optimize_to_target(self):
        if self.optimize_state == self.OPTIMIZE_TRANS:
            optimizer = self.optimizer["trans"]
        else:
            optimizer = self.optimizer["pose"]
        optimizer.zero_grad()
        # forward
        self.verts, self.joints = self.mano_layer(th_pose_coeffs=self.pose_param,
                                                  th_betas=self.shape_param,
                                                  th_trans=self.trans_param)
        self.faces = self.mano_layer.th_faces
        # loss term
        loss = self._mse_loss()
        loss.backward()
        optimizer.step()
    
    def get_target(self):
        return self.targets.cpu().detach()[0, :]
    
    def set_target(self, targets):
        self.targets = torch.Tensor(targets).unsqueeze(0)

    def _mse_loss(self):
        
        if self.optimize_state == self.OPTIMIZE_TRANS:
            target_idx = self.IDX_ROOT
        elif self.optimize_state == self.OPTIMIZE_TIPS:
            target_idx = self.IDX_TIPS
        elif self.optimize_state == self.OPTIMIZE_THUMB:
            target_idx = self.IDX_THUMB
        elif self.optimize_state == self.OPTIMIZE_FORE:
            target_idx = self.IDX_FORE
        elif self.optimize_state == self.OPTIMIZE_MIDDLE:
            target_idx = self.IDX_MIDDLE
        elif self.optimize_state == self.OPTIMIZE_RING:
            target_idx = self.IDX_RING
        elif self.optimize_state == self.OPTIMIZE_LITTLE:
            target_idx = self.IDX_LITTLE
        else:
            target_idx = []
        return torch.norm(self.targets[:, target_idx]-self.joints[:, target_idx])
        
    def get_optimize_state(self):
        return self.optimize_state
    
    def set_optimize_state(self, state):
        self.optimize_state = state
    
    def get_geometry(self):
        return {
            "mesh": self._get_mesh(),
            "joint": self._get_joints(),
            "link": self._get_links()
        }

    def get_target_geometry(self):
        return {
            "joint": self._get_target_joints(),
            "link": self._get_target_links()
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

    def _get_target_joints(self):
        joints = self.targets.cpu().detach()[0, :]
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
        
        hands = {}
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
        
        # scale
        pcd.scale(1000, pcd.get_center())
        
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
        self.active_hand_mesh_material.base_color = [0.0, 1.0, 0.0, 0.5]
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
        self.active_target_joint_material.base_color = [0.0, 1.0, 0.0, 1.0]
        self.active_target_joint_material.shader = Settings.SHADER_POINT
        self.active_target_joint_material.point_size = 20.0
        
        self.active_target_link_material = rendering.MaterialRecord()
        self.active_target_link_material.base_color = [0.0, 1.0, 0.0, 1.0]
        self.active_target_link_material.shader = Settings.SHADER_POINT
        self.active_target_link_material.point_size = 5.0
        
        


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
        
        self._labeling_stage = LabelingStage.LOADING
        self._hands = None
   
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
            self._log.text = "\t라벨링 대상 파일을 불러왔습니다."
        except Exception as e:
            print(e)
            self._on_error("잘못된 경로가 입력되었습니다. (error at _on_filedlg_done)")
            self._log.text = "\t올바른 파일 경로를 선택하세요."
    
    # scene edit
    def _init_sceneedit_layout(self):
        em = self.window.theme.font_size
        
        sceneedit_layout = gui.CollapsableVert("편의 기능", 0.33*em,
                                          gui.Margins(em, 0, 0, 0))
        sceneedit_layout.set_is_open(True)
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
        
        sceneedit_layout.add_child(grid)
        
        self._settings_panel.add_child(sceneedit_layout)
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

        # link        
        mat = self.settings.hand_link_material
        mat.line_width = int(size)
        if self._check_geometry(self._right_hand_link_name):
            self._set_geometry_material(self._right_hand_link_name, mat)
        if self._check_geometry(self._left_hand_link_name):
            self._set_geometry_material(self._left_hand_link_name, mat)

        self._hand_line_size.double_value = size
    
    # labeling stage edit
    def _init_stageedit_layout(self):
        em = self.window.theme.font_size
        stageedit_layout = gui.CollapsableVert("라벨링 단계 선택", 0.33*em,
                                                  gui.Margins(0.25*em, 0, 0, 0))
        stageedit_layout.set_is_open(True)
        self._current_stage_str = gui.Label("현재 상태: 준비중")
        stageedit_layout.add_child(self._current_stage_str)
        
        button = gui.Button(LabelingStage.TRANSLATION)
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
        self._convert_stage(LabelingStage.TRANSLATION)
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

    # labeling hand edit
    def _init_handedit_layout(self):
        em = self.window.theme.font_size
        self._current_hand_id = 1 # 1 is right, -1 is left
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

        self._current_hand_id *= -1
        if self._current_hand_id > 0:
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
        target_geo = self._active_hand.get_target_geometry()
        self._add_geometry(self._target_joint_name, 
                           target_geo['joint'], self.settings.target_joint_material)
        self._add_geometry(self._target_link_name, 
                           target_geo['link'], self.settings.target_link_material)
    
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
        if self._current_hand_id > 0:
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

        self._convert_stage(LabelingStage.TRANSLATION)
        
    
    
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
    
    def _on_mouse_widget3d(self, event):
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
                        event.x, event.y, depth, self.scene_widget.frame.width,
                        self.scene_widget.frame.height)
                    self._move_target(world_xyz)
                    
            self._scene.scene.scene.render_to_depth_image(depth_callback)
            
            return gui.Widget.EventCallbackResult.HANDLED

        elif event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.RIGHT):
            self.toggle_label_visible()

        return gui.Widget.EventCallbackResult.IGNORED

    def _move_target(xyz):
        #TODO:
        if self.move_joint == 0:
            delta_root = torch.Tensor(world) - self.targets[0]
            self.targets = self.targets + delta_root
        else:
            self.targets[self.move_joint] = torch.Tensor(world)

    def _on_key_widget3d(self, event):
        # optimze
        if event.key == gui.KeyName.SPACE:
            self.optimize_to_target()
        # reset target
        elif event.key == gui.KeyName.HOME:
            self.initialize_target()

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
    