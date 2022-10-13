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
import logging
import atexit

from scipy.spatial.transform import Rotation as Rot
import psutil


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
        


class HandModel:
    def __init__(self, side):
        self.side = side
        self.mano_layer = ManoLayer(mano_root=MANO_PATH, side=side,
                            use_pca=False, flat_hand_mean=True, joint_rot_mode='axisang')
        
        self.pose_param = torch.zeros(1, 45)
        self.shape_param = torch.zeros(1, 10)
        self.root_trans = torch.zeros(1, 3)

        self.faces = self.mano_layer.th_faces

        self.verts, self.joints = self.mano_layer(th_pose_coeffs=self.pose_param,
                                        th_betas=self.shape_param,
                                        th_trans=self.root_trans)
        self.verts = self.verts.to(torch.float32)
        self.joints = self.joints.to(torch.float32)
    
    def reset(self):
        # reset to default selection
        pass

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
        for side, hand_model in self.hand_models.items():
            hand_model.reset(sc_hand_shapes[side])
        
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
                     hands=self.hand_models, 
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
        '000364922112', # ego_centric
    ]
    _CAMERAS = [
        "좌하단",
        "정하단",
        "정중단",
        "우하단",
        "우중단",
        "정상단",
        "좌중단",
        "후상단",
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
            self._scene_hand[sc_name] = {
                "right": shape['right'],
                "left": shape['left'],
            }
            
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
                t = extr[:3, 3] / 1e3
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
    
    if os.path.basename(dataset_dir)=='data4-source':
        return OurDataset(dataset_dir)
    else:
        raise NotImplementedError


class Camera:
    def __init__(self, name, serial, intrinsic, extrinsics, folder):
        self.name = name
        self.serial = serial
        self.intrinsic = intrinsic
        self.extrinsics = extrinsics
        self.folder = folder




class AppWindow:

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

            # ----- hand model setting
            self.hand_mesh_material = rendering.MaterialRecord()
            self.hand_mesh_material.base_color = [0.8, 0.8, 0.8, 1.0-self.hand_transparency]
            self.hand_mesh_material.shader = self.SHADER_LIT_TRANS
            # self.hand_mesh_material.line_width = 2.0
            
            self.hand_joint_material = rendering.MaterialRecord()
            self.hand_joint_material.base_color = [1.0, 0.0, 0.0, 1.0]
            self.hand_joint_material.shader = self.SHADER_UNLIT
            self.hand_joint_material.point_size = 5.0
            
            self.hand_link_material = rendering.MaterialRecord()
            self.hand_link_material.base_color = [1.0, 0.0, 0.0, 1.0]
            self.hand_link_material.shader = self.SHADER_LINE
            self.hand_link_material.line_width = 2.0
            
            self.active_hand_mesh_material = rendering.MaterialRecord()
            self.active_hand_mesh_material.base_color = [0.0, 1.0, 0.0, 1.0-self.hand_transparency]
            self.active_hand_mesh_material.shader = self.SHADER_LIT_TRANS
            
    def __init__(self, width, height, logger):
        #---- geometry name
        self._window_name = "Mano Hand Part Annotator by GIST AILAB"
        self.logger = logger


        self._right_hand_mesh_name = "right_hand_mesh"
        self._right_hand_joint_name = "right_hand_joint"
        self._right_hand_link_name = "right_hand_link"
        self._left_hand_mesh_name = "left_hand_mesh"
        self._left_hand_joint_name = "left_hand_joint"
        self._left_hand_link_name = "left_hand_link"
        
        self.hand_models = {
            "right": HandModel(side='right'),
            "left": HandModel(side='left')
        }

        self.window = gui.Application.instance.create_window(self._window_name, width, height)
        w = self.window
        
        self.settings = self.Settings()

        # 3D widget
        self._scene = gui.SceneWidget()
        scene = rendering.Open3DScene(w.renderer)
        scene.set_lighting(scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
        self._scene.scene = scene

        # ---- Settings panel
        em = w.theme.font_size
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # file edit
        self._init_fileeidt_layout()
        
        # image
        self._rgb_proxy = gui.WidgetProxy()
        self._rgb_proxy.set_widget(gui.ImageWidget())
        self._settings_panel.add_child(self._rgb_proxy)

        # view control

        # scene control

        # label control

        self._images_panel.set_is_open(False)
        self._diff_proxy = gui.WidgetProxy()
        self._diff_proxy.set_widget(gui.ImageWidget())
        self._images_panel.add_child(self._diff_proxy)
        self._images_panel.set_is_open(False)


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
        if os.environ.get("USERNAME")=='ailab':
            filedlg.set_path('/home/ailab/catkin_ws/src/gail-camera-manager/data/data4-source')
        elif os.environ.get("USERNAME")=='raeyo':
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