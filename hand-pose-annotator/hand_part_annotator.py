import os
import sys

from sympy import det
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
        
        self.verts_to_points, self.sampled_face_idx = torch.load(os.path.join(MANO_PATH, 'verts_to_points_{}.pt'.format(side)))

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
            "points": self._get_points(),
            "mesh": self._get_mesh(),
            "joints": self._get_joints(),
            "links": self._get_links()
        }
    
    def _get_points(self):
        verts = self.verts.cpu().detach()[0, :]
        points = torch.matmul(self.verts_to_points, verts)
        points = o3d.utility.Vector3dVector(points)
        
        return o3d.geometry.PointCloud(points=points)
    
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

    def get_points(self):
        verts = self.verts.cpu().detach()[0, :]
        return torch.matmul(self.verts_to_points, verts)

    def get_face_indices_from_points(self, point_idxs):
        return self.sampled_face_idx[point_idxs]


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
        sc_cam_info = self._scene_camera[sc_name] 
        sc_frame_list = self._scene_frame[sc_name] 
        
        
        # camera
        cameras = {}
        for cam, cam_info in sc_cam_info.items():
            cameras[cam] = Camera(cam, self._cam2serial[cam], cam_info['intrinsic'], cam_info['extrinsics'], self._cam2serial[cam])
        
        self.current_scene_idx = self._total_scene.index(sc_name)
        self.current_scene_file = sc_name
        self.current_frame_file = frame_id
            
        return Scene(scene_dir=sc_path, 
                     cameras=cameras,
                     frame_list=sc_frame_list,
                     current_frame=frame_id,
                     )

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


class Scene:
    
    class Frame:
        class SingleView:
            def __init__(self, rgb, frame_id):
                self.rgb = rgb
                self.frame_id = frame_id
                self.part_label = None

        def __init__(self, scene_dir, frame_id, cams):
            self.scene_dir = scene_dir
            
            self.id = frame_id
            
            self.cameras = cams
            self.camera_names = list(self.cameras.keys())
            self.camera_names.sort()
            
            self.rgb_format = os.path.join(self.scene_dir, "{}", "rgb", "{:06d}.png".format(self.id)) 

            self.single_views = {}
            for cam_name, cam in cams.items():
                self.single_views[cam_name] = self.SingleView(rgb=self._load_rgb(cam_name),
                                                              frame_id=self.id)
            
            self._camera_idx = 0
            self.active_cam = self.camera_names[self._camera_idx]
        
        def get_camera_name_list(self):
            return self.camera_names
        def get_current_camera_idx(self):
            return self._camera_idx
        def get_current_camera_name(self):
            return self.active_cam
        
        def move_to_next_camera(self):
            camera_idx = self._camera_idx + 1
            if camera_idx > len(self.camera_names) - 1:
                return False
            else:
                self._camera_idx = camera_idx
                self.active_cam = self.camera_names[self._camera_idx]
                return True
        def move_to_previous_camera(self):
            camera_idx = self._camera_idx - 1
            if camera_idx < 0:
                return False
            else:
                self._camera_idx = camera_idx
                self.active_cam = self.camera_names[self._camera_idx]
                return True
        def _load_rgb(self, cam_name):
            folder = self.cameras[cam_name].folder
            rgb_path = self.rgb_format.format(folder)
            return cv2.imread(rgb_path)
        def get_rgb(self, cam_name):
            return self.single_views[cam_name].rgb
        def get_progress(self):
            return "현재 진행률: {} [{}/{}]".format(self.active_cam, self._camera_idx+1, len(self.camera_names))
        
    def __init__(self, scene_dir, cameras, frame_list, current_frame):
        self._scene_dir = scene_dir
        self._label_dir = os.path.join(scene_dir, 'labels')
        os.makedirs(self._label_dir, exist_ok=True)
        
        self._cameras = cameras
        
        self.total_frame = len(frame_list) # for master camera
        self._frame_list = frame_list
        self.frame_id = current_frame
        self._frame_idx = self._frame_list.index(self.frame_id)
        
        self._part_label_format = os.path.join(self._label_dir, "parts_{:06d}.npz")

        self._part_label = None
        self._previous_part_label = None

    def _load_frame(self):
        try:
            self.current_frame = Scene.Frame(scene_dir=self._scene_dir,
                                             frame_id=self.frame_id,
                                             cams=self._cameras,
                                             )
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
        part_label = {}
        np.savez(self._part_label_path, **part_label)
        self._part_label = part_label
    def load_label(self):
        # part label
        if self._part_label is not None:
            self._previous_part_label = self._part_label.copy()
        try:
            self._part_label = dict(np.load(self._part_label_path))
        except:
            print("Fail to load part Label -> Load previous Label")
            try:
                self._part_label = self._previous_part_label.copy()
            except:
                print("Fail to load previous Label -> Reset Label")
                self._part_label = {
                    cam.folder: None for cam in self._cameras.values()
                }
    def load_previous_label(self):
        if self._previous_part_label is None:
            return False
        try:
            self._part_label = self._previous_part_label.copy()
            return True
        except:
            return False
    def update_label(self, cam_name, label):
        serial = self._cameras[cam_name].folder
        self._part_label[serial] = label.copy()
    def get_label(self, cam_name):
        serial = self._cameras[cam_name].folder
        try:
            label = self._part_label[serial]
        except:
            label = None
        return label

    @property
    def _part_label_path(self):
        return os.path.join(self._scene_dir, self._part_label_format.format(self.frame_id))


def load_dataset_from_file(file_path):
    data_dir = os.path.dirname(file_path)
    camera_dir = os.path.dirname(data_dir)
    scene_dir = os.path.dirname(camera_dir)
    dataset_dir = os.path.dirname(scene_dir)
    
    return OurDataset(dataset_dir)
    

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
            self.hand_mesh_material.base_color = [0.5, 0.5, 0.5, 1.0]
            self.hand_mesh_material.shader = self.SHADER_LIT_TRANS

            # activate material
            self.active_mesh_material = rendering.MaterialRecord()
            self.active_mesh_material.base_color = [1.0, 0, 0, 1.0]
            self.active_mesh_material.shader = self.SHADER_LIT_TRANS

            
    def __init__(self, width, height, logger):
        #---- geometry name
        self._window_name = "Mano Hand Part Annotator by GIST AILAB"
        self.logger = logger


        self._hand_mesh_name = "{}_hand_mesh"
        self._active_mesh_name = "{}_active_mesh"
        
        # initialize values
        self._active_sphere = None
        self.prev_ms_x, self.prev_ms_y = None, None
        self.dataset = None
        self.annotation_scene = None
        
        self._view_rock = False
        self._depth_image = None
        
        self._annotation_changed = False
        self._last_change = time.time()
        self._last_saved = time.time()
        
        self.scale_factor = None
        

        self.hand_models = {}
        self.hand_geometry = {}
        self.hand_points = {}
        self.active_faces = {}
        for side in ['right', 'left']:
            hand_model = HandModel(side=side)
            self.hand_models[side] = hand_model
            self.hand_geometry[side] = hand_model._get_mesh()
            self.hand_points[side] = hand_model.get_points()
            self.active_faces[side] = []
        
        self.window = gui.Application.instance.create_window(self._window_name, width, height)
        w = self.window
        
        self.settings = self.Settings()

        # 3D widget
        self._scene = gui.SceneWidget()
        scene = rendering.Open3DScene(w.renderer)
        scene.set_lighting(scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
        self._scene.scene = scene

        em = w.theme.font_size

        #----- image panel
        self._images_panel = gui.CollapsableVert("이미지 보기", 0.33 * em,
                                                 gui.Margins(em, 0, 0, 0))
        
        self._rgb_proxy = gui.WidgetProxy()
        self._rgb_proxy.set_widget(gui.ImageWidget())
        self._images_panel.add_child(self._rgb_proxy)
        
        h = gui.Horiz()
        self._image_progress = gui.Label("현재 시점: 준비중 [00/00]")
        self._image_progress_bar = gui.ProgressBar()
        self._image_progress_bar.value = 0.0
        
        h.add_child(self._image_progress)
        h.add_child(self._image_progress_bar)
        self._images_panel.add_child(h)
        
        
        # ---- Settings panel
        
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # file edit
        self._init_fileeidt_layout()
        self._init_viewctrl_layout()
        self._init_scene_control_layout()
        self._init_label_control_layout()

        # label control
        
        # ---- log panel
        self._log_panel = gui.VGrid(1, em)
        self._log = gui.Label("\t 라벨링 대상 파일을 선택하세요. ")
        self._log_panel.add_child(self._log)
        
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        w.add_child(self._log_panel)
        w.add_child(self._images_panel)
        w.set_on_layout(self._on_layout)
        
        self._scene.set_on_mouse(self._on_mouse)
        self._scene.set_on_key(self._on_key)
        self.window.set_on_tick_event(self._on_tick)
        
        
        self._initialize_background()
        self._initialize_hand_geometry()
    
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
    
    def _initialize_background(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._set_background_color(bg_color)
    def _initialize_hand_geometry(self):
        for side, hand_geo_mesh in self.hand_geometry.items():
            self._add_geometry(self._hand_mesh_name.format(side), hand_geo_mesh, material=self.settings.hand_mesh_material)
        self._on_initial_viewpoint()
    def _on_initial_viewpoint(self):
        self.bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, self.bounds, self.bounds.get_center())
        center = np.array([0, 1, 0.3])
        eye = np.array([0, -0.3, 0])
        up = np.array([0, 0, 1])
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
            self._load_scene()
            self.window.close_dialog()
            self._log.text = "\t 라벨링 대상 파일을 불러왔습니다."
        except Exception as e:
            print(e)
            self._on_error("잘못된 경로가 입력되었습니다. (error at _on_filedlg_done)")
            self._log.text = "\t 올바른 파일 경로를 선택하세요."

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

        self._auto_save = gui.Checkbox("자동 저장 활성화")
        viewctrl_layout.add_child(self._auto_save)
        self._auto_save.checked = True

        grid = gui.VGrid(2, 0.25 * em)    
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
    def _on_show_hand(self, show):
        self.settings.show_hand = show
        self._update_hand_layer()
    def _on_responsiveness(self, responsiveness):
        self.logger.debug('_on_responsiveness')
        self._log.text = "\t 라벨링 민감도 값을 변경합니다."
        self.window.set_needs_layout()
        self._last_change = time.time()
        self.dist = 0.0004 * responsiveness
        self._responsiveness.double_value = responsiveness
    def _on_auto_save_interval(self, interval):
        self.logger.debug('_on_auto_save_interval')
        self._log.text = "\t 자동 저장 간격을 변경합니다."
        self.window.set_needs_layout()
        self._auto_save_interval.double_value = interval
    
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
        
        v = gui.Vert(0.4 * em)
        
        v.add_child(self._image_progress)
        h = gui.Horiz(0.4 * em)
        button = gui.Button("이전 시점 (A)")
        button.horizontal_padding_em = 0.8
        button.vertical_padding_em = 0
        button.set_on_clicked(self._on_previous_camera)
        h.add_child(button)
        button = gui.Button("다음 시점 (D)")
        button.horizontal_padding_em = 0.8
        button.vertical_padding_em = 0
        button.set_on_clicked(self._on_next_camera)
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
        self._init_cam_name()
        self.annotation_scene.load_label()
        self._reset_image_viewer()
        self._update_image_viewer()
        self._update_progress_str()
        
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
    def _on_previous_camera(self):
        if not self._check_annotation_scene():
            return
        if self._check_changes():
            return
        self._log.text = "\t 이전 시점으로 이동했습니다."
        ret = self._frame.move_to_previous_camera()
        if not ret:
            self._on_error("이전 시점이 존재하지 않습니다.")
        self._camera_idx = self._frame.get_current_camera_idx()
        self._cam_name = self._frame.get_current_camera_name()
        self._on_change_camera()
    def _on_next_camera(self):
        if not self._check_annotation_scene():
            return
        if self._check_changes():
            return
        self._log.text = "\t 다음 시점으로 이동했습니다."
        ret = self._frame.move_to_next_camera()
        if not ret:
            self._on_error("다음 시점이 존재하지 않습니다.")
        self._camera_idx = self._frame.get_current_camera_idx()
        self._cam_name = self._frame.get_current_camera_name()
        self._on_change_camera()
    def _on_change_camera(self):
        self._load_part_label()
        self._image_progress.text = self._frame.get_progress()
        self._image_progress_bar.value = (self._camera_idx + 1) / len(self._cam_name_list)
        self._reset_image_viewer()
        self._update_image_viewer()
        self._update_hand_layer()
    def _init_label_control_layout(self):
        self.logger.debug('_init_label_control_layout')
        em = self.window.theme.font_size
        label_control_layout = gui.CollapsableVert("", 0.33 * em,
                                                   gui.Margins(0.25 * em, 0, 0, 0))
        label_control_layout.set_is_open(True)
        
        self._labeling_mode = gui.Label("라벨링 모드: 화면 조작중")
        label_control_layout.add_child(self._labeling_mode)
        button = gui.Button("라벨링 모드 전환 (B)")
        button.set_on_clicked(self._toggle_labeling_mode)
        label_control_layout.add_child(button)
        
        button = gui.Button("라벨링 결과 저장하기 (F)")
        button.set_on_clicked(self._on_save_label)
        label_control_layout.add_child(button)
        
        button = gui.Button("이전 이미지 라벨 불러오기")
        button.set_on_clicked(self._on_load_previous_label)
        label_control_layout.add_child(button)
        self._settings_panel.add_child(label_control_layout)
    def _toggle_labeling_mode(self):
        # if not self._check_annotation_scene():
        #     return 
        self._view_rock = not self._view_rock
        if self._view_rock:
            self._labeling_mode.text = "라벨링 모드: 라벨링 중"
            self.settings.hand_mesh_material.base_color = [0.11, 0.11, 0.11, 1.0]
            self._scene.scene.scene.render_to_depth_image(self._depth_callback)
        else:
            self._labeling_mode.text = "라벨링 모드: 화면 조작중"
            self.settings.hand_mesh_material.base_color = [0.99, 0.99, 0.99, 1.0]
        self._update_hand_layer()
        self.window.set_needs_layout()
    def _on_save_label(self):
        self.logger.debug('_on_save_label')
        self._log.text = "\t라벨링 결과를 저장 중입니다."
        self.window.set_needs_layout()
        if not self._check_annotation_scene():
            return
        self.annotation_scene.save_label()
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
    def _init_cam_name(self):
        self._cam_name_list = self._frame.get_camera_name_list()
        self._camera_idx = self._frame.get_current_camera_idx()
        self._cam_name = self._frame.get_current_camera_name()
        self._on_change_camera()
    def _load_part_label(self):
        loaded_label = self.annotation_scene.get_label(self._cam_name)
        if loaded_label is not None:
            self.active_faces = loaded_label
        else:
            self._reset_label()
    def _update_image_viewer(self):
        if self._camera_idx == -1:
            self._rgb_proxy.set_widget(gui.ImageWidget())
            return
        current_cam = self._cam_name_list[self._camera_idx]
        rgb_img = self._frame.get_rgb(current_cam)
        self.rgb_img = rgb_img
        self.H, self.W, _ = rgb_img.shape
        self._rgb_proxy.set_widget(gui.ImageWidget(self._img_wrapper(self.rgb_img)))
    def _reset_image_viewer(self):
        self.logger.debug('_reset_image_viewer')
        self.icx, self.icy = self.W / 2, self.H / 2
        self.scale_factor = 1
        self._move_viewer()
    def _img_wrapper(self, img):
        self.logger.debug('_img_wrapper')
        ratio = 640 / self.W
        img = cv2.resize(img.copy(), (640, int(self.H*ratio)))
        return o3d.geometry.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
    def _reset_label(self):
        self.active_faces = {side: [] for side in ['right', 'left']}

    def _on_mouse(self, event):
        if self._view_rock:
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
            return gui.Widget.EventCallbackResult.IGNORED
    def _check_xyz_movement(self, xyz):
        try:
            movement = np.linalg.norm(xyz - self._last_xyz)
        except:
            movement = 10
        print(movement)
        if movement < 0.01:
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
        self._annotation_changed = True
        for side, points in self.hand_points.items():
            dist = np.linalg.norm(points - xyz, axis=1)
            active_faces = self.active_faces[side]
            inlier_points = np.where(dist < 0.005)[0].tolist()
            inlier_faces = np.unique(self.hand_models[side].get_face_indices_from_points(inlier_points))
            if invert:
                active_faces = list(set(active_faces) - set(inlier_faces))
            else:
                active_faces = list(set(active_faces) | set(inlier_faces))
            self.active_faces[side] = active_faces
        self._update_hand_layer()
        self.annotation_scene.update_label(self._cam_name, self.active_faces)
    def _update_hand_layer(self):
        self.logger.debug('_update_hand_layer')
        for side, hand_mesh in self.hand_geometry.items():
            indices = self.hand_models[side].faces[self.active_faces[side]]
            inlier_idx = torch.unique(indices).tolist()
            active = hand_mesh.select_by_index(inlier_idx)
            self._add_geometry(self._active_mesh_name.format(side), active, material=self.settings.active_mesh_material)
            if self._show_hands.checked:
                nonactive = hand_mesh.__copy__()
                nonactive.remove_triangles_by_index(self.active_faces[side])
                self._add_geometry(self._hand_mesh_name.format(side), nonactive, material=self.settings.hand_mesh_material)
            else:
                self._remove_geometry(self._hand_mesh_name.format(side))
    def _toggle_hand_visible(self):
        self.logger.debug('_toggle_hand_visible')
        show = self._show_hands.checked
        self._show_hands.checked = not show
        self._on_show_hand(not show)
    def _on_key(self, event):
        if event.key == gui.KeyName.T and event.type == gui.KeyEvent.DOWN:
            self._on_initial_viewpoint()
            return gui.Widget.EventCallbackResult.HANDLED
        
        if event.key==gui.KeyName.Z and event.type==gui.KeyEvent.DOWN:
            self._toggle_hand_visible()
            return gui.Widget.EventCallbackResult.CONSUMED
        
        if self.annotation_scene is None:
            return gui.Widget.EventCallbackResult.IGNORED
        
        if event.key==gui.KeyName.B and event.type==gui.KeyEvent.DOWN:
            self._toggle_labeling_mode()

        # if press R then reset label
        if event.key==gui.KeyName.R and event.type==gui.KeyEvent.DOWN:
            self._reset_label()
            self._update_hand_layer()
        
        # previous, next camera
        if event.key==gui.KeyName.A and event.type==gui.KeyEvent.DOWN:
            self._on_previous_camera()
        if event.key==gui.KeyName.D and event.type==gui.KeyEvent.DOWN:
            self._on_next_camera()
        
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
        
        if event.key==gui.KeyName.F and event.type==gui.KeyEvent.DOWN:
            self._on_save_label()
            return gui.Widget.EventCallbackResult.HANDLED
        
        return gui.Widget.EventCallbackResult.IGNORED
    def _on_tick(self):
        if (time.time()-self._last_change) > 1:
            if self.annotation_scene is None:
                self._log.text = "\t라벨링 대상 파일을 선택하세요."
                self.window.set_needs_layout()
            else:
                self._log.text = "라벨링 중입니다."
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

    #region ----- Open3DScene 
    #----- geometry
    def _check_geometry(self, name):
        return self._scene.scene.has_geometry(name)
    
    def _remove_geometry(self, name):
        if self._check_geometry(name):
            self._scene.scene.remove_geometry(name)
    
    def _add_geometry(self, name, geo, material=None):
        self._remove_geometry(name)
        self._scene.scene.add_geometry(name, geo, material,
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