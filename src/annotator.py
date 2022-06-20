# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

from email.mime import base
import os
import cv2
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import json
import threading
import time
import numpy as np
from pathlib import Path
from os.path import basename, dirname, abspath
import warnings
from src.utils import *
import glob

def tail(stream_file):
    """ Read a file like the Unix command `tail`. Code from https://stackoverflow.com/questions/44895527/reading-infinite-stream-tail """
    stream_file.seek(0, os.SEEK_END)  # Go to the end of file
    while not stream_file.closed:
        line = stream_file.readline()
        yield line

class dataset:
    def __init__(self, dataset_path, dataset_split):
        self.scenes_path = os.path.join(dataset_path, dataset_split)
        self.objects_path = os.path.join(dataset_path, 'models')

class annotationScene:
    def __init__(self, scene_point_cloud, scene_num, image_num):
        self.annotation_scene = scene_point_cloud
        self.scene_num = scene_num
        self.image_num = image_num

        self.obj_list = list()

    def add_obj(self, obj_geometry, obj_name, obj_instance, transform=np.identity(4)):
        self.obj_list.append(self.SceneObject(obj_geometry, obj_name, obj_instance, transform))

    def get_objects(self):
        return self.obj_list[:]

    def remove_obj(self, index):
        self.obj_list.pop(index)

    class SceneObject:
        def __init__(self, obj_geometry, obj_name, obj_instance, transform):
            self.obj_geometry = obj_geometry
            self.obj_name = obj_name
            self.obj_instance = obj_instance
            self.transform = transform


class settings:
    UNLIT = "defaultUnlit"

    def __init__(self):
        self.bg_color = gui.Color(1, 1, 1)
        self.show_axes = False
        self.highlight_obj = True

        self.apply_material = True  # clear to False after processing

        self.scene_material = rendering.MaterialRecord()
        self.scene_material.base_color = [0.9, 0.9, 0.9, 1.0]
        self.scene_material.shader = settings.UNLIT

        self.annotation_obj_material = rendering.MaterialRecord()
        self.annotation_obj_material.base_color = [0.9, 0.3, 0.3, 1.0]
        self.annotation_obj_material.shader = settings.UNLIT


class objectPoseAnnotator():

    def __init__(self):

        self.is_done = False
        self.n_snapshots = 0
        self.cloud = None
        self.main_vis = None
        self.snapshot_pos = None

        self.annoDataPath = None
        self.annoObjectPath = None
        self.keyInput = None

        # flags
        self.keyPress = False
        self.selectAnnoData = False
        self.openObjectDir = False
        self.addAnnoObject = False

        self.sceneCloudName = "scene_points"
        self.objectCloudName = "object_points"
        self._meshes_used_idx = 0
        self.dist = 0.005
        self.deg = 1

        # open3d GUI
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        self.settings = settings()
        self.window = gui.Application.instance.create_window(
            "BOP manual annotation tool", 1200, 900)
        w = self.window  # to make the code more concise
        
        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        # ---- Settings panel ----
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        view_ctrls = gui.CollapsableVert("View control", 0,
                                         gui.Margins(em, 0, 0, 0))
        view_ctrls.set_is_open(True)

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_child(self._show_axes)

        self._highlight_obj = gui.Checkbox("Highligh annotation objects")
        self._highlight_obj.set_on_checked(self._on_highlight_obj)
        view_ctrls.add_child(self._highlight_obj)

        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 5)
        self._point_size.set_on_value_changed(self._on_point_size)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        view_ctrls.add_child(grid)

        self._settings_panel.add_child(view_ctrls)
        # ----

        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)


        self._annotation_scene = None


        threading.Thread(target=self.getData, daemon=True).start()
        threading.Thread(target=self.update_thread, daemon=True).start()

        app.run()

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_axes(self.settings.show_axes)

        if self.settings.apply_material:
            self._scene.scene.modify_geometry_material("annotation_scene", self.settings.scene_material)
            self.settings.apply_material = False

        self._show_axes.checked = self.settings.show_axes
        self._highlight_obj.checked = self.settings.highlight_obj
        self._point_size.double_value = self.settings.scene_material.point_size

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_highlight_obj(self, light):
        self.settings.highlight_obj = light
        if light:
            self.settings.annotation_obj_material.base_color = [0.9, 0.3, 0.3, 1.0]
        elif not light:
            self.settings.annotation_obj_material.base_color = [0.9, 0.9, 0.9, 1.0]

        self._apply_settings()

        # update current object visualization
        meshes = self._annotation_scene.get_objects()
        for mesh in meshes:
            self._scene.scene.modify_geometry_material(mesh.obj_name, self.settings.annotation_obj_material)

    def _on_point_size(self, size):
        self.settings.scene_material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()


    def _add_mesh(self):
        meshes = self._annotation_scene.get_objects()
        meshes = [i.obj_name for i in meshes]

        object_geometry = o3d.io.read_point_cloud(self.annoObjectPath)
        object_geometry.points = o3d.utility.Vector3dVector(
            np.array(object_geometry.points) / 1000)  # convert mm to meter
        init_trans = np.identity(4)
        center = self._annotation_scene.annotation_scene.get_center()
        center[2] -= 0.2
        init_trans[0:3, 3] = center
        object_geometry.transform(init_trans)
        new_mesh_instance = self._obj_instance_count(self.annoObjectName , meshes)
        new_mesh_name = self.annoObjectName + '_' + str(new_mesh_instance)
        self._scene.scene.add_geometry(new_mesh_name, object_geometry, self.settings.annotation_obj_material,
                                       add_downsampled_copy_for_fast_rendering=True)
        self._annotation_scene.add_obj(object_geometry, new_mesh_name, new_mesh_instance, transform=init_trans)
        meshes = self._annotation_scene.get_objects()  # update list after adding current object
        meshes = [i.obj_name for i in meshes]
        self._meshes_used = meshes
        print("===>", meshes)
        self._meshes_used_idx = len(meshes) - 1

    def _remove_mesh(self):
        if not self._annotation_scene.get_objects():
            print("There are no object to be deleted.")
            return
        meshes = self._annotation_scene.get_objects()
        active_obj = meshes[self._meshes_used.selected_index]
        self._scene.scene.remove_geometry(active_obj.obj_name)  # remove mesh from scene
        self._annotation_scene.remove_obj(self._meshes_used.selected_index)  # remove mesh from class list
        # update list after adding removing object
        meshes = self._annotation_scene.get_objects()  # get new list after deletion
        meshes = [i.obj_name for i in meshes]
        self._meshes_used = meshes

    def _transform(self, keyInput):

        def move(x, y, z, rx, ry, rz):
            self._annotation_changed = True

            objects = self._annotation_scene.get_objects()
            active_obj = objects[self._meshes_used_idx]
            # translation or rotation
            if x != 0 or y != 0 or z != 0:
                h_transform = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
            else:  # elif rx!=0 or ry!=0 or rz!=0:
                center = active_obj.obj_geometry.get_center()
                rot_mat_obj_center = active_obj.obj_geometry.get_rotation_matrix_from_xyz((rx, ry, rz))
                T_neg = np.vstack((np.hstack((np.identity(3), -center.reshape(3, 1))), [0, 0, 0, 1]))
                R = np.vstack((np.hstack((rot_mat_obj_center, [[0], [0], [0]])), [0, 0, 0, 1]))
                T_pos = np.vstack((np.hstack((np.identity(3), center.reshape(3, 1))), [0, 0, 0, 1]))
                h_transform = np.matmul(T_pos, np.matmul(R, T_neg))
            active_obj.obj_geometry.transform(h_transform)
            center = active_obj.obj_geometry.get_center()
            print("remove")
            self._scene.scene.remove_geometry(active_obj.obj_name)
            print("add")
            self._scene.scene.add_geometry(active_obj.obj_name, active_obj.obj_geometry,
                                           self.settings.annotation_obj_material,
                                           add_downsampled_copy_for_fast_rendering=True)
            # update values stored of object
            active_obj.transform = np.matmul(h_transform, active_obj.transform)
        if keyInput not in ["i", "j", "k", "l", "u", "o", "r", "t", "f", "g", "d"]:
            return
        
        if keyInput == "g":
            self.dist = 0.05
            self.deg = 90
        elif keyInput == "f":
            self.dist = 0.005
            self.deg = 1

        elif keyInput == "d":
            self._on_refine()
            # Translation
        elif keyInput == "l":
            print("L pressed: translate in +ve X direction")
            move(self.dist, 0, 0, 0, 0, 0)
        elif keyInput == "j":
            print("J pressed: translate in -ve X direction")
            move(-self.dist, 0, 0, 0, 0, 0)
        elif keyInput == "k":
            print("K pressed: translate in +ve Y direction")
            move(0, self.dist, 0, 0, 0, 0)
        elif keyInput == "i":
            print("I pressed: translate in -ve Y direction")
            move(0, -self.dist, 0, 0, 0, 0)
        elif keyInput == "o":
            print("O pressed: translate in +ve Z direction")
            move(0, 0, self.dist, 0, 0, 0)
        elif keyInput == "u":
            print("U pressed: translate in -ve Z direction")
            move(0, 0, -self.dist, 0, 0, 0)
            # Rotation - keystrokes are not in same order as translation to make movement more human intuitive
        # elif keyInput == "":
        #     print("L pressed: rotate around +ve X direction")
        #     move(0, 0, 0, 0, 0, deg * np.pi / 180)
        # elif keyInput == "":
        #     print("H pressed: rotate around -ve X direction")
        #     move(0, 0, 0, 0, 0, -deg * np.pi / 180)
        # elif keyInput == "":
        #     print("K pressed: rotate around +ve Y direction")
        #     move(0, 0, 0, 0, deg * np.pi / 180, 0)
        # elif keyInput == "":
        #     print("J pressed: rotate around -ve Y direction")
        #     move(0, 0, 0, 0, -deg * np.pi / 180, 0)
        # elif keyInput == "":
        #     print("Comma pressed: rotate around +ve Z direction")
        #     move(0, 0, 0, deg * np.pi / 180, 0, 0)
        # elif keyInput == "":
        #     print("I pressed: rotate around -ve Z direction")
        #     move(0, 0, 0, -deg * np.pi / 180, 0, 0)

        return 


    def _make_point_cloud(self, rgb_img, depth_img, cam_K):
        # convert images to open3d types
        rgb_img_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        depth_img_o3d = o3d.geometry.Image(depth_img)

        # convert image to point cloud
        intrinsic = o3d.camera.PinholeCameraIntrinsic(rgb_img.shape[0], rgb_img.shape[1],
                                                      cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img_o3d, depth_img_o3d,
                                                                  depth_scale=1, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        return pcd

    def _obj_instance_count(self, mesh_to_add, meshes):
        types = [i[:-2] for i in meshes]  # remove last 3 character as they present instance number (OBJ_INSTANCE)
        equal_values = [i for i in range(len(types)) if types[i] == mesh_to_add]
        count = 0
        if len(equal_values):
            indices = np.array(meshes)
            indices = indices[equal_values]
            indices = [int(x[-1]) for x in indices]
            count = max(indices) + 1
            # TODO change to fill the numbers missing in sequence
        return count

    def on_main_window_closing(self):
        self.is_done = True
        return True  # False would cancel the close

    def scene_load(self, scenes_path, scene_num, image_num):
        self._annotation_changed = False

        self._scene.scene.clear_geometry()
        geometry = None

        scene_path = os.path.join(scenes_path, f'{scene_num:06}')

        camera_params_path = os.path.join(scene_path, 'scene_camera.json')
        with open(camera_params_path) as f:
            data = json.load(f)
            cam_K = data[str(image_num)]['cam_K']
            cam_K = np.array(cam_K).reshape((3, 3))
            depth_scale = data[str(image_num)]['depth_scale']

        rgb_path = os.path.join(scene_path, 'rgb', f'{image_num:06}' + '.png')
        rgb_img = cv2.imread(rgb_path)
        depth_path = os.path.join(scene_path, 'depth', f'{image_num:06}' + '.png')
        depth_img = cv2.imread(depth_path, -1)
        depth_img = np.float32(depth_img * depth_scale / 1000)

        try:
            geometry = self._make_point_cloud(rgb_img, depth_img, cam_K)
        except Exception:
            print("Failed to load scene.")

        if geometry is not None:
            print("[Info] Successfully read scene ", scene_num)
            if not geometry.has_normals():
                geometry.estimate_normals()
            geometry.normalize_normals()
        else:
            print("[WARNING] Failed to read points")

        # try:
        self._scene.scene.add_geometry("annotation_scene", geometry, self.settings.scene_material,
                                        add_downsampled_copy_for_fast_rendering=True)
        bounds = geometry.get_axis_aligned_bounding_box()
        self._scene.setup_camera(60, bounds, bounds.get_center())
        center = np.array([0, 0, 0])
        eye = center + np.array([0, 0, -0.5])
        up = np.array([0, -1, 0])
        self._scene.look_at(center, eye, up)

        self._annotation_scene = annotationScene(geometry, scene_num, image_num)
        self._meshes_used = []  # clear list from last loaded scene

        # load values if an annotation already exists

        model_names = self.load_model_names()
        scene_gt_path = os.path.join(self.annoScenesPath, f"{self._annotation_scene.scene_num:06}",
                                        'scene_gt.json')
        # if os.path.exists(json_path):
        with open(scene_gt_path) as scene_gt_file:
            data = json.load(scene_gt_file)
            scene_data = data[str(image_num)]
            active_meshes = list()
            for obj in scene_data:
                # add object to annotation_scene object
                obj_geometry = o3d.io.read_point_cloud(
                    os.path.join(self.annoObjectDirPath, 'obj_' + f"{int(obj['obj_id']):06}" + '.ply'))
                obj_geometry.points = o3d.utility.Vector3dVector(
                    np.array(obj_geometry.points) / 1000)  # convert mm to meter
                model_name = 'obj_' + f'{ + obj["obj_id"]:06}'
                obj_instance = self._obj_instance_count(model_name, active_meshes)
                obj_name = model_name + '_' + str(obj_instance)
                translation = np.array(np.array(obj['cam_t_m2c']), dtype=np.float64) / 1000  # convert to meter
                orientation = np.array(np.array(obj['cam_R_m2c']), dtype=np.float64)
                transform = np.concatenate((orientation.reshape((3, 3)), translation.reshape(3, 1)), axis=1)
                transform_cam_to_obj = np.concatenate(
                    (transform, np.array([0, 0, 0, 1]).reshape(1, 4)))  # homogeneous transform

                self._annotation_scene.add_obj(obj_geometry, obj_name, obj_instance, transform_cam_to_obj)
                # adding object to the scene
                obj_geometry.translate(transform_cam_to_obj[0:3, 3])
                center = obj_geometry.get_center()
                obj_geometry.rotate(transform_cam_to_obj[0:3, 0:3], center=center)
                self._scene.scene.add_geometry(obj_name, obj_geometry, self.settings.annotation_obj_material,
                                                add_downsampled_copy_for_fast_rendering=True)
                active_meshes.append(obj_name)
        self._meshes_used = active_meshes

        # except Exception as e:
        #     print(e)

    def load_model_names(self):
    
        obj_ids = sorted([int(basename(x)[5:-4]) for x in glob.glob(self.annoObjectDirPath + '/*.ply')])
        model_names = ['obj_' + f'{ + obj_id:06}' for obj_id in obj_ids]
        return model_names

    # def loadScene(self, annoDataPath):
    #     rgbPath = self.annoRGBPath
    #     depthPath = self.annoDepthPath
    #     sceneCameraPath = self.annoSceneCameraPath

    #     color_raw = o3d.io.read_image(rgbPath)
    #     width, height = color_raw.get_max_bound()
    #     self.width, self.height = int(width), int(height)
    #     depth_raw = o3d.io.read_image(depthPath)
    #     with open(sceneCameraPath, "r") as f:
    #         scene_camera_info = json.load(f)
    #     K = scene_camera_info["3"]["cam_K"]
    #     self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    #         width=self.width, height=self.height, fx=K[0], fy=K[4], cx=K[2], cy=K[5])

    #     im_color = np.asarray(color_raw)
    #     self.im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
    #     self.im_depth = np.asarray(depth_raw)
    #     self.x, self.y = im_color.shape[1] // 2, im_color.shape[0] // 2

    #     rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, 1000.0, 2.0, convert_rgb_to_intensity=False)
    #     self.sceneCloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.camera_intrinsic)

    #     self._scene.scene.add_geometry("annotation_scene", self.sceneCloud, self.settings.scene_material,
    #                                     add_downsampled_copy_for_fast_rendering=True)
    #     bounds = self.sceneCloud.get_axis_aligned_bounding_box()
    #     self._scene.setup_camera(60, bounds, bounds.get_center())
    #     center = np.array([0, 0, 0])
    #     eye = center + np.array([0, 0, -0.5])
    #     up = np.array([0, -1, 0])
    #     self._scene.look_at(center, eye, up)


    def loadObject(self, annoObjectPath):
        self.objectCloud = o3d.io.read_point_cloud(annoObjectPath)
        self.objectCloud = scaleCloud(self.objectCloud, 0.001)


    def update_thread(self):
        # This is NOT the UI thread, need to call post_to_main_thread() to update
        # the scene or any part of the UI.

        while not self.is_done:

            if self.keyPress:
                self.keyPress = False
                print("_transform start")
                self._transform(self.keyInput)
                print("_transform end")
                pass

            if self.selectAnnoData:
                self.selectAnnoData = False
                scene_num = int(basename((Path(self.annoRGBPath).parent.parent)))
                image_num = int(basename(self.annoRGBPath)[:-4])
                self.scene_load(self.annoScenesPath, scene_num, image_num)
                # o3d.visualization.gui.Application.instance.post_to_main_thread(
                #  self.main_vis, updateSceneCloud)
                pass

            if self.openObjectDir:
                self.openObjectDir = False
                pass

            if self.addAnnoObject:
                self._add_mesh()
                # o3d.visualization.gui.Application.instance.post_to_main_thread(
                #  self.main_vis, addObjectCloud)
                self.addAnnoObject = False
                pass

            if self.is_done:  # might have changed while sleeping
                break

        def addSceneCloud():

            self.loadScene(self.annoDataPath)
            bounds = self.sceneCloud.get_axis_aligned_bounding_box()
            extent = bounds.get_extent()
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            self.main_vis.add_geometry(self.sceneCloudName, self.sceneCloud, mat)
            self.main_vis.reset_camera_to_default()
            self.main_vis.setup_camera(60, bounds.get_center(),
                                       bounds.get_center() + [0, 0, -3],
                                       [0, -1, 0])

        def updateSceneCloud():
            self.main_vis.remove_geometry(self.sceneCloudName)
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            self.main_vis.add_geometry(self.sceneCloudName, self.sceneCloud, mat)

        def addObjectCloud():
            try:
                self.main_vis.remove_geometry(self.objectCloudName)
            except:
                pass
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            self.main_vis.add_geometry(self.objectCloudName, self.objectCloud, mat)

        o3d.visualization.gui.Application.instance.post_to_main_thread(
            self.main_vis, addSceneCloud)
        self.updateScene = False


    def getData(self):

        with open("./comm.json", "r") as file:
            while not self.is_done:
                for line in tail(file):
                    if len(line) == 0:
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError:
                        # Bad json format, maybe corrupted...
                        continue  # Read next line

                    print("recv:", data)
                    action = data["action"]
                    data = data["data"]
                    if action == "keyPress":
                        self.keyPress = True
                        self.keyInput = data["keyInput"]
                    elif action == "selectAnnoData":
                        self.selectAnnoData = True
                        self.annoRGBPath = data["annoRgbPath"]
                        self.annoDepthPath = data["annoDepthPath"]
                        self.annoSceneCameraPath = data["annoSceneCameraPath"]
                        self.annoScenesPath = data["annoScenesPath"]
                        self.annoObjectDirPath = data["annoObjectDirPath"]
                    elif action == "openObjectDir":
                        self.openObjectDir = True
                        self.annoObjectDirPath = data["annoObjectDirPath"]
                    elif action == "addAnnoObject":
                        self.addAnnoObject = True
                        self.annoObjectPath = data["annoObjectPath"]
                        self.annoObjectName = basename(self.annoObjectPath).split(".")[0]
                    else:
                        print("Not defined action: {}".format(action))


def runObjectPoseAnnotator():

    o3d.visualization.webrtc_server.enable_webrtc()
    p = objectPoseAnnotator()




if __name__ == "__main__":
    
    runObjectPoseAnnotator()