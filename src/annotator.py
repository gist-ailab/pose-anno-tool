import open3d as o3d 
import numpy as np
import cv2
import copy
import argparse
import os
import common3Dfunc as c3D
from math import *
import json
import glob
import vedo
from src.visualizer import *
from src.loader import *
from src.utils import *
import dearpygui.dearpygui as dpg
import open3d.visualization.rendering as rendering

class Annotator():

    def __init__(self):

        """ Object model to be transformed """
        self.CLOUD_ROT = o3d.geometry.PointCloud()
        """ Total transformation"""
        self.all_transformation = np.identity(4)
        """ Step size for rotation """
        self.rot_step = 0.05*pi
        self.pos_step = 5
        """ Voxel size for downsampling"""
        self.voxel_size = 0.001
        self.img_mapped = np.zeros([480, 640, 3])
        self.trasparency = 0.5

        self.load_objects()



    def load_objects(self):

        model_path = "/home/seung/Workspace/custom/6DPoseAnnotator/data/lm/models"
        rgb_path = "/home/seung/Workspace/custom/6DPoseAnnotator/data/lm/test/000002/rgb/000003.png"
        depth_path = "/home/seung/Workspace/custom/6DPoseAnnotator/data/lm/test/000002/depth/000003.png"
        scene_camera_path = "/home/seung/Workspace/custom/6DPoseAnnotator/data/lm/test/000002/scene_camera.json"


        self.obj_ply_paths = sorted(glob.glob(model_path + "/*.ply"))
        self.obj_names = [os.path.basename(p).split(".")[0] for p in self.obj_ply_paths]
        obj_list_img = create_obj_list_img(self.obj_ply_paths, self.obj_names)

        color_raw = o3d.io.read_image(rgb_path)
        width, height = color_raw.get_max_bound()
        self.width, self.height = int(width), int(height)
        depth_raw = o3d.io.read_image(depth_path)
        with open(scene_camera_path, "r") as f:
            scene_camera_info = json.load(f)
        K = scene_camera_info["3"]["cam_K"]
        self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.width, height=self.height, fx=K[0], fy=K[4], cx=K[2], cy=K[5])

        im_color = np.asarray(color_raw)
        self.im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB )
        self.im_depth = np.asarray(depth_raw)
        self.x, self.y = im_color.shape[1] // 2, im_color.shape[0] // 2

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, 1000.0, 2.0, convert_rgb_to_intensity=False)
        self.o3dpc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.camera_intrinsic)

        # np_pcd = np.asarray(self.obj_o3dpc.points)
        


        self.object_select_button_callback(self.obj_names[0], None)
        self.update_2d_anno_vis()


    def object_select_button_callback(self, sender, app_data):

        print(f"{sender} is selected.")
        object_model_path = [p for p in self.obj_ply_paths if os.path.basename(p).split(".")[0] == sender][0]
        cloud_m = o3d.io.read_point_cloud(object_model_path)
        colors = cloud_m.colors

        """ if you use object model with meter scale, try this code to convert meter scale."""
        cloud_m = c3D.Scaling(cloud_m, 0.001 )
        cloud_m_ds = cloud_m
        # cloud_m_ds = cloud_m.voxel_down_sample( voxel_size )


        """Loading of the initial transformation"""
        initial_trans = np.identity(4)
        # if initial transformation is not avairable, 
        # the object model is moved to its center.
        cloud_m_c, offset = c3D.Centering( cloud_m_ds )
        mat_centering = c3D.makeTranslation4x4( -1.0*offset )
        self.all_transformation = np.dot( mat_centering, self.all_transformation )

        cloud_m_ds.colors = colors
        self.CLOUD_ROT = copy.deepcopy(cloud_m_ds)
        self.CLOUD_ROT.transform(self.all_transformation)
        self.mapping = Mapping(self.camera_intrinsic, self.width, self.height)
        self.img_mapped = self.mapping.Cloud2Image(self.CLOUD_ROT)
        o3d.visualization.draw_geometries([self.o3dpc], width=640*2, height=480, left=1720, top=800)

    def update_position(self):

        # Direct move. Object model will be moved to clicked position.""
        pnt = self.mapping.Pix2Pnt( [self.x, self.y], self.im_depth[self.y, self.x] )

        #compute current center of the cloud
        cloud_c = copy.deepcopy(self.CLOUD_ROT)
        cloud_c, center = c3D.Centering(cloud_c)
        np_cloud = np.asarray(cloud_c.points) 

        np_cloud += pnt
        offset = np.identity(4)
        offset[0:3,3] -= center
        offset[0:3,3] += pnt
        self.all_transformation = np.dot( offset, self.all_transformation )
        self.CLOUD_ROT.points = o3d.utility.Vector3dVector(np_cloud)

    def mouse_click_callback(self, sender, app_data):
        pos = dpg.get_mouse_pos(local=True)
        self.x, self.y = int(pos[0]), int(pos[1])
        pnt = self.mapping.Pix2Pnt( [self.x, self.y], self.im_depth[self.y, self.x] )

        #compute current center of the cloud
        cloud_c = copy.deepcopy(self.CLOUD_ROT)
        cloud_c, center = c3D.Centering(cloud_c)
        np_cloud = np.asarray(cloud_c.points) 

        np_cloud += pnt
        offset = np.identity(4)
        offset[0:3,3] -= center
        offset[0:3,3] += pnt
        self.all_transformation = np.dot( offset, self.all_transformation )

        self.CLOUD_ROT.points = o3d.utility.Vector3dVector(np_cloud)
        self.update_2d_anno_vis()
        self.update_3d_object_vis()
        self.update_3d_anno_vis()


    def keyboard_press_callback(self, sender, app_data):
        if app_data == 87:
            print("w")
            rotation = c3D.ComputeTransformationMatrixAroundCentroid(self.CLOUD_ROT, self.rot_step, 0, 0 )
            self.CLOUD_ROT.transform(rotation)
            self.all_transformation = np.dot(rotation, self.all_transformation)
            self.update_2d_anno_vis()
        if app_data == 83:
            print("s")
            rotation = c3D.ComputeTransformationMatrixAroundCentroid( self.CLOUD_ROT, -self.rot_step, 0, 0 )
            self.CLOUD_ROT.transform(rotation)
            self.all_transformation = np.dot(rotation, self.all_transformation)
            self.update_2d_anno_vis()
        if app_data == 65:
            print("a")
            rotation = c3D.ComputeTransformationMatrixAroundCentroid( self.CLOUD_ROT, 0, self.rot_step, 0 )
            self.CLOUD_ROT.transform(rotation)
            self.all_transformation = np.dot(rotation, self.all_transformation)
            self.update_2d_anno_vis()
        if app_data == 68:
            print("d")
            rotation = c3D.ComputeTransformationMatrixAroundCentroid( self.CLOUD_ROT, 0, -self.rot_step, 0 )
            self.CLOUD_ROT.transform(rotation)
            self.all_transformation = np.dot(rotation, self.all_transformation)
            self.update_2d_anno_vis()
        if app_data == 81:
            print("q")
            rotation = c3D.ComputeTransformationMatrixAroundCentroid( self.CLOUD_ROT, 0, 0, self.rot_step,)
            self.CLOUD_ROT.transform(rotation)
            self.all_transformation = np.dot(rotation, self.all_transformation)
            self.update_2d_anno_vis()
        if app_data == 69:
            print("e")
            rotation = c3D.ComputeTransformationMatrixAroundCentroid( self.CLOUD_ROT, 0, 0, -self.rot_step,)
            self.CLOUD_ROT.transform(rotation)
            self.all_transformation = np.dot(rotation, self.all_transformation)
            self.update_2d_anno_vis()
        if app_data == 73:
            self.y -= self.pos_step
            self.update_position()
            self.update_2d_anno_vis()
            print("i")
        if app_data == 75:
            self.y += self.pos_step
            self.update_position()
            self.update_2d_anno_vis()
            print("k")
        if app_data == 74:
            self.x -= self.pos_step
            self.update_position()
            self.update_2d_anno_vis()
            print("j")
        if app_data == 76:
            self.x += self.pos_step
            self.update_position()
            self.update_2d_anno_vis()
            print("l")
        if app_data == 85:
            print("u")
            print('ICP start (coarse mode)')
            result_icp = refine_registration(self.CLOUD_ROT, self.o3dpc, np.identity(4), 5.0*self.voxel_size)
            self.CLOUD_ROT.transform( result_icp )
            self.all_transformation = np.dot( result_icp, self.all_transformation )
            self.update_2d_anno_vis()
        if app_data == 79:
            print("o")
            print('ICP start (fine mode)')
            result_icp = refine_registration(self.CLOUD_ROT, self.o3dpc, np.identity(4), 1.0*self.voxel_size)
            self.CLOUD_ROT.transform( result_icp )
            self.all_transformation = np.dot( result_icp, self.all_transformation )
            self.update_2d_anno_vis()
        if app_data == 67:
            print("c")
            print('3d visualization')
            o3d.visualization.draw_geometries([self.CLOUD_ROT, self.o3dpc], width=640*2, height=480, left=1720, top=800)


        self.update_3d_object_vis()


    def trasparency_callback(self, sender, appdata):
        self.trasparency = appdata
        self.update_2d_anno_vis()

    def update_2d_anno_vis(self):
        img_m = self.mapping.Cloud2Image(self.CLOUD_ROT )
        # !TODO: adjust traspancy
        img_mapped = cv2.addWeighted(img_m, self.trasparency, self.im_color, 1-self.trasparency, 0)
        frame = cv2.cvtColor(img_mapped, cv2.COLOR_BGR2RGBA).astype(float)/255
        dpg.set_value("2d_anno_vis", frame)

    def update_3d_object_vis(self):
        img_m = self.mapping.Cloud2Image(self.CLOUD_ROT)
        frame = cv2.cvtColor(img_m, cv2.COLOR_BGR2RGBA).astype(float)/255
        dpg.set_value("3d_object_vis", frame)



