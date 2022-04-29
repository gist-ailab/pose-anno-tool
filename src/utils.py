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
import imgviz
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

class Mapping():
    def __init__(self, camera_intrinsic, _w=640, _h=480, _d=1000.0 ):
        self.camera_intrinsic = camera_intrinsic
        self.width = _w
        self.height = _h
        self.d = _d
        self.camera_intrinsic4x4 = np.identity(4)
        self.camera_intrinsic4x4[0,0] = self.camera_intrinsic.intrinsic_matrix[0,0]
        self.camera_intrinsic4x4[1,1] = self.camera_intrinsic.intrinsic_matrix[1,1]
        self.camera_intrinsic4x4[0,3] = self.camera_intrinsic.intrinsic_matrix[0,2]
        self.camera_intrinsic4x4[1,3] = self.camera_intrinsic.intrinsic_matrix[1,2]
        
    def showCameraIntrinsic(self):
        print(self.camera_intrinsic.intrinsic_matrix)
        print(self.camera_intrinsic4x4)

    def Cloud2Image( self, cloud_in ):
        
        img = np.zeros( [self.height, self.width], dtype=np.uint8 )
        img_zero = np.zeros( [self.height, self.width], dtype=np.uint8 )
        
        cloud_np1 = np.asarray(cloud_in.points)
        sorted_indices = np.argsort(cloud_np1[:,2])[::-1]
        cloud_np = cloud_np1[sorted_indices]
        cloud_np_xy = cloud_np[:,0:2] / cloud_np[:,[2]]
        # cloud_np ... (x/z, y/z, z)
        cloud_np = np.hstack((cloud_np_xy,cloud_np[:,[2]])) 

        cloud_color1 = np.asarray(cloud_in.colors)
        
        cloud_mapped = o3d.geometry.PointCloud()
        cloud_mapped.points = o3d.utility.Vector3dVector(cloud_np)
        cloud_mapped.transform(self.camera_intrinsic4x4)


        """ If cloud_in has the field of color, color is mapped into the image. """
        print(len(cloud_color1), len(cloud_np))
        if len(cloud_color1) == len(cloud_np):
            cloud_color = cloud_color1[sorted_indices]
            print(cloud_color)
            img = cv2.merge((img,img,img))
            for i, pix in enumerate(cloud_mapped.points):
                if pix[0]<self.width and 0<pix[0] and pix[1]<self.height and 0<pix[1]:
                    color = (cloud_color[i]*255.0).astype(np.uint8)
                    img[int(pix[1]),int(pix[0])] = [color[2], color[1], color[0]]

        else:
            for i, pix in enumerate(cloud_mapped.points):
                if pix[0]<self.width and 0<pix[0] and pix[1]<self.height and 0<pix[1]:
                    img[int(pix[1]),int(pix[0])] = int(255.0*(cloud_np[i,2]/cloud_np[0,2]))

            img = cv2.merge((img_zero,img,img_zero))
        
        return img
    
    def Pix2Pnt( self, pix, val ):
        pnt = np.array([0.0,0.0,0.0], dtype=np.float)
        depth = val / self.d
        #print('[0,2]: {}'.format(self.camera_intrinsic.intrinsic_matrix[0,2]))
        #print('[1,2]: {}'.format(self.camera_intrinsic.intrinsic_matrix[1,2]))
        #print(self.camera_intrinsic.intrinsic_matrix)
        pnt[0] = (float(pix[0]) - self.camera_intrinsic.intrinsic_matrix[0,2]) * depth / self.camera_intrinsic.intrinsic_matrix[0,0]
        pnt[1] = (float(pix[1]) - self.camera_intrinsic.intrinsic_matrix[1,2]) * depth / self.camera_intrinsic.intrinsic_matrix[1,1]
        pnt[2] = depth

        return pnt


def update_position(x, y, w_name, img_c, img_d, mapping):


    """Direct move. Object model will be moved to clicked position."""
    global all_transformation
    print('Clicked({},{}): depth:{}'.format(x, y, img_d[y,x]))
    print(img_d[y,x])
    pnt = mapping.Pix2Pnt( [x,y], img_d[y,x] )
    print('3D position is', pnt)

    #compute current center of the cloud
    cloud_c = copy.deepcopy(CLOUD_ROT)
    cloud_c, center = c3D.Centering(cloud_c)
    np_cloud = np.asarray(cloud_c.points) 

    np_cloud += pnt
    print('Offset:', pnt )
    offset = np.identity(4)
    offset[0:3,3] -= center
    offset[0:3,3] += pnt
    all_transformation = np.dot( offset, all_transformation )

    # CLOUD_ROT.points = o3d.utility.Vector3dVector(np_cloud)
    # generateImage( mapping, img_c )


# Pose refinement by ICP
def refine_registration(source, target, trans, voxel_size):
    global all_transformation
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-point ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(source, target, 
                                            distance_threshold, trans,
                                            o3d.pipelines.registration.TransformationEstimationPointToPoint())

    return result.transformation


def mouse_event(event, x, y, flags, param):
    w_name, img_c, img_d, mapping = param

    """Direct move. Object model will be moved to clicked position."""
    if event == cv2.EVENT_LBUTTONUP:
        global all_transformation
        print('Clicked({},{}): depth:{}'.format(x, y, img_d[y,x]))
        print(img_d[y,x])
        pnt = mapping.Pix2Pnt( [x,y], img_d[y,x] )
        print('3D position is', pnt)

        #compute current center of the cloud
        cloud_c = copy.deepcopy(CLOUD_ROT)
        cloud_c, center = c3D.Centering(cloud_c)
        np_cloud = np.asarray(cloud_c.points) 

        np_cloud += pnt
        print('Offset:', pnt )
        offset = np.identity(4)
        offset[0:3,3] -= center
        offset[0:3,3] += pnt
        all_transformation = np.dot( offset, all_transformation )

        # CLOUD_ROT.points = o3d.utility.Vector3dVector(np_cloud)
        # generateImage( mapping, img_c )




