# 6DoF pose annotator 
# Shuichi Akizuki, Chukyo Univ.
# Email: s-akizuki@sist.chukyo-u.ac.jp
#
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

""" Object model to be transformed """
CLOUD_ROT = o3d.geometry.PointCloud()
""" Total transformation"""
all_transformation = np.identity(4)
""" Step size for rotation """
step = 0.05*pi
""" Voxel size for downsampling"""
voxel_size = 0.001

def get_argumets():
    """
        Parse arguments from command line
    """

    parser = argparse.ArgumentParser( description='Interactive 6DoF pose annotator')
    parser.add_argument('--cimg', type=str, default='data/rgb.png',
                        help='file name of the RGB image of the input scene.')
    parser.add_argument('--dimg', type=str, default='data/depth.png',
                        help='file name of the depth image of the input scene. We assume that RGB and depth image have pixel-to-pixel correspondence.')
    parser.add_argument('--intrin', type=str, default='data/realsense_intrinsic.json',
                        help='file name of the camera intrinsic.')
    parser.add_argument('--model', type=str, default='data/hammer_mm.ply',
                        help='file name of the object model (.pcd or .ply).')
    parser.add_argument('--init', type=str, default='data/init.json',
                        help='file name of the initial transformation (.json).')
    
    return parser.parse_args()

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

    CLOUD_ROT.points = o3d.utility.Vector3dVector(np_cloud)
    generateImage( mapping, img_c )


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

        CLOUD_ROT.points = o3d.utility.Vector3dVector(np_cloud)
        generateImage( mapping, img_c )

def generateImage( mapping, im_color ):
    global CLOUD_ROT
    img_m = mapping.Cloud2Image( CLOUD_ROT )
    img_mapped = cv2.addWeighted(img_m, 0.5, im_color, 0.5, 0 )
    cv2.imshow( window_name, img_mapped )


def trim(im, border):
  bg = Image.new(im.mode, im.size, border)
  diff = ImageChops.difference(im, bg)
  bbox = diff.getbbox()
  if bbox:
    return im.crop(bbox)

def create_thumbnail(path, size):
  image = Image.open(path)
  name, extension = path.split('.')
  options = {}
  if 'transparency' in image.info:
    options['transparency'] = image.info["transparency"]
  
  image.thumbnail((size, size), Image.ANTIALIAS)
  image = trim(image, 255) ## Trim whitespace
  return image

import imgviz
import matplotlib.pyplot as plt
from PIL import Image, ImageChops


if __name__ == "__main__":

    args = get_argumets()

    model_path = "/home/seung/Workspace/custom/6DPoseAnnotator/data/lm/models"
    obj_model_paths = glob.glob(model_path + "/*.ply")
    imgs = []
    texture_paths = []
    obj_ply_paths = []
    for i, obj_ply_path in enumerate(obj_model_paths):
        # texture_path = os.path.join(model_path, f"obj_{i+1:06}.png")
        # obj_ply_path = os.path.join(model_path, f"obj_{i+1:06}.ply")
        # texture_paths.append(texture_path)
        obj_ply_paths.append(obj_ply_path)
        
        m = vedo.load(obj_ply_path)
        # m.texture(texture_path)
        m.show(interactive=False, viewup='z')
        vedo.screenshot('tmp.png')
        img = np.uint8(create_thumbnail('tmp.png', 256))
        img = cv2.putText(img, str(i+1),  (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        imgs.append(img)

    tiled = imgviz.tile(imgs=imgs, border=(255, 255, 255), cval=(255, 255, 255))
    plt.figure(dpi=700)
    plt.title("test")
    plt.imshow(tiled)
    plt.axis("off")
    img = imgviz.io.pyplot_to_numpy()
    plt.close()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("tmp/obj_list.png", img)
    obj_list_img = cv2.imread("tmp/obj_list.png")

    rgb_path = "/home/seung/Workspace/custom/6DPoseAnnotator/data/lm/test/000002/rgb/000003.png"
    depth_path = "/home/seung/Workspace/custom/6DPoseAnnotator/data/lm/test/000002/depth/000003.png"
    scene_camera_path = "/home/seung/Workspace/custom/6DPoseAnnotator/data/lm/test/000002/scene_camera.json"



    color_raw = o3d.io.read_image(rgb_path)
    width, height = color_raw.get_max_bound()
    width, height = int(width), int(height)
    depth_raw = o3d.io.read_image(depth_path)
    with open(scene_camera_path, "r") as f:
        scene_camera_info = json.load(f)
    K = scene_camera_info["3"]["cam_K"]
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=int(width), height=int(height), fx=K[0], fy=K[4], cx=K[2], cy=K[5])

    im_color = np.asarray(color_raw)
    im_color = cv2.cvtColor( im_color, cv2.COLOR_BGR2RGB )
    im_depth = np.asarray(depth_raw)

    im_color_obj = np.hstack([im_color, cv2.resize(obj_list_img, [height, height])])
    cv2.imshow("test", im_color_obj)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    obj_id = input("Enter: ")


    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth( color_raw, depth_raw, 1000.0, 2.0 )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic )
    o3d.io.write_point_cloud("tmp/cloud_in.ply", pcd)
    # cloud_in_ds = pcd.voxel_down_sample(voxel_size)
    cloud_in_ds = pcd
    o3d.io.write_point_cloud("tmp/cloud_in_ds.ply", cloud_in_ds)

    np_pcd = np.asarray(pcd.points)



    """Loading of the object model"""

    object_model_path = obj_ply_paths[int(obj_id)-1]
    cloud_m = o3d.io.read_point_cloud(object_model_path)
    colors = cloud_m.colors
    # object_mesh = o3d.io.read_triangle_mesh(object_model_path)
    # object_model = o3d.io.read_triangle_model(object_model_path)


    # cloud_m = object_mesh

    """ if you use object model with meter scale, try this code to convert meter scale."""
    cloud_m = c3D.Scaling( cloud_m, 0.001 )
    cloud_m_ds = cloud_m
    # cloud_m_ds = cloud_m.voxel_down_sample( voxel_size )


    """Loading of the initial transformation"""
    initial_trans = np.identity(4)
    if os.path.exists( args.init ):
        initial_trans = c3D.load_transformation( args.init )
        print('Use initial transformation\n', initial_trans )
        all_transformation = np.dot( initial_trans, all_transformation )
    else:
        # if initial transformation is not avairable, 
        # the object model is moved to its center.
        cloud_m_c, offset = c3D.Centering( cloud_m_ds )
        mat_centering = c3D.makeTranslation4x4( -1.0*offset )
        all_transformation = np.dot( mat_centering, all_transformation )


    cloud_m_ds.colors = colors
    CLOUD_ROT = copy.deepcopy(cloud_m_ds)
    CLOUD_ROT.transform( all_transformation )


    mapping = Mapping(camera_intrinsic, int(width), int(height))
    img_mapped = mapping.Cloud2Image( CLOUD_ROT )

    """Mouse event"""
    window_name = '6DoF Pose Annotator'
    cv2.namedWindow( window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback( window_name, mouse_event, 
                        [window_name, im_color, im_depth, mapping])
    x, y = 100, 100

    generateImage( mapping, im_color )
    while (True):
        key = cv2.waitKey(1) & 0xFF
        """Quit"""
        if key == 27:
            break

        if key == ord("l"):
            x += 1
            update_position(x, y, window_name, im_color, im_depth, mapping)

        if key == ord("j"):
            x -= 1
            update_position(x, y, window_name, im_color, im_depth, mapping)

        if key == ord("i"):
            y += 1
            update_position(x, y, window_name, im_color, im_depth, mapping)

        if key == ord("k"):
            y += 1
            update_position(x, y, window_name, im_color, im_depth, mapping)

        if key == ord("i"):
            print('ICP start (coarse mode)')
            result_icp = refine_registration( CLOUD_ROT, pcd, np.identity(4), 10.0*voxel_size)
            print(result_icp)
            CLOUD_ROT.transform( result_icp )
            all_transformation = np.dot( result_icp, all_transformation )
            generateImage( mapping, im_color )

        if key == ord("f"):
            print('ICP start (fine mode)')
            result_icp = refine_registration( CLOUD_ROT, pcd, np.identity(4), 3.0*voxel_size)
            print(result_icp)
            CLOUD_ROT.transform( result_icp )
            all_transformation = np.dot( result_icp, all_transformation )
            generateImage( mapping, im_color )


        """Step rotation"""
        if key == ord("w"):
            print('Rotation around roll axis')
            rotation = c3D.ComputeTransformationMatrixAroundCentroid( CLOUD_ROT, step, 0, 0 )
            CLOUD_ROT.transform( rotation )
            all_transformation = np.dot( rotation, all_transformation )
            generateImage( mapping, im_color )
            
        if key == ord("s"):
            print('Rotation around roll axis')
            rotation = c3D.ComputeTransformationMatrixAroundCentroid( CLOUD_ROT, -step, 0, 0 )
            CLOUD_ROT.transform( rotation )
            all_transformation = np.dot( rotation, all_transformation )
            generateImage( mapping, im_color )

        if key == ord("a"):
            print('Rotation around pitch axis')
            rotation = c3D.ComputeTransformationMatrixAroundCentroid( CLOUD_ROT, 0, step, 0 )
            CLOUD_ROT.transform( rotation )
            all_transformation = np.dot( rotation, all_transformation )
            generateImage( mapping, im_color )

        if key == ord("d"):
            print('Rotation around pitch axis')
            rotation = c3D.ComputeTransformationMatrixAroundCentroid( CLOUD_ROT, 0, -step, 0 )
            CLOUD_ROT.transform( rotation )
            all_transformation = np.dot( rotation, all_transformation )
            generateImage( mapping, im_color )

        if key == ord("q"):
            print('Rotation around yaw axis')
            rotation = c3D.ComputeTransformationMatrixAroundCentroid( CLOUD_ROT, 0, 0, step )
            CLOUD_ROT.transform( rotation )
            all_transformation = np.dot( rotation, all_transformation )
            generateImage( mapping, im_color )
        
        if key == ord("e"):
            print('Rotation around yaw axis')
            rotation = c3D.ComputeTransformationMatrixAroundCentroid( CLOUD_ROT, 0, 0, -step )
            CLOUD_ROT.transform( rotation )
            all_transformation = np.dot( rotation, all_transformation )
            generateImage( mapping, im_color )
            

    cv2.destroyAllWindows()
    CLOUD_ROT.paint_uniform_color([0.9, 0.1, 0.1])
    o3d.visualization.draw_geometries([CLOUD_ROT, cloud_in_ds])
    """ Save output files """
    # o3d.io.write_point_cloud( "cloud_rot_ds.ply", CLOUD_ROT )
    
    cloud_m.transform( all_transformation )
    # o3d.io.write_point_cloud( "cloud_rot.ply", cloud_m )
    img_mapped_original = mapping.Cloud2Image( cloud_m )
    # cv2.imwrite("img_mapped.png", img_mapped_original)

    print("\n\nFinal transformation is\n", all_transformation)
    print("You can transform the original model to the final pose by multiplying above matrix.")
    c3D.save_transformation( all_transformation, 'trans.json')
    
