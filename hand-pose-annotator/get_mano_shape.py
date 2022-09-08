from re import I
from turtle import color
import pyk4a
from pyk4a import PyK4A, Config
import cv2
import numpy as np
import open3d as o3d
import trimesh
import torch
from manopth.manolayer import ManoLayer
import image_geometry

# hyper parameters
roi = [0.3, 0.7, 0.2, 0.8] # x1, x2, y1, y2
wrist_pos = [1920, 1700]
mano_model_path = '/home/seung/Workspace/papers/2022/dyn2hand/frankmocap/mano'
side = 'left'

def convert_uvd_to_xyz(u, v, depth, cam_model, fs=5, rectify=False):
    """
    Convert 2D coordinates to 3D coordinate based on camera.
    Args: 
        u (int):   X coordinate of 2D pixel
        v (int):   Y coordinate of 2D pixel
        depth (np.float32): depth image array in meters (H, W, 1)
        cam_model (image_geometry.PinholeCameraModel): camera model initialized with camera 
    Returns:
        np.array : 3D coordinates array
    """

    d = depth[v][u]  
    # Fix depth value if d is 0
    if d == 0:
        depth_crop = depth[v-fs:v+fs, u-fs:u+fs]
        depth_crop = depth_crop[depth_crop != 0]
        d = np.median(depth_crop, axis=0)

    # Rectify point
    if rectify:
        u, v = cam_model.rectifyPoint((u, v))
    # Converting function (2D to 3D)
    ray = np.array(cam_model.projectPixelTo3dRay((u, v))) 
    ray = ray * d 
    ray[2] = d
    return ray


# Load camera with the default config
config = Config(
    color_resolution=pyk4a.ColorResolution.RES_2160P,
    depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
)

# init kinect azure
k4a = PyK4A(config = config)
k4a.start()
cam_K = k4a.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)
distortion = k4a.calibration.get_distortion_coefficients(pyk4a.calibration.CalibrationType.COLOR)
intrinsic = o3d.camera.PinholeCameraIntrinsic(3840, 2160, cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2])

# init mano
mano_layer = ManoLayer(mano_root=mano_model_path, use_pca=False, ncomps=45, side=side)



# Get the next capture (blocking function)
while True:
    capture = k4a.get_capture()
    color_img = capture.color[:, :, :3]
    height, width, _ = color_img.shape
    depth_img = capture.transformed_depth 

    # background_removal
    tmp = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(tmp, 150, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(color_img)
    # roi mask 
    roi_mask = np.zeros(depth_img.shape, dtype=np.uint8)
    roi_mask[int(roi[2]*height):int(roi[3]*height), int(roi[0]*width):int(roi[1]*width)] = 1
    mask = mask * roi_mask
    mask[mask==255] = 1


    # get largest contour regions
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        mask = mask * cv2.drawContours(np.zeros_like(mask), [c], -1, 1, -1)
    
    # visualization
    mask_3ch = np.expand_dims(mask, axis=2).repeat(3, axis=2)
    vis_img = color_img * mask_3ch
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(vis_img, (x,y), (x+w,y+h),(0,255,0),2)
    cv2.circle(vis_img, (wrist_pos[0], wrist_pos[1]), 50, (0, 0, 255), -1)
    cv2.rectangle(vis_img, (int(width*roi[0]), int(height*roi[2])), (int(width*roi[1]), int(height*roi[3])), (0, 255, 0), 2)



    vis_img = cv2.resize(vis_img, (1280, 720))
    cv2.imshow("test", vis_img)


    key = cv2.waitKey(1)
    if key == ord("v"):
        # rgb-d image to trimesh point cloud
        depth_img = np.float32(depth_img) / 1000
        depth_img = np.where(mask == 1, depth_img, 0)
        rgb_img_o3d = o3d.geometry.Image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        depth_img_o3d = o3d.geometry.Image(depth_img)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_img_o3d, depth_img_o3d, depth_scale=1, convert_rgb_to_intensity=False)
        pcd_captured = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        u, v = wrist_pos
        cx, cy = cam_K[0, 2], cam_K[1, 2]
        fx, fy = cam_K[0, 0], cam_K[1, 1]
        d = depth_img[v][u]
        x = (u - cx) * d / fx
        y = (v - cy) * d / fy
        z = d
        points = np.asarray(pcd_captured.points)
        # points -= np.array([x, y, z])
        points *= 1000
        pcd_captured.points = o3d.utility.Vector3dVector(points)
        # pcd = trimesh.points.PointCloud(pcd_o3d.points, pcd_o3d.colors)
        # pcd.show()

        wrist_pos_3d = torch.Tensor([[x, y, z]]) 


        betas = torch.rand(1, 10)*.1
        pose = torch.cat([torch.zeros(1, 45)*.1, torch.Tensor([[0, 0, 0]])], 1)
        hand_verts, hand_joints = mano_layer(
                                pose, 
                                betas, 
                                th_trans = wrist_pos_3d, 
                                root_palm=torch.Tensor([1]))
        global_orient = torch.Tensor([0, 0, 0]).view(1, 3)
        transl = torch.Tensor([0, 0, 0]).view(1, 3)


        pcd_mano = o3d.geometry.PointCloud()
        # pcd_mano.points = o3d.utility.Vector3dVector(h_meshes[0].vertices)
        # pcd_mano.colors = o3d.utility.Vector3dVector(h_meshes[0].vertex_colors)
        hand_verts = hand_verts[0].detach().cpu().numpy() 
        print(hand_verts.shape)
        pcd_mano.points = o3d.utility.Vector3dVector(hand_verts)
        pcd_mano.paint_uniform_color([1, 0, 0])

        o3d.visualization.draw_geometries([pcd_captured, pcd_mano])

    if key == ord('q'):
        cv2.destroyAllWindows()
        break



    # Display with pyplot
    # from matplotlib import pyplot as plt
    # plt.imshow(img_color[:, :, 2::-1]) # BGRA to RGB
    # plt.show()
