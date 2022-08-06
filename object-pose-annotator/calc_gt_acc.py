import cv2
import json
import numpy as np
import open3d as o3d
import os

if __name__ == "__main__":


    scene_id = 2 
    img_h, img_w = 1080, 1920

    ##############################
    # load model and annotations #
    ##############################
    # path
    root_object = 'PATH/TO/models'
    root_data = 'PATH/TO/raw/000001'
    path_rgb = os.path.join(root_data, 'rgb', '{:06d}.png')
    path_depth = os.path.join(root_data, 'depth', '{:06d}.png')
    path_anno_cam = os.path.join(root_data, 'scene_camera.json')
    path_anno_obj = os.path.join(root_data, 'scene_gt.json')
    
    # load camera pose annotation
    with open(path_anno_cam) as gt_file:
        anno_cam = json.load(gt_file)
    # load object pose annotation
    with open(path_anno_obj) as gt_file:
        anno_obj = json.load(gt_file)
    anno_obj = anno_obj[str(scene_id)] # type: list

    # load object pointcloud and transform as annotation
    obj_geometries = {}
    for i, obj in enumerate(anno_obj):
        # get transform
        translation = np.array(np.array(obj['cam_t_m2c']), dtype=np.float64) / 1000  # convert to meter
        orientation = np.array(np.array(obj['cam_R_m2c']), dtype=np.float64)
        transform = np.concatenate((orientation.reshape((3, 3)), translation.reshape(3, 1)), axis=1)
        transform_cam_to_obj = np.concatenate(
            (transform, np.array([0, 0, 0, 1]).reshape(1, 4)))  # homogeneous transform
        # load pointcloud (.ply)
        obj_geometry = o3d.io.read_point_cloud(
            os.path.join(root_object, 'obj_' + f"{int(obj['obj_id']):06}" + '.ply'))
        obj_geometry.points = o3d.utility.Vector3dVector(
            np.array(obj_geometry.points) / 1000)  # convert mm to meter
        # move object
        obj_geometry.translate(transform_cam_to_obj[0:3, 3])
        center = obj_geometry.get_center()
        obj_geometry.rotate(transform_cam_to_obj[0:3, 0:3], center=center)
        # save in dictionary
        obj_geometries[i] = obj_geometry


    ###############################
    # generate offscreen renderer #
    ###############################
    # generate offscreen renderer
    render = o3d.visualization.rendering.OffscreenRenderer(
                                        width=img_w, height=img_h)
    # black background color
    render.scene.set_background([0, 0, 0, 1])
    render.scene.set_lighting(render.scene.LightingProfile.SOFT_SHADOWS, [0,0,0])
    # generate object material
    obj_mtl = o3d.visualization.rendering.MaterialRecord()
    obj_mtl.base_color = [1.0, 1.0, 1.0, 1.0]
    obj_mtl.shader = "defaultUnlit"
    # set camera intrinsic
    cam_K = anno_cam[str(scene_id)]["cam_K"]
    intrinsic = np.array(cam_K).reshape((3, 3))
    extrinsic = np.array([[1, 0, 0, 0],
                          [0, 1, 0 ,0],
                          [0, 0, 1, -0.13128653],
                          [0, 0, 0, 1]])
    render.setup_camera(intrinsic, extrinsic, img_w, img_h)
    # set camera pose
    center = [0, 0, 1]  # look_at target
    eye = [0, 0, 0]  # camera position
    up = [0, -1, 0]  # camera orientation
    render.scene.camera.look_at(center, eye, up)

    ###################
    # rendering depth #
    ###################
    # add geometry for rendering depth
    for i, obj_geometry in obj_geometries.items():
        render.scene.add_geometry("obj_{}".format(i), obj_geometry, obj_mtl,                              
                                  add_downsampled_copy_for_fast_rendering=True)
    # render DEPTH (meter)
    depth_rendered = render.render_to_depth_image(z_in_view_space=True)
    depth_rendered = np.array(depth_rendered)
    depth_rendered[np.isposinf(depth_rendered)] = 0
    depth_rendered /= anno_cam[str(scene_id)]["depth_scale"]
    
    render.scene.clear_geometry()

    ##########################
    # rendering object masks #
    ##########################
    obj_masks = {}
    for i in obj_geometries:
        # add geometry and set color (target object as white / others as black)
        for j, obj_geometry in obj_geometries.items():
            color = [1,0,0] if i==j else [0,0,0]
            obj_geometry.paint_uniform_color(color)
            render.scene.add_geometry("mask{}_obj_{}".format(i, j), obj_geometry, obj_mtl,                              
                                    add_downsampled_copy_for_fast_rendering=True)
        # render mask as RGB
        mask_obj = render.render_to_image()
        # mask_obj = cv2.cvtColor(np.array(mask_obj), cv2.COLOR_RGBA2BGRA)
        mask_obj = np.array(mask_obj)
        # save in dictionary
        obj_masks[i] = mask_obj.copy()
        # clear geometry
        render.scene.clear_geometry()

        # visualize for debugging
        cv2.imwrite("tmp_mask_obj_{}.png".format(i), mask_obj)


    #######################
    # load captured depth #
    #######################
    depth_captured = path_depth.format(scene_id)
    # depth_captured = cv2.imread(depth_captured, -1).astype(np.float32)
    # depth_captured /= 1000 # convert mm -> meter
    depth_captured = cv2.imread(depth_captured, -1) / 1000
    print("... DEPTH cap: shape: {} | min: {:.3f} | max: {:.3f} | mean: {:.3f} | std: {:.3f}".format(
           depth_captured.shape, depth_captured.min(), depth_captured.max(), 
           depth_captured.mean(), depth_captured.std()))
    print("... DEPTH ren: shape: {} | min: {:.3f} | max: {:.3f} | mean: {:.3f} | std: {:.3f}".format(
           depth_rendered.shape, depth_rendered.min(), depth_rendered.max(), 
           depth_rendered.mean(), depth_rendered.std()))

    print("---" * 20)
    rgb_captured = path_rgb.format(scene_id)
    rgb_captured = cv2.imread(rgb_captured)
    ########################################
    # calculate depth difference with mask #
    # depth_diff = depth_cap - depth_ren   #
    ########################################
    for obj_id, obj_mask in obj_masks.items():
        cnd_r = obj_mask[:, :, 0] != 0
        cnd_g = obj_mask[:, :, 1] == 0
        cnd_b = obj_mask[:, :, 2] == 0
        cnd_obj = np.bitwise_and(np.bitwise_and(cnd_r, cnd_g), cnd_b)

        cnd_bg = np.zeros((img_h+2, img_w+2), dtype=np.uint8)
        newVal, loDiff, upDiff = 1, 1, 0
        cv2.floodFill(cnd_obj.copy().astype(np.uint8), cnd_bg, 
                                   (0,0), newVal, loDiff, upDiff)

        cnd_bg = cnd_bg[1:img_h+1, 1:img_w+1].astype(bool)
        cnd_obj = 1 - cnd_bg.copy() 
        cnd_obj = cnd_obj.astype(bool)

        # visualize for debugging #
        print("... [cnd {}] MASK: {} | BG: {}".format(
              obj_id, cnd_obj.sum(), cnd_bg.sum()))
        cv2.imwrite("tmp_cnd_obj_{}.png".format(obj_id), cnd_obj*255)
        cv2.imwrite("tmp_cnd_bg_{}.png".format(obj_id), cnd_bg*255)
        rgb_captured_obj = rgb_captured.copy()
        rgb_captured_obj[cnd_bg] = 0
        cv2.imwrite("tmp_rgb_crop_obj_{}.png".format(obj_id), rgb_captured_obj)

        # get only object depth of captured depth
        depth_captured_obj = depth_captured.copy()
        depth_captured_obj[cnd_bg] = 0

        # get only object depth of rendered depth
        depth_rendered_obj = depth_rendered.copy()
        depth_rendered_obj[cnd_bg] = 0

        depth_diff = depth_captured_obj - depth_rendered_obj
        depth_diff_mean = depth_diff.mean()
        depth_diff_std = depth_diff.std()

        print("[object {}]".format(obj_id))
        print("...  (M) mean: {:.5f} / std: {:.5f}".format(depth_diff_mean, depth_diff_std))
        print("... (mm) mean: {:.3f} / std: {:.3f}".format(depth_diff_mean*1000, depth_diff_std*1000))