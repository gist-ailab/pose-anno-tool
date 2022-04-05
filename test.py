import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import platform
import sys
import trimesh
import pyrender
import json
import cv2



def main():
    render = rendering.OffscreenRenderer(640, 480)


    scene_camera_path = "/home/seung/Workspace/custom/6DPoseAnnotator/data/hope/test/000001/scene_camera.json"
    path = "/home/seung/Workspace/custom/6DPoseAnnotator/data/hope/models/obj_000001.ply"
    # object_mesh = o3d.io.read_point_cloud(path, print_progress=True)
    # object_mesh = o3d.io.read_point_cloud(path)
    # print(np.min(np.asarray(object_mesh.colors)))

    mesh_tri = trimesh.load(path)
    mesh_tri = mesh_tri.apply_scale(0.001)
    mesh = pyrender.Mesh.from_trimesh(mesh_tri)
    scene = pyrender.Scene()
    pose = np.eye(4)
    pose[:3, 3] = [0, 0, -1]
    print(pose)
    scene.add(mesh, pose=pose)
    with open(scene_camera_path, "r") as f:
        scene_camera_info = json.load(f)
    K = scene_camera_info["0"]["cam_K"]
    camera = pyrender.IntrinsicsCamera(K[0], K[4], K[2], K[5])
    camera_pose = np.eye(4)
    s = np.sqrt(2)/2
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                                innerConeAngle=np.pi/16.0,
                                outerConeAngle=np.pi/6.0)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(1920, 1080)
    color, depth = r.render(scene)
    cv2.imshow("test", color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    # o3d.visualization.draw_geometries([object_mesh])
    exit()



    
    render.scene.add_model("cyl", mesh)


    render.setup_camera(60.0, [0, 0, 0], [0, 10, 0], [0, 0, 1])
    render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
                                     75000)
    render.scene.scene.enable_sun_light(True)
    render.scene.show_axes(True)

    img = render.render_to_image()
    print("Saving image at test.png")
    o3d.io.write_image("test.png", img, 9)

    render.setup_camera(60.0, [0, 0, 0], [-10, 0, 0], [0, 0, 1])
    img = render.render_to_image()
    print("Saving image at test2.png")
    o3d.io.write_image("test2.png", img, 9)


if __name__ == "__main__":
    main()
