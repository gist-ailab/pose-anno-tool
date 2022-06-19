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

import os
import cv2
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import json
import threading
import time
import numpy as np


def tail(stream_file):
    """ Read a file like the Unix command `tail`. Code from https://stackoverflow.com/questions/44895527/reading-infinite-stream-tail """
    stream_file.seek(0, os.SEEK_END)  # Go to the end of file

    while True:
        if stream_file.closed:
            raise StopIteration

        line = stream_file.readline()

        yield line

CLOUD_NAME = "points"


class poseAnnotator():

    def __init__(self):

        self.is_done = False
        self.n_snapshots = 0
        self.cloud = None
        self.main_vis = None
        self.snapshot_pos = None

        self.annoRGBPath = None
        self.annoDepthPath = None
        self.annoSceneCameraPath = None
        self.annoObjectPath = None
        self.keyInput = None

        app = o3d.visualization.gui.Application.instance
        app.initialize()

        self.main_vis = o3d.visualization.O3DVisualizer(
            "Open3D - Multi-Window Demo")
        self.main_vis.set_on_close(self.on_main_window_closing)
        app.add_window(self.main_vis)



        threading.Thread(target=self.getData, daemon=True).start()
        threading.Thread(target=self.update_thread, daemon=True).start()

        app.run()

    def on_main_window_closing(self):
        self.is_done = True
        return True  # False would cancel the close


    def loadScene(self, rgbPath, depthPath, sceneCameraPath):
        color_raw = o3d.io.read_image(rgbPath)
        width, height = color_raw.get_max_bound()
        self.width, self.height = int(width), int(height)
        depth_raw = o3d.io.read_image(depthPath)
        with open(sceneCameraPath, "r") as f:
            scene_camera_info = json.load(f)
        K = scene_camera_info["3"]["cam_K"]
        self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.width, height=self.height, fx=K[0], fy=K[4], cx=K[2], cy=K[5])

        im_color = np.asarray(color_raw)
        self.im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
        self.im_depth = np.asarray(depth_raw)
        self.x, self.y = im_color.shape[1] // 2, im_color.shape[0] // 2

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, 1000.0, 2.0, convert_rgb_to_intensity=False)
        self.sceneCloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.camera_intrinsic)


    def update_thread(self):
        # This is NOT the UI thread, need to call post_to_main_thread() to update
        # the scene or any part of the UI.


        while self.annoRGBPath is None or self.annoDepthPath is None or self.annoSceneCameraPath is None:
            pass

        self.loadScene(self.annoRGBPath, self.annoDepthPath, self.annoSceneCameraPath)
        bounds = self.sceneCloud.get_axis_aligned_bounding_box()
        extent = bounds.get_extent()

        def add_first_cloud():
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            self.main_vis.add_geometry(CLOUD_NAME, self.sceneCloud, mat)
            self.main_vis.reset_camera_to_default()
            self.main_vis.setup_camera(60, bounds.get_center(),
                                       bounds.get_center() + [0, 0, -3],
                                       [0, -1, 0])

        o3d.visualization.gui.Application.instance.post_to_main_thread(
            self.main_vis, add_first_cloud)

        while not self.is_done:
            time.sleep(0.1)

            # Perturb the cloud with a random walk to simulate an actual read
            pts = np.asarray(self.sceneCloud.points)
            magnitude = 0.005 * extent
            displacement = magnitude * (np.random.random_sample(pts.shape) -
                                        0.5)
            new_pts = pts + displacement
            self.sceneCloud.points = o3d.utility.Vector3dVector(new_pts)

            def update_cloud():
                # Note: if the number of points is less than or equal to the
                #       number of points in the original object that was added,
                #       using self.scene.update_geometry() will be faster.
                #       Requires that the point cloud be a t.PointCloud.
                self.main_vis.remove_geometry(CLOUD_NAME)
                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = "defaultUnlit"
                self.main_vis.add_geometry(CLOUD_NAME, self.sceneCloud, mat)

            if self.is_done:  # might have changed while sleeping
                break
            o3d.visualization.gui.Application.instance.post_to_main_thread(
                self.main_vis, update_cloud)

    # def _add_geometry(self, name, geo, mat):
        # self.scene_widget.scene.add_geometry(name, geo, mat)

    def getData(self):

        while not self.is_done:
            with open("./comm.json", "r") as log_file:
                for line in tail(log_file):
                    try:
                        data = json.loads(line)
                    except ValueError:
                        # Bad json format, maybe corrupted...
                        continue  # Read next line

                    # Do what you want with data:
                    # db.execute("INSERT INTO ...", log_data["level"], ...)
                    print("recv:", data)
                    keys = data.keys()

                    if "keyInput" in keys:
                        self.keyInput = data["keyInput"]
                    if "annoRGBPath" in keys:
                        self.annoRGBPath = data["annoRGBPath"]
                    if "annoDepthPath" in keys:
                        self.annoDepthPath = data["annoDepthPath"]
                    if "annoSceneCameraPath" in keys:
                        self.annoSceneCameraPath = data["annoSceneCameraPath"]


                    if "annoObjectPath" in keys:
                        self.annoObjectPath = data["annoObjectPath"]
                        geo = o3d.io.read_point_cloud(self.annoObjectPath)
                        geo.scale(1000, geo.get_center())
                        mat = rendering.MaterialRecord()
                        mat.base_color = [0, 0, 0, 0.5]
                        mat.point_size = 5.0
                        # self._add_geometry("point cloud", geo, mat)

def run_test_o3d():

    o3d.visualization.webrtc_server.enable_webrtc()
    p = poseAnnotator()






if __name__ == "__main__":
    run_test_o3d()