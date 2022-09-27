from __future__ import annotations
import imgviz
import cv2
import numpy as np
import glob
import os
from threading import Thread, Lock
import json
from pycocotools import mask as M
import matplotlib
import networkx as nx
import string
import matplotlib.pyplot as plt
matplotlib.use('agg')


lock = Lock()


class GTVisualizer():

    def __init__(self):
       
        self.stopped = True
        self.width, self.height = 1920, 720
        self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.is_updated = True
        # !TODO: get this by user
        self.aihub_root = "/home/seung/OccludedObjectDataset/aihub/원천데이터/다수물체가림/실제"
        self.scene_id = 1
        self.get_sub_dirs_from_scene_id
        self.image_id = 1

        self.init_cv2()
        self.on_scene_id(self.scene_id)
        self.on_image_id(self.image_id)

    def get_sub_dirs_from_scene_id(self):
        
        for sub_dir_1 in os.listdir(self.aihub_root):
            for sub_dir_2 in os.listdir(os.path.join(self.aihub_root, sub_dir_1)):
                for scene_id in os.listdir(os.path.join(self.aihub_root, sub_dir_1, sub_dir_2)):
                    if int(scene_id) == self.scene_id:
                        self.sub_dir_1 = sub_dir_1
                        self.sub_dir_2 = sub_dir_2
                        return
        print("scene_id is not valid")
        exit()
        


    def init_cv2(self):
        cv2.namedWindow('GIST AILAB Data2 GT Visualizer')
        cv2.createTrackbar('scene_id','GIST AILAB Data2 GT Visualizer', 1, 1000, self.on_scene_id)
        cv2.createTrackbar('image_id','GIST AILAB Data2 GT Visualizer', 1, 52, self.on_image_id)
        cv2.setTrackbarPos('scene_id','GIST AILAB Data2 GT Visualizer', self.scene_id)
        cv2.setTrackbarPos('image_id','GIST AILAB Data2 GT Visualizer', self.image_id)

    def on_scene_id(self, val):
        self.scene_id = val
        self.get_sub_dirs_from_scene_id()
        self.scene_path = os.path.join(self.aihub_root, self.sub_dir_1, self.sub_dir_2, "{0:06d}".format(self.scene_id))
        self.is_updated = False

    def on_image_id(self, val):
        self.image_id = val
        self.rgb_path = os.path.join(self.scene_path, "rgb", "{0:06d}.png".format(self.image_id))
        self.depth_path = os.path.join(self.scene_path, "depth", "{0:06d}.png".format(self.image_id))
        self.gt_path = os.path.join(self.scene_path, "gt", "{0:06d}.json".format(self.image_id))
        self.is_updated = False

    def update_vis(self):
        if not self.is_updated:
            self.rgb, self.depth = self.load_rgbd()
            self.amodal, self.vis, self.occ = self.visualize_masks()

        self.rgb = cv2.putText(self.rgb, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.depth = cv2.putText(self.depth, "DEPTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.amodal = cv2.putText(self.amodal, "AMODAL MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.vis = cv2.putText(self.vis, "VISIBLE MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.occ = cv2.putText(self.occ, "INVISIBLE MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        black = np.zeros((self.height//2, self.width//3, 3), dtype=np.uint8)
        rgbd = np.hstack((self.rgb, self.depth, black))
        masks = np.hstack((self.amodal, self.vis, self.occ))
        self.frame = np.vstack((rgbd, masks))

        
    def load_rgbd(self):
        rgb = cv2.imread(self.rgb_path)
        depth = cv2.imread(self.depth_path, cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32)
        min_depth = np.unique(np.partition(depth.flatten(), 1))[1]
        depth[depth < min_depth] = min_depth
        depth = (depth - min_depth) / (depth.max() - min_depth)
        depth = depth * 255
        depth = depth.astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        rgb = cv2.resize(rgb, (self.width//3, self.height//2), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (self.width//3, self.height//2), interpolation=cv2.INTER_NEAREST)
        return rgb, depth

    def visualize_masks(self):
        
        self.gt = json.load(open(self.gt_path, "r"))
        self.annotations = self.gt["annotation"]
        amodal_masks = []
        vis_masks = []
        occ_masks = []
        amodal_toplefts = []
        vis_toplefts = []
        occ_toplefts = []
        for idx, anno in enumerate(self.annotations):
            amodal_mask = M.decode(anno["amodal_mask"])
            amodal_mask = amodal_mask.astype(np.uint8)
            amodal_mask = cv2.resize(amodal_mask, (self.width//3, self.height//2), interpolation=cv2.INTER_NEAREST)
            amodal_mask = amodal_mask.astype(bool)
            amodal_masks.append(amodal_mask)

            x, y = np.where(amodal_mask)
            x1, y1 = x.min(), y.min()
            x2, y2 = x.max(), y.max()
            x, y = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
            amodal_toplefts.append((y, x))

            vis_mask = M.decode(anno["visible_mask"])
            vis_mask = vis_mask.astype(np.uint8)
            vis_mask = cv2.resize(vis_mask, (self.width//3, self.height//2), interpolation=cv2.INTER_NEAREST)
            vis_mask = vis_mask.astype(bool)
            vis_masks.append(vis_mask)

            x, y = np.where(vis_mask)
            x1, y1 = x.min(), y.min()
            x2, y2 = x.max(), y.max()
            x, y = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
            vis_toplefts.append((y, x))

            occ_mask = M.decode(anno["invisible_mask"])
            occ_mask = occ_mask.astype(np.uint8)
            occ_mask = cv2.resize(occ_mask, (self.width//3, self.height//2), interpolation=cv2.INTER_NEAREST)
            occ_mask = occ_mask.astype(bool)
            occ_masks.append(occ_mask)

            x, y = np.where(occ_mask)
            try:
                x1, y1 = x.min(), y.min()
                x2, y2 = x.max(), y.max()
                x, y = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
            except:
                x, y = 0, 0
            occ_toplefts.append((y, x))



        # draw amodal and visible masks on rgb
        amodal = self.rgb.copy()
        vis = self.rgb.copy()
        occ = self.rgb.copy()
        cmap = matplotlib.cm.get_cmap('gist_rainbow')

        for i, (amodal_mask, vis_mask, amodal_topleft, vis_topleft, occ_mask, occ_top_left) in enumerate(zip(amodal_masks, vis_masks, amodal_toplefts, vis_toplefts, occ_masks, occ_toplefts)):
            amodal[amodal_mask] = np.array(cmap(i/len(amodal_masks))[:3]) * 255 * 0.6 + amodal[amodal_mask] * 0.4
            vis[vis_mask] = np.array(cmap(i/len(vis_masks))[:3]) * 255 * 0.6 + vis[vis_mask] * 0.4
            occ[occ_mask] = np.array(cmap(i/len(occ_masks))[:3]) * 255 * 0.6 + occ[occ_mask] * 0.4
            amodal = cv2.putText(amodal, string.ascii_uppercase[i], amodal_topleft, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            vis = cv2.putText(vis, string.ascii_uppercase[i], vis_topleft, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            if occ_top_left[0] != 0 and occ_top_left[1] != 0:
                occ = cv2.putText(occ, string.ascii_uppercase[i], occ_top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        amodal = cv2.resize(amodal, (self.width//3, self.height//2), interpolation=cv2.INTER_NEAREST)
        vis = cv2.resize(vis, (self.width//3, self.height//2), interpolation=cv2.INTER_NEAREST)
        occ = cv2.resize(occ, (self.width//3, self.height//2), interpolation=cv2.INTER_NEAREST)
        return amodal, vis, occ


    def start_frame(self):
        if not self.stopped:
            return
        self.stopped = False
        Thread(target=self.update_vis).start()
        return self

    def stop_frame(self):
        self.stopped = True

    def get_frame(self):
        self.update_vis()
        return np.uint8(self.frame)

if __name__ == "__main__":
    
    gt_visualizer = GTVisualizer()
    gt_visualizer.start_frame()
    while True:
        cv2.imshow("GIST AILAB Data2 GT Visualizer", gt_visualizer.get_frame())
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break

    gt_visualizer.stop_frame()
    cv2.destroyAllWindows()