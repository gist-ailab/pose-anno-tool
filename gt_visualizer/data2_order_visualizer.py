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

def draw_graph_with_color(matrix, color_matrix, color='red', pos=None):
    edges = np.where(matrix == 1)

    from_idx = edges[0].tolist()
    to_idx = edges[1].tolist()

    from_node = [string.ascii_uppercase[i] for i in from_idx]
    to_node = [string.ascii_uppercase[i] for i in to_idx]

    G = nx.DiGraph()
    for i in range(matrix.shape[0]):
        G.add_node(string.ascii_uppercase[i])

    pos = nx.circular_layout(G)
    G.add_edges_from(list(zip(from_node, to_node)))
    colors_tf = [color_matrix[pair[0], pair[1]] for pair in list(zip(from_idx, to_idx))]
    edge_color = [color if color_tf else 'black' for color_tf in colors_tf]
    node_size=[600]*len(G)

    nx.draw_networkx_nodes(G, pos, node_size=node_size)
    nx.draw_networkx_labels(G, pos, font_color='w', font_size=15)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=40, width=2, edge_color=edge_color)
    
    return pos

def draw_occ_graph(occ_str_all, num_inst):
    occ_matrix = np.zeros((num_inst, num_inst)).astype(np.uint8)
    is_overlap_matrix = np.zeros((num_inst, num_inst)).astype(bool)
    for occ_str in occ_str_all:
        idx1, idx2 = occ_str['order'].split(' & ')[0].split('<')
        idx1, idx2 = int(idx1), int(idx2)
        if '&' in occ_str['order']: #bidirection
            occ_matrix[idx1, idx2] = 1
            occ_matrix[idx2, idx1] = 1
        else:
            occ_matrix[idx1, idx2] = 1
    draw_graph_with_color(occ_matrix, is_overlap_matrix, color='green')

def draw_depth_graph(depth_str_all, num_inst):
    depth_matrix = np.zeros((num_inst, num_inst)).astype(np.uint8)
    is_overlap_matrix = np.zeros((num_inst, num_inst)).astype(bool)
    for depth_str in depth_str_all:
        if '=' in depth_str['order']:
            eq1_idx, eq2_idx = list(map(int,depth_str['order'].split('=')))

            depth_matrix[eq1_idx, eq2_idx] = 1
            depth_matrix[eq2_idx, eq1_idx] = 1
            is_overlap_matrix[eq1_idx, eq2_idx] = 1 if depth_str['overlap'] == True else 0
            is_overlap_matrix[eq2_idx, eq1_idx] = 1 if depth_str['overlap'] == True else 0
            
        elif '<' in depth_str['order']:
            near_idx, far_idx = list(map(int,depth_str['order'].split('<')))
            depth_matrix[near_idx, far_idx] = 1
            is_overlap_matrix[near_idx, far_idx] = 1 if depth_str['overlap'] == True else 0
    draw_graph_with_color(depth_matrix, is_overlap_matrix, color='green')


class GTVisualizer():

    def __init__(self):
       
        self.stopped = True
        self.width, self.height = 1920, 1000
        self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.is_updated = True
        # !TODO: get this by user
        self.aihub_root = "/home/seung/OccludedObjectDataset/aihub/원천데이터/다수물체가림/실제"
        self.sub_dir_1 = "혼합"
        self.sub_dir_2 = "PublicObject_혼합"
        self.scene_id = 1
        self.image_id = 1

        self.init_cv2()
        self.on_scene_id(self.scene_id)
        self.on_image_id(self.image_id)

    

    def init_cv2(self):
        cv2.namedWindow('GIST AILAB Data2 Order Visualizer')
        cv2.createTrackbar('scene_id','GIST AILAB Data2 Order Visualizer', 1, 1000, self.on_scene_id)
        cv2.createTrackbar('image_id','GIST AILAB Data2 Order Visualizer', 1, 52, self.on_image_id)
        cv2.setTrackbarPos('scene_id','GIST AILAB Data2 Order Visualizer', self.scene_id)
        cv2.setTrackbarPos('image_id','GIST AILAB Data2 Order Visualizer', self.image_id)

    def on_scene_id(self, val):
        self.scene_id = val
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
            self.vis = self.visualize_masks()
            self.occ_graph, self.depth_graph = self.visualize_graphs()

        self.rgb = cv2.putText(self.rgb, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.depth = cv2.putText(self.depth, "DEPTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.vis = cv2.putText(self.vis, "VISIBLE MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.occ_graph = cv2.putText(self.occ_graph, "OCC GRAPH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.depth_graph = cv2.putText(self.depth_graph, "DEPTH GRAPH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        rgbd = np.hstack((self.rgb, self.depth, self.vis))
        graphs = np.hstack((self.occ_graph, self.depth_graph))
        self.frame = np.vstack((rgbd, graphs))

        
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
        rgb = cv2.resize(rgb, (self.width//3, self.height//3), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (self.width//3, self.height//3), interpolation=cv2.INTER_NEAREST)
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
          
            vis_mask = M.decode(anno["visible_mask"])
            vis_mask = vis_mask.astype(np.uint8)
            vis_mask = cv2.resize(vis_mask, (self.width//3, self.height//3), interpolation=cv2.INTER_NEAREST)
            vis_mask = vis_mask.astype(bool)
            vis_masks.append(vis_mask)

            x, y = np.where(vis_mask)
            x1, y1 = x.min(), y.min()
            x2, y2 = x.max(), y.max()
            x, y = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
            vis_toplefts.append((y, x))


        # draw amodal and visible masks on rgb
        vis = self.rgb.copy()
        cmap = matplotlib.cm.get_cmap('gist_rainbow')

        for i, (vis_mask, vis_topleft) in enumerate(zip(vis_masks, vis_toplefts)):
            vis[vis_mask] = np.array(cmap(i/len(vis_masks))[:3]) * 255 * 0.6 + vis[vis_mask] * 0.4
            vis = cv2.putText(vis, string.ascii_uppercase[i], vis_topleft, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        vis = cv2.resize(vis, (self.width//3, self.height//3), interpolation=cv2.INTER_NEAREST)
        return vis

    def visualize_graphs(self):

        occ_str_all = []
        depth_str_all = []
        for anno in self.annotations:
            occ_str_all += anno["occlusion_order"]
            depth_str_all += anno["depth_order"]
        occ_str_all = [dict(s) for s in set(frozenset(d.items()) for d in occ_str_all)]
        depth_str_all = [dict(s) for s in set(frozenset(d.items()) for d in depth_str_all)]
        lock.acquire()
        fig = plt.figure(figsize=(15, 10))
        plt.axis('off')
        fig.tight_layout()
        draw_occ_graph(occ_str_all, len(self.annotations))
        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        occlusion_graph = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        occlusion_graph = cv2.resize(occlusion_graph, (self.width//2, self.height//3*2), interpolation=cv2.INTER_NEAREST)
        plt.cla()
        plt.clf()

        fig = plt.figure(figsize=(15, 10))
        plt.axis('off')
        draw_depth_graph(depth_str_all, len(self.annotations))
        fig.tight_layout()
        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        depth_graph = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        depth_graph = cv2.resize(depth_graph, (self.width//2, self.height//3*2), interpolation=cv2.INTER_NEAREST)
        plt.cla()
        plt.clf()
        lock.release()

        return occlusion_graph, depth_graph


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
        cv2.imshow("GIST AILAB Data2 Order Visualizer", gt_visualizer.get_frame())
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break

    gt_visualizer.stop_frame()
    cv2.destroyAllWindows()