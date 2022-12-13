import cv2
import numpy as np
import os
from threading import Thread, Lock
from pycocotools import mask as M
import matplotlib
import networkx as nx
import string
import json
from tkinter import Tk
from tkinter import filedialog
import matplotlib.pyplot as plt
matplotlib.use('agg')



lock = Lock()

def read_image(path):
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    return bgrImage

def draw_graph_with_color(str_all, matrix, color_matrix, color='red', pos=None):
    edges = np.where(matrix == 1)

    from_idx = edges[0].tolist()
    to_idx = edges[1].tolist()

    from_node = [str_all[i] for i in from_idx]
    to_node = [str_all[i] for i in to_idx]

    G = nx.DiGraph()
    for i in range(matrix.shape[0]):
        G.add_node(str_all[i])

    pos = nx.circular_layout(G)
    G.add_edges_from(list(zip(from_node, to_node)))
    colors_tf = [color_matrix[pair[0], pair[1]] for pair in list(zip(from_idx, to_idx))]
    edge_color = [color if color_tf else 'black' for color_tf in colors_tf]
    cmap = matplotlib.cm.get_cmap('gist_rainbow')
    edge_color = [cmap(i/len(str_all))[::-1][1:] for i in from_idx]
    node_size=[3000]*len(G)

    nx.draw_networkx_nodes(G, pos, node_size=node_size)
    nx.draw_networkx_labels(G, pos, font_color='w', font_size=22)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=50, width=3, edge_color=edge_color, min_source_margin=25, min_target_margin=25)
    
    return pos

def draw_occ_graph(obj_names, occ_str_all, num_inst):
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
    draw_graph_with_color(obj_names, occ_matrix, is_overlap_matrix, color='green')

def draw_depth_graph(obj_names, depth_str_all, num_inst):
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
    draw_graph_with_color(obj_names, depth_matrix, is_overlap_matrix, color='green')


def translate(tx=0, ty=0):
    T = np.eye(3)
    T[0:2,2] = [tx, ty]
    return T

def scale(s=1, sx=1, sy=1):
    T = np.diag([s*sx, s*sy, 1])
    return T

def rotate(degrees):
    T = np.eye(3)
    # just involves some sin() and cos()
    T[0:2] = cv2.getRotationMatrix2D(center=(0,0), angle=-degrees, scale=1.0)
    return T

class GTVisualizer():

    def __init__(self):
       
        self.stopped = True
        self.width, self.height = 1920, 720
        self.M = None
        self.max_scene_id = 1000
        self.min_scene_id = 1
        self.max_image_id = 1000
        self.min_image_id = 1
        self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.grid_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.sub_dir_1 = ''
        self.sub_dir_2 = ''
        self.is_img_load = True
        self.is_updated = True
        self.rgb_path = None
        self.depth_path = None
        self.scale_factor = 1
        self.rbtn_down = False

        self.icx, self.icy = self.width/2, self.height/2
        self.data_type = None



    def get_sub_dirs_from_scene_id(self):
        
        for sub_dir_1 in os.listdir(self.aihub_root):
            for sub_dir_2 in os.listdir(os.path.join(self.aihub_root, sub_dir_1)):
                for scene_id in os.listdir(os.path.join(self.aihub_root, sub_dir_1, sub_dir_2)):
                    if int(scene_id) == self.scene_id:
                        self.sub_dir_1 = sub_dir_1
                        self.sub_dir_2 = sub_dir_2
                        print("scene_id에 해당하는 {}을 불러옵니다.".format(os.path.join(self.aihub_root, sub_dir_1, sub_dir_2)))
                        return
        print("존재하지 않는 scene_id가 입력되었습니다.")
        
    def get_filename_from_image_id(self):
        if not os.path.exists(os.path.join(self.scene_path, "rgb")):
            return None
        for file_name in os.listdir(os.path.join(self.scene_path, "rgb")):
            if int(file_name.split("_")[-1].split(".")[0]) == self.image_id:
                return file_name

    def init_cv2(self):
        cv2.namedWindow('GIST AILAB Data GT Visualizer')
        cv2.createTrackbar('scene_id','GIST AILAB Data GT Visualizer', self.min_scene_id, self.max_scene_id, self.on_scene_id)
        cv2.createTrackbar('image_id','GIST AILAB Data GT Visualizer', self.min_image_id, self.max_image_id, self.on_image_id)
        cv2.setTrackbarPos('scene_id','GIST AILAB Data GT Visualizer', self.scene_id)
        cv2.setTrackbarPos('image_id','GIST AILAB Data GT Visualizer', self.image_id)
        cv2.setMouseCallback('GIST AILAB Data GT Visualizer', self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.scale_factor += 0.1
            else:
                self.scale_factor -= 0.1
            self.is_updated = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.M is not None:
                H_inv = cv2.invertAffineTransform(self.M)
                x, y = H_inv @ np.array([x, y, 1])
                x, y = int(x), int(y)
            self.rbtn_down = True
            self.prev_x = x
            self.prev_y = y
        elif event == cv2.EVENT_RBUTTONUP and self.rbtn_down:
            if self.M is not None:
                H_inv = cv2.invertAffineTransform(self.M)
                x, y = H_inv @ np.array([x, y, 1])
                x, y = int(x), int(y)
            self.rbtn_down = False
            # move image
            self.icx += x - self.prev_x
            self.icy += y - self.prev_y
            self.is_updated = False
        
        

    def on_scene_id(self, val):
        if val < self.min_scene_id:
            val = self.min_scene_id
        elif val > self.max_scene_id:
            val = self.max_scene_id
        self.scene_id = val
        self.get_sub_dirs_from_scene_id()
        self.scene_path = os.path.join(self.aihub_root, self.sub_dir_1, self.sub_dir_2, "{0:06d}".format(self.scene_id))
        self.on_image_id(self.image_id)
        self.is_img_load = False
        cv2.setTrackbarPos('scene_id','GIST AILAB Data GT Visualizer', self.scene_id)

    def on_image_id(self, val):
        if val < self.min_image_id:
            val = self.min_image_id
        elif val > self.max_image_id:
            val = self.max_image_id
        self.image_id = val
        cv2.setTrackbarPos('image_id','GIST AILAB Data GT Visualizer', self.image_id)
        file_name = self.get_filename_from_image_id()
        if file_name is None:
            return
        self.rgb_path = os.path.join(self.scene_path, "rgb", file_name)
        self.depth_path = os.path.join(self.scene_path, "depth", file_name)
        if ".jpg" in self.depth_path:
            self.depth_path = self.depth_path.replace(".jpg", ".png")
        self.gt_path = os.path.join(self.scene_path, "gt", file_name.split(".")[0] + ".json")
        self.is_img_load = False
        cv2.setTrackbarPos('image_id','GIST AILAB Data GT Visualizer', self.image_id)

    def on_key(self, key):

        if key in [ord('q'), ord('w'), ord('e'), ord('a'), ord('s'), ord('d'), ord('z')]:
            self.apply_affine_transform(key)
        elif key == ord('o'):
            self.open_file()
        elif key == ord('r'):
            self.scene_id -= 1
            self.on_scene_id(self.scene_id)
            self.is_img_load = False
        elif key == ord('t'):
            self.scene_id += 1
            self.on_scene_id(self.scene_id)
            self.is_img_load = False
        elif key == ord('f'):
            self.image_id -= 1
            self.on_image_id(self.image_id)
            self.is_img_load = False
        elif key == ord('g'):
            self.image_id += 1
            self.on_image_id(self.image_id)
            self.is_img_load = False
        

    def affine_transform(self, img):
        ow, oh = self.width, self.height
        (ocx, ocy) = ((ow-1)/2, (oh-1)/2) 
        H = translate(+ocx, +ocy) @ scale(self.scale_factor) @ translate(-self.icx, -self.icy)
        self.M = H[0:2]
        img = cv2.warpAffine(img, self.M, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return img


    def apply_affine_transform(self, key):
        if key == ord('q'):
            self.scale_factor += 0.1
        elif key == ord('e'):
            self.scale_factor -= 0.1
        elif key == ord('w'):
            self.icy -= 10
        elif key == ord('s'):
            self.icy += 10
        elif key == ord('a'):
            self.icx -= 10
        elif key == ord('d'):
            self.icx += 10
        elif key == ord('z'):
            self.icx, self.icy = self.width/2, self.height/2
            self.scale_factor = 1
        if self.icx < 0:
            self.icx = 0
        if self.icy < 0:
            self.icy = 0
        if self.icx > self.width:
            self.icx = self.width
        if self.icy > self.height:
            self.icy = self.height
        self.is_updated = False


    def update_vis(self):
        if not self.is_img_load and self.rgb_path is not None:
            self.rgb, self.depth = self.load_rgbd()
            self.amodal = self.visualize_masks()
            self.occ_graph, self.depth_graph = self.visualize_graphs()
            self.rgb = cv2.putText(self.rgb.copy(), "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.depth = cv2.putText(self.depth.copy(), "DEPTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.amodal = cv2.putText(self.amodal.copy(), "AMODAL MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.occ_graph = cv2.putText(self.occ_graph, "OCC GRAPH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.depth_graph = cv2.putText(self.depth_graph, "DEPTH GRAPH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            rgbd = np.hstack((self.rgb, self.depth, self.amodal))
            graphs = np.hstack((self.occ_graph, self.depth_graph))
            self.frame = np.vstack((rgbd, graphs))
            self.frame_original = self.frame.copy()
            self.is_img_load = True

        if not self.is_updated and self.rgb_path is not None:
            self.frame = self.affine_transform(self.frame_original.copy())
            self.is_updated = True

        
        
    def load_rgbd(self):

        rgb = read_image(self.rgb_path)
        self.size = list(rgb.shape[:2] )
        depth = read_image(self.depth_path)
        depth = depth.astype(np.float32)
        min_depth = np.unique(np.partition(depth.flatten(), 1))[1]
        depth[depth < min_depth] = min_depth
        depth = (depth - min_depth) / (depth.max() - min_depth)
        depth = depth * 255
        depth = depth.astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        rgb = cv2.resize(rgb, (self.width//3, self.height//5*2), interpolation=cv2.INTER_NEAREST)[:,:,:3]
        depth = cv2.resize(depth, (self.width//3, self.height//5*2), interpolation=cv2.INTER_NEAREST)[:,:,:3]
        rgb = np.uint8(rgb)
        depth = np.uint8(depth)
        return rgb, depth


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
        draw_occ_graph(self.obj_names, occ_str_all, len(self.annotations))
        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        occlusion_graph = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        occlusion_graph = cv2.resize(occlusion_graph, (self.width//2, self.height//5*3), interpolation=cv2.INTER_NEAREST)
        plt.cla()
        plt.clf()

        fig = plt.figure(figsize=(15, 10))
        plt.axis('off')
        draw_depth_graph(self.obj_names, depth_str_all, len(self.annotations))
        fig.tight_layout()
        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        depth_graph = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        depth_graph = cv2.resize(depth_graph, (self.width//2, self.height//5*3), interpolation=cv2.INTER_NEAREST)
        plt.cla()
        plt.clf()
        lock.release()

        return occlusion_graph, depth_graph



    def visualize_masks(self):
        
        self.gt = json.load(open(self.gt_path, "r"))
        self.annotations = self.gt["annotation"]
        amodal_masks = []
        amodal_toplefts = []
        obj_names = []
        for idx, anno in enumerate(self.annotations):
            amodal_mask = M.decode({'counts': anno["amodal_mask"], 'size': self.size})
            amodal_mask = amodal_mask.astype(np.uint8)
            amodal_mask = cv2.resize(amodal_mask, (self.width//3, self.height//5*2), interpolation=cv2.INTER_NEAREST)
            amodal_mask = amodal_mask.astype(bool)
            amodal_masks.append(amodal_mask)
            try:
                x, y = np.where(amodal_mask)
                x, y = x.min(), y.min()
            except:
                x, y = 0, 0
            amodal_toplefts.append((y, x))
            obj_names.append("{}_{}".format(anno["object_id"], anno["instance_id"]))
        print(len(amodal_masks), len(amodal_toplefts), len(obj_names))

        self.obj_names = obj_names
        # draw amodal and visible masks on rgb
        amodal = self.rgb.copy()
        cmap = matplotlib.cm.get_cmap('gist_rainbow')

        for i, (amodal_mask, amodal_topleft) in enumerate(zip(amodal_masks, amodal_toplefts)):
            amodal[amodal_mask] = np.array(cmap(i/len(amodal_masks))[:3]) * 255 * 0.6 + amodal[amodal_mask] * 0.4
            if amodal_topleft[0] > 0 and amodal_topleft[1] > 0:
                amodal = cv2.putText(amodal, obj_names[i], amodal_topleft, cv2.FONT_HERSHEY_SIMPLEX, 0.7, np.array(cmap(i/len(amodal_masks))[:3]) * 255, 2)
        amodal = cv2.resize(amodal, (self.width//3, self.height//5*2), interpolation=cv2.INTER_NEAREST)
        return amodal


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

    def open_file(self):
        Tk().withdraw()
        file_path = filedialog.askopenfilename(title="Select A File", filetypes=(("image files", "*.png *.jpg"),("all files", "*.*")))
        image_id = os.path.basename(file_path).split(".")[0].split("_")[-1]
        sub_path, _ = os.path.split(file_path)
        sub_path, _ = os.path.split(sub_path)
        sub_path, scene_id = os.path.split(sub_path)
        sub_path, sub_dir_2 = os.path.split(sub_path)
        aihub_root, sub_dir_1 = os.path.split(sub_path)
        if sub_dir_1 in ["YCB", "HOPE", "APC", "GraspNet1Billion", "DexNet", "가정", "산업", "물류", "혼합"]:
            if os.path.basename(aihub_root) == "02_실제":
                self.data_type = "data2_real"
            elif os.path.basename(aihub_root) == "01_가상":
                self.data_type = "data2_syn"
            else:
                print("폴더 구조를 확인하세요.")
        elif sub_dir_1 in ["01_UR5", "02_Panda"]:
            if os.path.basename(aihub_root) == "02_실제":
                self.data_type = "data3_real"
            elif os.path.basename(aihub_root) == "01_가상":
                self.data_type = "data3_syn"
            else:
                print("폴더 구조를 확인하세요.")  
        else:
            print("잘못된 경로가 입력되었습니다: {}".format(file_path))
            return
        
        self.sub_dir_1 = sub_dir_1
        self.sub_dir_2 = sub_dir_2
        self.aihub_root = aihub_root
        self.scene_id = int(scene_id)
        self.image_id = int(image_id)
        if self.data_type == "data2_real":
            self.max_scene_id = 1000
            self.min_scene_id = 1
            self.max_image_id = 52
            self.min_image_id = 1
        elif self.data_type == "data2_syn":
            self.max_scene_id = 100
            self.min_scene_id = 0
            self.max_image_id = 999
            self.min_image_id = 0
        elif self.data_type == "data3_real":
            self.max_scene_id = 1000
            self.min_scene_id = 1
            self.max_image_id = 999
            self.min_image_id = 1
        elif self.data_type == "data3_syn":
            self.max_scene_id = 184
            self.min_scene_id = 1
            self.max_image_id = 250
            self.min_image_id = 1
        self.init_cv2()
        cv2.setTrackbarMax('scene_id', 'GIST AILAB Data GT Visualizer', self.max_scene_id)
        cv2.setTrackbarMin('scene_id', 'GIST AILAB Data GT Visualizer', self.min_scene_id)
        cv2.setTrackbarMax('image_id', 'GIST AILAB Data GT Visualizer', self.max_image_id)
        cv2.setTrackbarMin('image_id', 'GIST AILAB Data GT Visualizer', self.min_image_id)

        self.on_scene_id(self.scene_id)
        self.on_image_id(self.image_id)



if __name__ == "__main__":
    
    gt_visualizer = GTVisualizer()
    gt_visualizer.start_frame()
    gt_visualizer.open_file()
    while True:
        cv2.imshow("GIST AILAB Data GT Visualizer", gt_visualizer.get_frame())
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        else:
            gt_visualizer.on_key(k)
            




    gt_visualizer.stop_frame()
    cv2.destroyAllWindows()