from fileinput import filename
import cv2
import numpy as np
import os
from threading import Thread, Lock
import json
from pycocotools import mask as M
import matplotlib
from tkinter import Tk
from tkinter import filedialog
matplotlib.use('agg')


lock = Lock()

def read_image(path):
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    return bgrImage

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
        self.H = None
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
        self.angle = 15
        self.icx, self.icy = self.width/2, self.height/2
        self.data_type = None
        self.black = np.zeros((self.height//2, self.width//3, 3), dtype=np.uint8)



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
        cv2.createTrackbar('scene_id','GIST AILAB Data GT Visualizer', 0, 1000, self.on_scene_id)
        cv2.createTrackbar('image_id','GIST AILAB Data GT Visualizer', 0, 1000, self.on_image_id)
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
        elif event == cv2.EVENT_LBUTTONDOWN:
            # if self.icx == self.width/2 and self.icy == self.height/2 and self.scale_factor == 1:
            if self.H is not None:
                H_inv = cv2.invertAffineTransform(self.H[0:2])
                x, y = H_inv @ np.array([x, y, 1])
                x, y = int(x), int(y)
            # draw rectangular grid on rgb image centered at (x, y)
            grid_size = 9
            # draw horizontal lines
            min_x = x - grid_size//2
            max_x = x + grid_size//2
            min_y = y - grid_size//2
            max_y = y + grid_size//2
            for i in range(grid_size):
                if i % 2 == 0:
                    cv2.line(self.grid_img, (min_x, min_y+i), (max_x, min_y+i), (0, 255, 0), 1)
                    cv2.line(self.grid_img, (min_x+i, min_y), (min_x+i, max_y), (0, 255, 0), 1)
            # overlay grid image on self.frame
            self.frame = cv2.addWeighted(self.frame_original.copy(), 1.0, self.grid_img, 1.0, 0)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.frame = self.frame_original.copy()
            self.grid_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)


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
        self.H = H
        M = H[0:2]
        img = cv2.warpAffine(img, M, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
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
            self.amodal, self.vis, self.occ = self.visualize_masks()
            self.rgb = cv2.putText(self.rgb.copy(), "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.depth = cv2.putText(self.depth.copy(), "DEPTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.amodal = cv2.putText(self.amodal.copy(), "AMODAL MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.vis = cv2.putText(self.vis, "VISIBLE MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.occ = cv2.putText(self.occ, "INVISIBLE MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            black = cv2.putText(self.black.copy(), "Scene ID: {}".format(self.scene_id), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            black = cv2.putText(black, "Image ID: {}".format(self.image_id), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            rgbd = np.hstack((self.rgb, self.depth, black))
            masks = np.hstack((self.amodal, self.vis, self.occ))
            self.frame = np.vstack((rgbd, masks))
            self.frame_original = self.frame.copy()
            self.frame = cv2.addWeighted(self.frame, 1.0, self.grid_img, 1.0, 0)
            self.is_img_load = True

        if not self.is_updated and self.rgb_path is not None:
            self.frame = cv2.addWeighted(self.frame_original.copy(), 1.0, self.grid_img, 1.0, 0)
            self.frame = self.affine_transform(self.frame)
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
        rgb = cv2.resize(rgb, (self.width//3, self.height//2), interpolation=cv2.INTER_NEAREST)[:,:,:3]
        depth = cv2.resize(depth, (self.width//3, self.height//2), interpolation=cv2.INTER_NEAREST)[:,:,:3]
        rgb = np.uint8(rgb)
        depth = np.uint8(depth)
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
        obj_names = []
        for idx, anno in enumerate(self.annotations):
            amodal_mask = M.decode({'counts': anno["amodal_mask"], 'size': self.size})
            amodal_mask = amodal_mask.astype(np.uint8)
            amodal_mask = cv2.resize(amodal_mask, (self.width//3, self.height//2), interpolation=cv2.INTER_NEAREST)
            amodal_mask = amodal_mask.astype(bool)
            amodal_masks.append(amodal_mask)

            try:
                x, y = np.where(amodal_mask)
                x, y = x.min(), y.min()
            except:
                x, y = 0, 0
            amodal_toplefts.append((y, x))

            vis_mask = M.decode({'counts': anno["visible_mask"], 'size': self.size})
            vis_mask = vis_mask.astype(np.uint8)
            vis_mask = cv2.resize(vis_mask, (self.width//3, self.height//2), interpolation=cv2.INTER_NEAREST)
            vis_mask = vis_mask.astype(bool)
            vis_masks.append(vis_mask)

            try:
                x, y = np.where(vis_mask)
                x, y = x.min(), y.min()
            except:
                x, y = 0, 0
            vis_toplefts.append((y, x))

            occ_mask = M.decode({'counts': anno["invisible_mask"], 'size': self.size})
            occ_mask = occ_mask.astype(np.uint8)
            occ_mask = cv2.resize(occ_mask, (self.width//3, self.height//2), interpolation=cv2.INTER_NEAREST)
            occ_mask = occ_mask.astype(bool)
            occ_masks.append(occ_mask)
            x, y = np.where(occ_mask)
            try:
                x, y = x.min(), y.min()
            except:
                x, y = 0, 0
            occ_toplefts.append((y, x))
            if "data2" in self.data_type:
                obj_names.append("{}_{}".format(anno["object_id"], anno["instance_id"]))
            else:
                obj_names.append("{}".format(anno["object_id"]))

        # draw amodal and visible masks on rgb
        amodal = self.rgb.copy()
        vis = self.rgb.copy()
        occ = self.rgb.copy()
        cmap = matplotlib.cm.get_cmap('gist_rainbow')

        if "data2" in self.data_type:
            for i, (amodal_mask, vis_mask, amodal_topleft, vis_topleft, occ_mask, occ_top_left) in enumerate(zip(amodal_masks, vis_masks, amodal_toplefts, vis_toplefts, occ_masks, occ_toplefts)):
                amodal[amodal_mask] = np.array(cmap(i/len(amodal_masks))[:3]) * 255 * 0.6 + amodal[amodal_mask] * 0.4
                vis[vis_mask] = np.array(cmap(i/len(vis_masks))[:3]) * 255 * 0.6 + vis[vis_mask] * 0.4
                occ[occ_mask] = np.array(cmap(i/len(occ_masks))[:3]) * 255 * 0.6 + occ[occ_mask] * 0.4
                if amodal_topleft[0] > 0 and amodal_topleft[1] > 0:
                    amodal = cv2.putText(amodal, obj_names[i], amodal_topleft, cv2.FONT_HERSHEY_SIMPLEX, 0.7, np.array(cmap(i/len(vis_masks))[:3]) * 255, 2)
                if vis_topleft[0] != 0 and vis_topleft[1] != 0:
                    vis = cv2.putText(vis, obj_names[i], vis_topleft, cv2.FONT_HERSHEY_SIMPLEX, 0.7, np.array(cmap(i/len(vis_masks))[:3]) * 255, 2)
                if occ_top_left[0] != 0 and occ_top_left[1] != 0:
                    occ = cv2.putText(occ, obj_names[i], occ_top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, np.array(cmap(i/len(vis_masks))[:3]) * 255, 2)
        elif "data3-1" in self.data_type and len(amodal_masks) > 0:
            amodal_mask = amodal_masks[0]
            vis_mask = vis_masks[0]
            occ_mask = occ_masks[0]
            amodal_topleft = amodal_toplefts[0]
            vis_topleft = vis_toplefts[0]
            occ_top_left = occ_toplefts[0]
            amodal[amodal_mask] = np.array(cmap(0)[:3]) * 255 * 0.6 + amodal[amodal_mask] * 0.4
            vis[vis_mask] = np.array(cmap(0)[:3]) * 255 * 0.6 + vis[vis_mask] * 0.4
            occ[occ_mask] = np.array(cmap(0)[:3]) * 255 * 0.6 + occ[occ_mask] * 0.4
            amodal = np.uint8(amodal)[:, :, :3]
            vis = np.uint8(vis)[:, :, :3]
            occ = np.uint8(occ)[:, :, :3]
            if amodal_topleft[0] > 0 and amodal_topleft[1] > 0:
                amodal = cv2.putText(amodal, obj_names[0], amodal_topleft, cv2.FONT_HERSHEY_SIMPLEX, 0.7, np.array(cmap(0)[:3]) * 255, 2)
            if vis_topleft[0] != 0 and vis_topleft[1] != 0:
                vis = cv2.putText(vis, obj_names[0], vis_topleft, cv2.FONT_HERSHEY_SIMPLEX, 0.7, np.array(cmap(0)[:3]) * 255, 2)
            if occ_top_left[0] != 0 and occ_top_left[1] != 0:
                occ = cv2.putText(occ, obj_names[0], occ_top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, np.array(cmap(0)[:3]) * 255, 2)

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
            if os.path.basename(aihub_root) == "실제":
                self.data_type = "data2_real"
            elif os.path.basename(aihub_root) == "가상":
                self.data_type = "data2_syn"
            else:
                print("폴더 구조를 확인하세요.")
        elif sub_dir_1 in ["UR5", "Panda"]:
            if os.path.basename(aihub_root) == "실제":
                self.data_type = "data3-1_real"
            elif os.path.basename(aihub_root) == "가상":
                self.data_type = "data3-1_syn"
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
        elif self.data_type == "data3-1_real":
            self.max_scene_id = 1000
            self.min_scene_id = 1
            self.max_image_id = 999
            self.min_image_id = 1
        elif self.data_type == "data3-1_syn":
            self.max_scene_id = 100
            self.min_scene_id = 1
            self.max_image_id = 150
            self.min_image_id = 0
        
        self.init_cv2()
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