from fileinput import filename
import cv2
import numpy as np
import os
from threading import Thread, Lock
import json
from pycocotools import mask as M
import matplotlib
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
        self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.sub_dir_1 = ''
        self.sub_dir_2 = ''
        self.is_img_load = True
        self.is_updated = True
        self.rgb_path = None
        self.depth_path = None
        self.scale_factor = 1
        self.angle = 15
        self.icx, self.icy = self.width/2, self.height/2
        

        # aihub_root = input("실제 / 가상 폴더의 경로를 입력해주세요: \n")
        aihub_root = "/OccludedObjectDataset/aihub/원천데이터/다수물체가림/실제"
        self.aihub_root = aihub_root
        self.scene_id = 0
        self.get_sub_dirs_from_scene_id()
        self.image_id = 0

        self.init_cv2()
        self.on_scene_id(self.scene_id)
        self.on_image_id(self.image_id)
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
        cv2.namedWindow('GIST AILAB Data2 GT Visualizer')
        cv2.createTrackbar('scene_id','GIST AILAB Data2 GT Visualizer', 1, 1000, self.on_scene_id)
        cv2.createTrackbar('image_id','GIST AILAB Data2 GT Visualizer', 1, 52, self.on_image_id)
        cv2.setTrackbarPos('scene_id','GIST AILAB Data2 GT Visualizer', self.scene_id)
        cv2.setTrackbarPos('image_id','GIST AILAB Data2 GT Visualizer', self.image_id)

    def on_scene_id(self, val):
        self.scene_id = val
        self.get_sub_dirs_from_scene_id()
        self.scene_path = os.path.join(self.aihub_root, self.sub_dir_1, self.sub_dir_2, "{0:06d}".format(self.scene_id))
        self.on_image_id(self.image_id)
        self.is_img_load = False
        cv2.setTrackbarPos('scene_id','GIST AILAB Data2 GT Visualizer', self.scene_id)

    def on_image_id(self, val):
        self.image_id = val
        cv2.setTrackbarPos('image_id','GIST AILAB Data2 GT Visualizer', self.image_id)
        file_name = self.get_filename_from_image_id()
        if file_name is None:
            return
        self.rgb_path = os.path.join(self.scene_path, "rgb", file_name)
        self.depth_path = os.path.join(self.scene_path, "depth", file_name)
        if ".jpg" in self.depth_path:
            self.depth_path = self.depth_path.replace(".jpg", ".png")
        self.gt_path = os.path.join(self.scene_path, "gt", file_name.split(".")[0] + ".json")
        self.is_img_load = False
        cv2.setTrackbarPos('image_id','GIST AILAB Data2 GT Visualizer', self.image_id)


    def affine_transform(self, img):
        ow, oh = self.width, self.height
        (ocx, ocy) = ((ow-1)/2, (oh-1)/2) 
        H = translate(+ocx, +ocy) @ scale(self.scale_factor) @ translate(-self.icx, -self.icy)
        M = H[0:2]
        img = cv2.warpAffine(img, dsize=(ow,oh), M=M, flags=cv2.INTER_NEAREST)
        self.is_zoomed = True
        return img

    def update_vis(self):
        if not self.is_img_load and self.rgb_path is not None:
            self.rgb, self.depth = self.load_rgbd()
            self.amodal, self.vis, self.occ = self.visualize_masks()
            self.rgb = cv2.putText(self.rgb, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.depth = cv2.putText(self.depth, "DEPTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.amodal = cv2.putText(self.amodal, "AMODAL MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.vis = cv2.putText(self.vis, "VISIBLE MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.occ = cv2.putText(self.occ, "INVISIBLE MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            rgbd = np.hstack((self.rgb, self.depth, self.black))
            masks = np.hstack((self.amodal, self.vis, self.occ))
            self.frame = np.vstack((rgbd, masks))
            self.frame_original = self.frame.copy()
            self.is_img_load = True

        if not self.is_updated and self.rgb_path is not None:
            self.frame = self.affine_transform(self.frame_original)
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
        obj_names = []
        for idx, anno in enumerate(self.annotations):
            amodal_mask = M.decode({'counts': anno["amodal_mask"], 'size': self.size})
            amodal_mask = amodal_mask.astype(np.uint8)
            amodal_mask = cv2.resize(amodal_mask, (self.width//3, self.height//2), interpolation=cv2.INTER_NEAREST)
            amodal_mask = amodal_mask.astype(bool)
            amodal_masks.append(amodal_mask)

            x, y = np.where(amodal_mask)
            x, y = x.min(), y.min()
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
            obj_names.append("{}_{}".format(anno["object_id"], anno["instance_id"]))

        # draw amodal and visible masks on rgb
        amodal = self.rgb.copy()
        vis = self.rgb.copy()
        occ = self.rgb.copy()
        cmap = matplotlib.cm.get_cmap('gist_rainbow')

        for i, (amodal_mask, vis_mask, amodal_topleft, vis_topleft, occ_mask, occ_top_left) in enumerate(zip(amodal_masks, vis_masks, amodal_toplefts, vis_toplefts, occ_masks, occ_toplefts)):
            amodal[amodal_mask] = np.array(cmap(i/len(amodal_masks))[:3]) * 255 * 0.6 + amodal[amodal_mask] * 0.4
            vis[vis_mask] = np.array(cmap(i/len(vis_masks))[:3]) * 255 * 0.6 + vis[vis_mask] * 0.4
            occ[occ_mask] = np.array(cmap(i/len(occ_masks))[:3]) * 255 * 0.6 + occ[occ_mask] * 0.4
            amodal = cv2.putText(amodal, obj_names[i], amodal_topleft, cv2.FONT_HERSHEY_SIMPLEX, 0.7, np.array(cmap(i/len(vis_masks))[:3]) * 255, 2)
            if vis_topleft[0] != 0 and vis_topleft[1] != 0:
                vis = cv2.putText(vis, obj_names[i], vis_topleft, cv2.FONT_HERSHEY_SIMPLEX, 0.7, np.array(cmap(i/len(vis_masks))[:3]) * 255, 2)
            if occ_top_left[0] != 0 and occ_top_left[1] != 0:
                occ = cv2.putText(occ, obj_names[i], occ_top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, np.array(cmap(i/len(vis_masks))[:3]) * 255, 2)

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
        if k == 27:
            break
        if k == ord('r'):
            scene_id = gt_visualizer.scene_id
            scene_id -= 1
            if scene_id < 1:
                scene_id = 1
            gt_visualizer.on_scene_id(scene_id)
            gt_visualizer.is_img_loaded = False
        if k == ord('t'):
            scene_id = gt_visualizer.scene_id
            scene_id += 1
            if scene_id > 1000:
                scene_id = 1000
            gt_visualizer.on_scene_id(scene_id)
            gt_visualizer.is_img_loaded = False
        if k == ord('v'):
            try:
                scene_id = int(input("scene_id를 입력하세요: \n"))
            except:
                pass
            gt_visualizer.on_scene_id(scene_id)    
        if k == ord('f'):
            image_id = gt_visualizer.image_id
            image_id -= 1
            if image_id < 1:
                image_id = 1
            gt_visualizer.on_image_id(image_id)
            gt_visualizer.is_img_loaded = False
        if k == ord('g'):
            image_id = gt_visualizer.image_id
            image_id += 1
            if image_id > 52:
                image_id = 52
            gt_visualizer.on_image_id(image_id)
            gt_visualizer.is_img_loaded = False
        if k == ord('b'):
            try:
                image_id = int(input("image_id를 입력하세요: \n"))
            except:
                pass
            gt_visualizer.on_image_id(image_id)    
        
        if k == ord('q'):
            gt_visualizer.scale_factor += 0.1
            gt_visualizer.is_updated = False
        if k == ord('e'):
            gt_visualizer.scale_factor -= 0.1
            gt_visualizer.is_updated = False
        if k == ord('w'):
            gt_visualizer.icy -= 10
            if gt_visualizer.icy < 0:
                gt_visualizer.icy = 0
            gt_visualizer.is_updated = False
        if k == ord('s'):
            gt_visualizer.icy += 10
            if gt_visualizer.icy > gt_visualizer.height:
                gt_visualizer.icy = gt_visualizer.height
            gt_visualizer.is_updated = False
        if k == ord('a'):
            gt_visualizer.icx -= 10
            if gt_visualizer.icx < 0:
                gt_visualizer.icx = 0
            gt_visualizer.is_updated = False
        if k == ord('d'):
            gt_visualizer.icx += 10
            if gt_visualizer.icx > gt_visualizer.width:
                gt_visualizer.icx = gt_visualizer.width
            gt_visualizer.is_updated = False
        if k == ord('z'):
            gt_visualizer.icx = gt_visualizer.width//2
            gt_visualizer.icy = gt_visualizer.height//2
            gt_visualizer.scale_factor = 1
            gt_visualizer.is_updated = False



    gt_visualizer.stop_frame()
    cv2.destroyAllWindows()