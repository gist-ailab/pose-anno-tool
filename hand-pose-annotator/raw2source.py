import os
import sys
import shutil

import open3d as o3d
import numpy as np
_SERIALS = [
        '000056922112', # master
        '000210922112',
        '000295922112',
        '000355922112',
        '000363922112',
        '000375922112',
        '000390922112',
        '000480922112',
    ]

raw_path = 'E:/OccludedObjectDataset/data_4/data4-raw'
source_path = 'E:/OccludedObjectDataset/data_4/data4-source'

os.makedirs(source_path, exist_ok=True)

for sc_dir in [os.path.join(raw_path, p) for p in os.listdir(raw_path)]:
    src_sc_dir = sc_dir.replace(raw_path, source_path)
    if os.path.isdir(src_sc_dir):
        continue
    os.makedirs(src_sc_dir, exist_ok=True)
    sc_name = os.path.basename(sc_dir)
    if sc_name in ['calibration', 'models']:
        shutil.copytree(sc_dir, src_sc_dir)
        continue
    meta_file = os.path.join(sc_dir, "meta.yml")
    src_meta = meta_file.replace(raw_path, source_path)
    shutil.copy(meta_file, src_meta)

    for serial in _SERIALS:
        cam_dir = os.path.join(sc_dir, serial)
        src_cam_dir = cam_dir.replace(raw_path, source_path)
        os.makedirs(src_cam_dir, exist_ok=True)

        # rgb, depth, hand_mocap
        rgb_dir = os.path.join(cam_dir, 'rgb')
        depth_dir = os.path.join(cam_dir, 'depth')
        pcd_dir = os.path.join(cam_dir, 'pcd')

        src_rgb_dir = rgb_dir.replace(raw_path, source_path)
        src_depth_dir = depth_dir.replace(raw_path, source_path)
        src_pcd_dir = pcd_dir.replace(raw_path, source_path)
        os.makedirs(src_rgb_dir, exist_ok=True)
        os.makedirs(src_depth_dir, exist_ok=True)
        os.makedirs(src_pcd_dir, exist_ok=True)


        frame_list = [int(os.path.splitext(p)[0]) for p in os.listdir(rgb_dir)]
        frame_list.sort()
        frame_list = frame_list[30:]
        interval = len(frame_list)//50
        select_frame_list = frame_list[::interval]
        select_frame_list = select_frame_list[:50]
        print(len(select_frame_list))

        for frame_id in select_frame_list:
            rgb = os.path.join(rgb_dir, "{:06d}.png".format(frame_id))
            src_rgb = rgb.replace(raw_path, source_path)
            shutil.copy(rgb, src_rgb)
            depth = os.path.join(depth_dir, "{:06d}.png".format(frame_id))
            src_depth = depth.replace(raw_path, source_path)
            shutil.copy(depth, src_depth)
            
        try:
            hand_mocap_dir = os.path.join(cam_dir, 'hand_mocap')
            src_hand_mocap_dir = hand_mocap_dir.replace(raw_path, source_path)
            shutil.copytree(hand_mocap_dir, src_hand_mocap_dir)
        except:
            pass

        for frame_id in select_frame_list:
            pcd_path = os.path.join(pcd_dir, "{:06d}.pcd".format(frame_id))
            src_pcd = pcd_path.replace(raw_path, source_path)
            pcd = o3d.io.read_point_cloud(pcd_path)
            pcd.scale(10, [0, 0, 0])
            center = np.array([0.1694395303428173,
                            -0.26324927164241674,
                            0.7835706684403122])
            pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(center-0.4, center+0.4))
            o3d.io.write_point_cloud(src_pcd, pcd)














