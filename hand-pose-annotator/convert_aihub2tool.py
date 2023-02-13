import os
import shutil
import json
import yaml

SERIAL2NAME = {
        "000390922112": "1. 좌하단",
        "000355922112": "2. 좌중단",
        "000375922112": "3. 정상단",
        "000363922112": "4. 우중단",
        "000210922112": "5. 우하단"
    }

NAME2SERIAL = {v: k for k, v in SERIAL2NAME.items()}

MANO_CALIB = {
    "subject_1":"20220926-subject-01",
    "subject_2":"20221016-subject-02",
    "subject_3":"20221016-subject-03",
    "subject_4":"20221018-subject-04",
    "subject_5":"20221018-subject-05",
    "subject_6":"20221018-subject-06",
    "subject_8":"20221019-subject-08",
    "subject_7":"20221020-subject-07",
    "subject_9":"20221020-subject-09",
    "subject_10":"20221020-subject-10",
    "subject_12":"20221020-subject-12",
    "subject_13":"20221020-subject-13",
    "subject_11":"20221031-subject-11",
    "subject_14":"20221103-subject-14",
    "subject_15":"20221103-subject-15",
    "subject_16":"20221103-subject-16",
    "subject_18":"20221103-subject-18",
    "subject_19":"20221103-subject-19",
    "subject_20":"20221103-subject-20",
    "subject_22":"20221103-subject-22",
    "subject_17":"20221104-subject-17",
    "subject_21":"20221104-subject-21",
    "subject_24":"20221104-subject-24",
    "subject_25":"20221104-subject-25",
    "subject_23":"20221107-subject-23"
}



#TODO: Change this to your own path
aihub_data_root = "/media/raeyo/ssd_f/data4-source/04_사람-물체 파지"
aihub_label_root = "/media/raeyo/ssd_f/data4-labeling/04_사람-물체 파지"
target_root = "/media/raeyo/ssd_f/data4-target"

assert os.path.exists(aihub_data_root), "No AIHub Data"
assert os.path.exists(target_root), "No Target Data Directory"



for hand_type in os.listdir(aihub_data_root):
    hand_dir = os.path.join(aihub_data_root, hand_type)
    for obj_type in os.listdir(hand_dir):
        obj_dir = os.path.join(hand_dir, obj_type)
        for sc_name in os.listdir(obj_dir):
            aihub_sc_dir = os.path.join(obj_dir, sc_name)
            trg_sc_dir = os.path.join(target_root, sc_name)
            print("Copy {} to {}".format(aihub_sc_dir, trg_sc_dir))
            
            # generate scene meta from label
            print("Generate scene meta from label")
            aihub_label_json = os.path.join(aihub_sc_dir.replace(aihub_data_root, aihub_label_root), '1. 좌하단', 'gt', f'H4_2_{sc_name}_000001.json')
            with open(aihub_label_json, 'r') as f:
                aihub_label = json.load(f)
            print()
            
            meta = {
                'cam_calib': '20221010', # fix
                'mano_calib': MANO_CALIB[aihub_label['scene_info']['subject_id']],
                'mano_sides': list(aihub_label['hand_pose_info'].keys()),
                'obj_ids': [val['object_id'] for val in aihub_label['object_6d_pose_info']]
            }
            with open(os.path.join(trg_sc_dir, 'meta.yml'), 'w') as f:
                yaml.dump(meta, f)
            # copy images
            for cam_name in os.listdir(aihub_sc_dir):
                print("Copy {} to {}".format(cam_name, trg_sc_dir))
                serial = NAME2SERIAL[cam_name]
                src_cam_dir = os.path.join(aihub_sc_dir, cam_name)
                trg_cam_dir = os.path.join(trg_sc_dir, serial)
                if not os.path.exists(trg_cam_dir):
                    shutil.copytree(src_cam_dir, trg_cam_dir)
            
            
            
            
            
            
            
            
            
            
            
            
            
        








