# 6DoF pose annotator 
# Shuichi Akizuki, Chukyo Univ.
# Email: s-akizuki@sist.chukyo-u.ac.jp
#
import open3d as o3d 
import numpy as np
import cv2
import copy
import argparse
import os
import utils as c3D
from math import *
import json
import glob
import vedo
from src.utils import *
from src.annotator import Annotator
# from src.gui import *
import sys
import dearpygui.dearpygui as dpg




if __name__ == "__main__":


    dpg.create_context()

    with dpg.texture_registry(show=True):
        width, height, channels, data = dpg.load_image("./tmp/obj_list.png")
        dpg.add_static_texture(width, height, data, tag="object_lists")
        width, height, channels, data = dpg.load_image("./assets/imgs/gist_logo.png")
        dpg.add_static_texture(width, height, data, tag="gist_logo")
        width, height, channels, data = dpg.load_image("./assets/imgs/ailab_logo.png")
        dpg.add_static_texture(width, height, data, tag="ailab_logo")
        dpg.add_dynamic_texture(640, 480, np.zeros([480, 640, 4]), tag="2d_anno_vis")
        dpg.add_dynamic_texture(640, 480, np.zeros([480, 640, 4]), tag="3d_object_vis")
        dpg.add_dynamic_texture(640, 480, np.zeros([480, 640, 4]), tag="3d_anno_vis")

    annotator = Annotator()
    # object selection button


    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=annotator.keyboard_press_callback)
        dpg.add_mouse_double_click_handler(callback=annotator.mouse_click_callback)

    with dpg.window(label="Object Lists"):
        dpg.add_image("object_lists") 
        for obj_name in annotator.obj_names:
            dpg.add_button(label=obj_name, callback=annotator.object_select_button_callback, tag=obj_name)

    with dpg.window(label="GIST", pos=[0, 730]):
        dpg.add_image("gist_logo") 

    with dpg.window(label="AILAB", pos=[230, 730]):
        dpg.add_image("ailab_logo") 

    with dpg.window(label="2D Annotation Visualization", pos=[512, 0]):
        dpg.add_image("2d_anno_vis")
        dpg.add_slider_float(label="trasparency", callback=annotator.trasparency_callback, min_value=0.0, max_value=1.0)
        # dpg.add_slider_float(label="x_axis", callback=annotator.x_axis_callback, min_value=0.0, max_value=1.0)
        # dpg.add_slider_float(label="y_axis", callback=annotator.y_axis_callback, min_value=0.0, max_value=1.0)
        # dpg.add_slider_float(label="z_axis", callback=annotator.z_axis_callback, min_value=0.0, max_value=1.0)

    with dpg.window(label="3D Object Visualization", pos=[512+640, 0]):
        dpg.add_image("3d_object_vis")


    dpg.create_viewport(title='GIST 6D Object Pose Annotator', width=1920, height=1080)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

