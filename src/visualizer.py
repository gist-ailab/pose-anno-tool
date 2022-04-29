import open3d as o3d 
import numpy as np
import cv2
import copy
import argparse
import os
import common3Dfunc as c3D
from math import *
import json
import glob
import vedo
import imgviz
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

def trim(im, border):
  bg = Image.new(im.mode, im.size, border)
  diff = ImageChops.difference(im, bg)
  bbox = diff.getbbox()
  if bbox:
    return im.crop(bbox)


def create_thumbnail(path, size):
  image = Image.open(path)
  name, extension = path.split('.')
  options = {}
  if 'transparency' in image.info:
    options['transparency'] = image.info["transparency"]
  
  image.thumbnail((size, size), Image.ANTIALIAS)
  image = trim(image, 255) ## Trim whitespace
  return image

def create_obj_list_img(obj_ply_paths, obj_names):

    print("Create object list images ...")
    imgs = []
    for obj_ply_path, obj_name in zip(obj_ply_paths, obj_names):
        o3d_mesh = o3d.io.read_triangle_mesh(obj_ply_path)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False) #works for me with False, on some systems needs to be true
        vis.add_geometry(o3d_mesh)
        vis.update_geometry(o3d_mesh)
        vis.get_view_control().set_front([1, 1, 0])
        vis.get_view_control().set_up([0, 0, 1])
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image("tmp/tmp.png", do_render=True)
        vis.destroy_window()
        img = np.uint8(create_thumbnail('tmp/tmp.png', 256))
        img = cv2.putText(img, obj_name,  (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        imgs.append(img)
    tiled = imgviz.tile(imgs=imgs, border=(255, 255, 255), cval=(255, 255, 255))
    plt.figure(dpi=700)
    plt.imshow(tiled)
    plt.axis("off")
    img = imgviz.io.pyplot_to_numpy()
    plt.close()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (512, 512))
    cv2.imwrite("tmp/obj_list.png", img)
    obj_list_img = cv2.imread("tmp/obj_list.png")
    return obj_list_img