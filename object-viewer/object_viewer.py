# Author: Seunghyeok Back (shback@gm.gist.ac.kr)
# GIST AILAB, Republic of Korea


import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import json
import cv2

from pathlib import Path
from os.path import basename, dirname

import numpy as np
import glob
import cv2
import os
import sys
import math


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    MATERIAL_NAMES = ["Unlit"]
    
    
    def __init__(self, width, height):

        self._annotation_scene = None
        self._annotation_changed = False
        self.current_scene_idx = None
        self.current_image_idx = None
        self.bounds = None
        self.coord_labels = []
        self.mesh_names = []
        self.window = gui.Application.instance.create_window(
            "GIST AILAB Object Viewer", width, height)
        w = self.window  # to make the code more 
        self.spl = "\\" if sys.platform.startswith("win") else "/"


        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.scene.set_background([1, 1, 1, 1])

        em = w.theme.font_size
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        self._fileedit = gui.TextEdit()
        filedlgbutton = gui.Button("파일 열기")
        filedlgbutton.horizontal_padding_em = 0.5
        filedlgbutton.vertical_padding_em = 0
        filedlgbutton.set_on_clicked(self._on_filedlg_button)

        fileedit_layout = gui.Horiz()
        fileedit_layout.add_child(gui.Label("파일 경로"))
        fileedit_layout.add_child(self._fileedit)
        fileedit_layout.add_fixed(0.25 * em)
        fileedit_layout.add_child(filedlgbutton)
        self._settings_panel.add_child(fileedit_layout)

        view_mesh_button = gui.Button("물체 자세히 보기")
        view_mesh_button.set_on_clicked(self._on_view_mesh_button)
        view_all_mesh_button = gui.Button("모든 물체 보기")
        view_all_mesh_button.set_on_clicked(self._on_view_all_mesh_button)
        hz = gui.Horiz(0.4 * em, gui.Margins(em, em, em, em))
        hz.add_child(view_mesh_button)
        hz.add_child(view_all_mesh_button)
        self._settings_panel.add_child(hz)

        self._meshes_available = gui.ListView()
        annotation_objects = gui.CollapsableVert("라벨링 대상 물체", 0.25 * em,
                                                 gui.Margins(0.25*em, 0, 0, 0))
        annotation_objects.set_is_open(True)
        annotation_objects.add_child(self._meshes_available)
        self._settings_panel.add_child(annotation_objects)

        self._log_panel = gui.VGrid(1, em)
        self._log = gui.Label("\t 불러올 물체 파일을 선택하세요. ")
        self._log_panel.add_child(self._log)

        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        w.add_child(self._log_panel)
        w.set_on_layout(self._on_layout)

       # ---- Menu ----
        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_separator()
            file_menu.add_item("종료하기", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("제작자 정보", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            menu.add_menu("파일", file_menu)
            menu.add_menu("도움말", help_menu)
            gui.Application.instance.menubar = menu
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        self.all_object_loaded = False

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        width_set = 17 * layout_context.theme.font_size
        height_set = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width_set, r.y, width_set,
                                              height_set)
        width_obj = 1.0 * width_set
        height_obj = 1.5 * layout_context.theme.font_size
        self._log_panel.frame = gui.Rect(r.get_right() - width_set - width_obj, r.y, width_obj, height_obj) 

    def _on_view_mesh_button(self):

        self._log.text = "\t물체를 불러옵니다."
        try:
            self._scene.scene.remove_geometry(self.curr_mesh_name)  # remove mesh from scene
            self.curr_mesh_name = self._meshes_available.selected_value
        except AttributeError:
            self._on_error("선택된 물체가 없습니다. error at _on_view_mesh_button")
            self._log.text = "\t올바른 물체 경로를 선택하세요."
            return
        if self.all_object_loaded:
            for mesh_name in self.mesh_names:
                self._scene.scene.remove_geometry(mesh_name)  # remove mesh from scene
            for label in self.mesh_labels:
                self._scene.remove_3d_label(label)
            self.all_object_loaded = False
        self._show_mesh()
        self._log.text = "\t물체를 불러왔습니다."

    def _on_view_all_mesh_button(self):

        self._log.text = "\t모든 물체를 불러오는 중입니다."
        if self.all_object_loaded:
            for mesh_name in self.mesh_names:
                self._scene.scene.remove_geometry(mesh_name)  # remove mesh from scene
            for label in self.mesh_labels:
                self._scene.remove_3d_label(label)

        self._scene.scene.remove_geometry(self.curr_mesh_name)  # remove mesh from scene
        n_mesh = len(self.mesh_names)
        n_grid = math.ceil(math.sqrt(n_mesh))
        gap = 0.3
        idx_to_pos = []
        for i in range(n_grid):
            for j in range(n_grid):
                idx_to_pos.append([i*gap, j*gap])
        self.mesh_labels = []
        for idx, mesh_name in enumerate(self.mesh_names):
            mesh_path = os.path.join(self.mesh_folder_path, mesh_name + ".ply")
            object_geometry = o3d.io.read_point_cloud(mesh_path)
            object_geometry.points = o3d.utility.Vector3dVector(
                np.array(object_geometry.points) / 1000)  # convert mm to meter
            init_trans = np.identity(4)
            init_trans[0:3, 3] = idx_to_pos[idx] + [0]
            object_geometry.transform(init_trans)
            material = rendering.MaterialRecord()
            material.base_color = [1.0, 1.0, 1.0, 1.0]
            material.shader = "defaultUnlit"
            material.point_size = 7
            self._scene.scene.add_geometry(mesh_name, object_geometry, material,
                                        add_downsampled_copy_for_fast_rendering=True)
            self.mesh_labels.append(self._scene.add_3d_label(idx_to_pos[idx] + [0], mesh_name))
        center = np.array([gap, gap, 0])
        eye = np.array([-0.2, -0.2, 0.4])
        up = np.array([0, 0, 1])
        self._scene.look_at(center, eye, up)
        self.all_object_loaded = True
        self._log.text = "\t모든 물체를 불러왔습니다."

    def _show_mesh(self):
        mesh_path = os.path.join(self.mesh_folder_path, self.curr_mesh_name + ".ply")
        if not os.path.exists(mesh_path):
            self._on_error("선택된 물체가 없습니다. error at _show_mesh")
            self._log.text = "\t물체를 선택하세요."
            return
        object_geometry = o3d.io.read_point_cloud(mesh_path)
        object_geometry.points = o3d.utility.Vector3dVector(
            np.array(object_geometry.points) / 1000)  # convert mm to meter
        init_trans = np.identity(4)
        center = object_geometry.get_center()
        init_trans[0:3, 3] = center
        object_geometry.transform(init_trans)
        bounds = object_geometry.get_axis_aligned_bounding_box()

        self._scene.setup_camera(60, bounds, bounds.get_center())
        center = np.array([0, 0, 0])
        eye = center + np.array([0, 0.25, 0.25])
        up = np.array([0, 0, 1])
        self._scene.look_at(center, eye, up)

        material = rendering.MaterialRecord()
        material.base_color = [1.0, 1.0, 1.0, 1.0]
        material.shader = "defaultUnlit"
        material.point_size = 7

        self._scene.scene.add_geometry(self.curr_mesh_name, object_geometry, material,
                                       add_downsampled_copy_for_fast_rendering=True)
        self._log.text = "\t물체를 불러왔습니다."

    def _on_filedlg_button(self):
        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "파일 선택",
                                 self.window.theme)
        filedlg.add_filter(".ply", "Mesh file (.ply)")
        filedlg.add_filter("", "모든 파일")
        filedlg.set_on_cancel(self._on_filedlg_cancel)
        filedlg.set_on_done(self._on_filedlg_done)
        self.window.show_dialog(filedlg)

    def _on_filedlg_cancel(self):
        self.window.close_dialog()

    def _on_filedlg_done(self, mesh_path):
        if mesh_path[-4:] != ".ply":
            self._on_error("잘못된 경로를 입력하였습니다. (error at _on_filedlg_done)")
            self._log.text = "\t올바른 물체 경로를 선택하세요."
            return
        self._fileedit.text_value = mesh_path
        self.mesh_folder_path = str(Path(mesh_path).parent)
        self.obj_ids = sorted([int(os.path.basename(x)[5:-4]) for x in glob.glob(self.mesh_folder_path + self.spl + '*.ply')])
        self.mesh_names = ['obj_' + f'{ + obj_id:06}' for obj_id in self.obj_ids]
        self._meshes_available.set_items(self.mesh_names)
        self.window.close_dialog()
        self.curr_mesh_name = basename(mesh_path)[:-4]
        self._show_mesh()


    def _on_error(self, err_msg):
        dlg = gui.Dialog("Error")

        em = self.window.theme.font_size
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(err_msg))

        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()


    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_about(self):
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")
        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("GIST AILAB Object Viewer.\nCopyright (c) 2022 Seunghyeok Back\nGwangju Institute of Science and Technology (GIST)\nshback@gm.gist.ac.kr"))
        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)
        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)


def main():


    gui.Application.instance.initialize()
    hangeul = "./lib/NanumGothic.ttf"
    font = gui.FontDescription(hangeul)
    font.add_typeface_for_language(hangeul, "ko")
    gui.Application.instance.set_font(gui.Application.DEFAULT_FONT_ID, font)

    w = AppWindow(1920, 1080)
    gui.Application.instance.run()


if __name__ == "__main__":

    main()
