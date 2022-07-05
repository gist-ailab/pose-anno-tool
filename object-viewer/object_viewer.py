# Author: Seunghyeok Back (shback@gm.gist.ac.kr)
# GIST AILAB, Republic of Korea
# Modified the codes of Anas Gouda (anas.gouda@tu-dortmund.de)
# FLW, TU Dortmund, Germany

"""Manual annotation tool for datasets with BOP format

Using RGB, Depth and Models the tool will generate the "scene_gt.json" annotation file

Other annotations can be generated usign other scripts [calc_gt_info.py, calc_gt_masks.py, ....]

original repo: https://github.com/FLW-TUDO/3d_annotation_tool

"""

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


class Settings:
    UNLIT = "defaultUnlit"

    def __init__(self):
        self.bg_color = gui.Color(1, 1, 1)
        self.show_axes = False
        self.show_coord_frame = False
        self.show_mesh_names = False
        self.highlight_obj = True
        self.transparency = 0.5

        self.apply_material = True  # clear to False after processing

        self.scene_material = rendering.MaterialRecord()
        self.scene_material.base_color = [1.0, 1.0, 1.0, 1.0]
        self.scene_material.shader = Settings.UNLIT

        self.annotation_obj_material = rendering.MaterialRecord()
        self.annotation_obj_material.base_color = [0.9, 0.3, 0.3, 1 - self.transparency]
        self.annotation_obj_material.shader = Settings.UNLIT

        self.annotation_active_obj_material = rendering.MaterialRecord()
        self.annotation_active_obj_material.base_color = [0.3, 0.9, 0.3, 1 - self.transparency]
        self.annotation_active_obj_material.shader = Settings.UNLIT


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    MATERIAL_NAMES = ["Unlit"]
    MATERIAL_SHADERS = [
        Settings.UNLIT
    ]
    
    
    def __init__(self, width, height):

        self._annotation_scene = None
        self._annotation_changed = False
        self.current_scene_idx = None
        self.current_image_idx = None
        self.bounds = None
        self.coord_labels = []
        self.mesh_names = []
        self.settings = Settings()
        self.window = gui.Application.instance.create_window(
            "3D Object Viewer by GIST AILAB", width, height)
        w = self.window  # to make the code more 

        self.spl = "\\" if sys.platform.startswith("win") else "/"

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

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

        annotation_objects = gui.CollapsableVert("라벨링 대상 물체", 0.25 * em,
                                                 gui.Margins(0.25*em, 0, 0, 0))
        annotation_objects.set_is_open(True)
        self._meshes_available = gui.ListView()
        self._meshes_available.set_on_selection_changed(self._on_selection_changed)
        view_mesh_button = gui.Button("물체 자세히 보기")
        view_mesh_button.set_on_clicked(self._view_mesh)
        hz = gui.Horiz(spacing=5)
        hz.add_child(view_mesh_button)
        annotation_objects.add_child(hz)
        annotation_objects.add_child(self._meshes_available)
        self._settings_panel.add_child(annotation_objects)

        w.add_child(self._settings_panel)
        w.set_on_layout(self._on_layout)

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

    def _view_mesh(self):
        pass

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
        if mesh_path[:-3] != "ply":
            self._on_error("잘못된 경로를 입력하였습니다. (error at _on_filedlg_done)")
            return
        self._fileedit.text_value = mesh_path
        self.mesh_folder_path = str(Path(mesh_path.parent))
        print(self.mesh_folder_path)
        self.obj_id = sorted([int(os.path.basename(x)[5:-4]) for x in glob.glob(self.mesh_folder_path + self.spl + '*.ply')])
        print(self.obj_id)
        self.model_names = ['obj_' + f'{ + obj_id:06}' for obj_id in self.obj_ids]
        print(self.model_names)
        self._meshes_available.set_items(self.model_names)
    
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


    def _on_selection_changed(self, a, b):
        self._log.text = "\t라벨링 대상 물체를 변경합니다."
        objects = self._annotation_scene.get_objects()
        for obj in objects:
            self._scene.scene.remove_geometry(obj.obj_name)
            self._scene.scene.add_geometry(obj.obj_name, obj.obj_geometry,
                                        self.settings.annotation_obj_material,
                                        add_downsampled_copy_for_fast_rendering=True)
        active_obj = objects[self._meshes_available.selected_index]
        self._scene.scene.remove_geometry(active_obj.obj_name)
        self._scene.scene.add_geometry(active_obj.obj_name, active_obj.obj_geometry,
                                    self.settings.annotation_active_obj_material,
                                    add_downsampled_copy_for_fast_rendering=True)
        self.inst_id_edit.set_value(int(active_obj.obj_name.split("_")[-1]))
        self._apply_settings()


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
