# Author: Seunghyeok Back (shback@gm.gist.ac.kr)
# GIST AILAB, Republic of Korea
# Modified from the codes of Anas Gouda (anas.gouda@tu-dortmund.de)
# FLW, TU Dortmund, Germany

import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import shutil
import os
import json
import cv2
import datetime
import numpy as np
import os
import sys
from pathlib import Path
from os.path import basename, dirname







class Dataset:
    def __init__(self, dataset_path, dataset_split):
        self.scenes_path = os.path.join(dataset_path, dataset_split)
        self.objects_path = os.path.join(dataset_path, 'models')


class AnnotationScene:
    def __init__(self, scene_point_cloud, scene_num, image_num):
        self.annotation_scene = scene_point_cloud
        self.scene_num = scene_num
        self.image_num = image_num

        self.obj_list = list()

    def add_obj(self, obj_geometry, obj_name, obj_instance, transform=np.identity(4)):
        self.obj_list.append(self.SceneObject(obj_geometry, obj_name, obj_instance, transform))

    def get_objects(self):
        return self.obj_list[:]

    def remove_obj(self, index):
        self.obj_list.pop(index)

    class SceneObject:
        def __init__(self, obj_geometry, obj_name, obj_instance, transform):
            self.obj_geometry = obj_geometry
            self.obj_name = obj_name
            self.obj_instance = obj_instance
            self.transform = transform


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

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_axes(self.settings.show_axes)

        if self.settings.apply_material:
            self._scene.scene.modify_geometry_material("annotation_scene", self.settings.scene_material)
            self.settings.apply_material = False
        self._show_axes.checked = self.settings.show_axes
        self._highlight_obj.checked = self.settings.highlight_obj
        self._show_coord_frame.checked = self.settings.show_coord_frame
        self._show_mesh_names.checked = self.settings.show_mesh_names
        self._point_size.double_value = self.settings.scene_material.point_size

        if self.settings.show_coord_frame:
            self._add_coord_frame("obj_coord_frame", size=0.1)
            self._add_coord_frame("world_coord_frame")
        else:
            self._scene.scene.remove_geometry("obj_coord_frame")
            self._scene.scene.remove_geometry("world_coord_frame")
            for label in self.coord_labels:
                self._scene.remove_3d_label(label)
            self.coord_labels = []

        if self.settings.show_mesh_names:
            try:
                self._update_and_show_mesh_name()
            except:
                self._on_error("라벨링 대상 파일을 선택하세요 (error at _apply_settings")
                pass
        else:
            for inst_label in self.mesh_names:
                self._scene.remove_3d_label(inst_label)

    def _update_and_show_mesh_name(self):
        meshes = self._annotation_scene.get_objects()  # get new list after deletion
        for inst_label in self.mesh_names:
            self._scene.remove_3d_label(inst_label)
        for mesh in meshes:
            self.mesh_names.append(self._scene.add_3d_label(mesh.transform[:3, 3], mesh.obj_name))

    def _add_coord_frame(self, name="coord_frame", size=0.2, origin=[0, 0, 0]):
        if self._annotation_scene is None: # shsh
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _add_coord_frame)")
            return
        objects = self._annotation_scene.get_objects()
        try:
            active_obj = objects[self._meshes_used.selected_index]
        except IndexError:
            self._on_error("라벨링 대상 물체를 선택하세요. (error at _add_coord_frame)")
            return
        self._scene.scene.remove_geometry(name)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
        if "world" in name:
            transform = np.eye(4)
            transform[:3, 3] = active_obj.transform[:3, 3]
            coord_frame.transform(transform)
            for label in self.coord_labels:
                self._scene.remove_3d_label(label)
            self.coord_labels = []
            size = size * 0.6
            self.coord_labels.append(self._scene.add_3d_label(active_obj.transform[:3, 3] + np.array([size, 0, 0]), "D (+)"))
            self.coord_labels.append(self._scene.add_3d_label(active_obj.transform[:3, 3] + np.array([-size, 0, 0]), "A (-)"))
            self.coord_labels.append(self._scene.add_3d_label(active_obj.transform[:3, 3] + np.array([0, size, 0]), "S (+)"))
            self.coord_labels.append(self._scene.add_3d_label(active_obj.transform[:3, 3] + np.array([0, -size, 0]), "W (-)"))
            self.coord_labels.append(self._scene.add_3d_label(active_obj.transform[:3, 3] + np.array([0, 0, size]), "Q (+)"))
            self.coord_labels.append(self._scene.add_3d_label(active_obj.transform[:3, 3] + np.array([0, 0, -size]), "E (-)"))

        else:
            coord_frame.transform(active_obj.transform)
        self._scene.scene.add_geometry(name, coord_frame, 
                                        self.settings.annotation_obj_material,
                                        add_downsampled_copy_for_fast_rendering=True) 

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
        width_im = min(
            r.width,
            self._images_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).width * 1.1)
        height_im = min(
            r.height,
            self._images_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height * 1.1)                     
        self._images_panel.frame = gui.Rect(0, r.y, width_im, height_im)   

        width_obj = 1.5 * width_set
        height_obj = 1.5 * layout_context.theme.font_size
        self._log_panel.frame = gui.Rect(r.get_right() - width_set - width_obj, r.y, width_obj, height_obj) 

    def __init__(self, width, height):

        self._annotation_scene = None
        self._annotation_changed = False
        self.current_scene_idx = None
        self.current_image_idx = None
        self.upscale_responsiveness = False
        self.bounds = None
        self.coord_labels = []
        self.mesh_names = []
        self.settings = Settings()
        self.window = gui.Application.instance.create_window(
            "6D Object Pose Annotator by GIST AILAB", width, height)
        w = self.window  # to make the code more 

        self.spl = "\\" if sys.platform.startswith("win") else "/"

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        # ---- Settings panel ----
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

        view_ctrls = gui.CollapsableVert("편의 기능", 0,
                                         gui.Margins(em, 0, 0, 0))
        view_ctrls.set_is_open(True)

        self._show_axes = gui.Checkbox("카메라 좌표계 보기")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_child(self._show_axes)

        self._highlight_obj = gui.Checkbox("라벨링 대상 물체 강조하기")
        self._highlight_obj.set_on_checked(self._on_highlight_obj)
        view_ctrls.add_child(self._highlight_obj)

        self._show_coord_frame = gui.Checkbox("물체 좌표계 보기")
        self._show_coord_frame.set_on_checked(self._on_show_coord_frame)
        view_ctrls.add_child(self._show_coord_frame)

        self._show_mesh_names = gui.Checkbox("물체 이름 보기")
        self._show_mesh_names.set_on_checked(self._on_show_mesh_names)
        view_ctrls.add_child(self._show_mesh_names)

        self._transparency = gui.Slider(gui.Slider.DOUBLE)
        self._transparency.set_limits(0, 1)
        self._transparency.set_on_value_changed(self._on_transparency)

        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)

        self.dist = 0.0004 * 5
        self.deg = 0.2 * 5
        self._responsiveness = gui.Slider(gui.Slider.INT)
        self._responsiveness.set_limits(1, 20)
        self._responsiveness.set_on_value_changed(self._on_responsiveness)
        self._responsiveness.double_value = 5.0

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("투명도"))
        grid.add_child(self._transparency)
        grid.add_child(gui.Label("포인트 크기"))
        grid.add_child(self._point_size)
        grid.add_child(gui.Label("민감도"))
        grid.add_child(self._responsiveness)
        view_ctrls.add_child(grid)

        self._settings_panel.add_child(view_ctrls)
        # ----
        self._images_panel = gui.CollapsableVert("이미지 보기", 0.33 * em,
                                                 gui.Margins(em, 0, 0, 0))
        self._rgb_proxy = gui.WidgetProxy()
        self._rgb_proxy.set_widget(gui.ImageWidget())
        self._images_panel.add_child(self._rgb_proxy)
        self._images_panel.set_is_open(False)
        
        self._log_panel = gui.VGrid(1, em)
        self._log = gui.Label("\t 라벨링 대상 파일을 선택하세요. ")
        self._log_panel.add_child(self._log)

        # 3D Annotation tool options
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        w.add_child(self._images_panel)
        w.add_child(self._log_panel)
        w.set_on_layout(self._on_layout)

        annotation_objects = gui.CollapsableVert("라벨링 대상 물체", 0.25 * em,
                                                 gui.Margins(0.25*em, 0, 0, 0))
        annotation_objects.set_is_open(True)
        self._meshes_available = gui.ListView()
        self._meshes_used = gui.ListView()
        self._meshes_used.set_on_selection_changed(self._on_selection_changed)
        add_mesh_button = gui.Button("물체 추가하기")
        remove_mesh_button = gui.Button("물체 삭제하기")
        add_mesh_button.set_on_clicked(self._add_mesh)
        remove_mesh_button.set_on_clicked(self._remove_mesh)
        annotation_objects.add_child(self._meshes_available)
        hz = gui.Horiz(spacing=5)
        hz.add_child(add_mesh_button)
        hz.add_child(remove_mesh_button)
        annotation_objects.add_child(hz)
        annotation_objects.add_child(self._meshes_used)

        # x, y, z axis
        x_grid = gui.VGrid(3, 0.25 * em)
        self._x_rot = gui.Slider(gui.Slider.DOUBLE)
        self._x_rot.set_limits(-0.5, 0.5)
        self._x_rot.set_on_value_changed(self._on_x_rot)
        x_grid.add_child(gui.Label("x축",))
        x_grid.add_child(self._x_rot)
        annotation_objects.add_child(x_grid)

        y_grid = gui.VGrid(3, 0.25 * em)
        self._y_rot = gui.Slider(gui.Slider.DOUBLE)
        self._y_rot.set_limits(-0.5, 0.5)
        self._y_rot.set_on_value_changed(self._on_y_rot)
        y_grid.add_child(gui.Label("y축",))
        y_grid.add_child(self._y_rot)
        annotation_objects.add_child(y_grid)

        z_grid = gui.VGrid(3, 0.25 * em)
        self._z_rot = gui.Slider(gui.Slider.DOUBLE)
        self._z_rot.set_limits(-0.5, 0.5)
        self._z_rot.set_on_value_changed(self._on_z_rot)
        z_grid.add_child(gui.Label("z축",))
        z_grid.add_child(self._z_rot)
        annotation_objects.add_child(z_grid)

        inst_grid = gui.VGrid(3, 0.25 * em)
        self.inst_id_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self.inst_id_edit.int_value = 0
        self.inst_id_edit.set_limits(0, 30)
        self.inst_id_edit.set_on_value_changed(self._on_inst_value_changed)

        inst_grid.add_child(gui.Label("인스턴스 아이디", ))
        inst_grid.add_child(self.inst_id_edit)
        annotation_objects.add_child(inst_grid)

        self._settings_panel.add_child(annotation_objects)

        self._scene_control = gui.CollapsableVert("작업 파일 리스트", 0.33 * em,
                                                  gui.Margins(0.25 * em, 0, 0, 0))
        self._scene_control.set_is_open(True)

        self._images_buttons_label = gui.Label("이미지:")
        self._samples_buttons_label = gui.Label("작업 폴더: ")

        self._pre_image_button = gui.Button("이전")
        self._pre_image_button.horizontal_padding_em = 0.8
        self._pre_image_button.vertical_padding_em = 0
        self._pre_image_button.set_on_clicked(self._on_previous_image)
        self._next_image_button = gui.Button("다음")
        self._next_image_button.horizontal_padding_em = 0.8
        self._next_image_button.vertical_padding_em = 0
        self._next_image_button.set_on_clicked(self._on_next_image)
        self._pre_sample_button = gui.Button("이전")
        self._pre_sample_button.horizontal_padding_em = 0.8
        self._pre_sample_button.vertical_padding_em = 0
        self._pre_sample_button.set_on_clicked(self._on_previous_scene)
        self._next_sample_button = gui.Button("다음")
        self._next_sample_button.horizontal_padding_em = 0.8
        self._next_sample_button.vertical_padding_em = 0
        self._next_sample_button.set_on_clicked(self._on_next_scene)
        # 2 rows for sample and scene control
        h = gui.Horiz(0.4 * em)  # row 1
        h.add_stretch()
        h.add_child(self._images_buttons_label)
        h.add_child(self._pre_image_button)
        h.add_child(self._next_image_button)
        h.add_stretch()
        self._scene_control.add_child(h)
        h = gui.Horiz(0.4 * em)  # row 2
        h.add_stretch()
        h.add_child(self._samples_buttons_label)
        h.add_child(self._pre_sample_button)
        h.add_child(self._next_sample_button)
        h.add_stretch()
        self._scene_control.add_child(h)

        progress_ctrls = gui.Vert(em)
        self._progress = gui.ProgressBar()
        self._progress.value = 0.0  # 25% complete
                # self._image_number = gui.Label("Image: " + f'{0:06}')
        prog_layout = gui.Vert(em)
        prog_layout.add_child(self._progress)
        self._progress_str = gui.Label("진행률: 0.0% [0/0]")
        progress_ctrls.add_child(self._progress_str)
        progress_ctrls.add_child(self._progress)
        self._scene_control.add_child(progress_ctrls)

        self._view_numbers = gui.Horiz(0.4 * em)
        self._image_number = gui.Label("이미지: " + f'{0:06}')
        self._scene_number = gui.Label("작업폴더: " + f'{0:06}')

        self._view_numbers.add_child(self._image_number)
        self._view_numbers.add_child(self._scene_number)
        self._scene_control.add_child(self._view_numbers)

        self._settings_panel.add_child(self._scene_control)
        initial_viewpoint = gui.Button("처음 시점으로 이동하기 (T)")
        initial_viewpoint.set_on_clicked(self._on_initial_viewpoint)
        self._scene_control.add_child(initial_viewpoint)
        refine_position = gui.Button("자동 정렬하기 (R)")
        refine_position.set_on_clicked(self._on_refine)
        generate_save_annotation = gui.Button("저장하기 (F)")
        generate_save_annotation.set_on_clicked(self._on_generate)
        self._scene_control.add_child(refine_position)
        self._scene_control.add_child(generate_save_annotation)

        # ---- Menu ----
        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_separator()
            file_menu.add_item("종료하기", AppWindow.MENU_QUIT)
            help_menu = gui.Menu()
            help_menu.add_item("제작자 정보", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            menu.add_menu("파일", file_menu)
            menu.add_menu("도움말", help_menu)
            gui.Application.instance.menubar = menu

        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        
        # ---- annotation tool settings ----
        self._on_transparency(0.5)
        self._on_point_size(5)  # set default size to 1
        self._apply_settings()

        # set callbacks for key control
        self._scene.set_on_key(self._transform)
        self._left_shift_modifier = False
        self._scene.set_on_mouse(self._on_mouse)
        self._log.text = "\t라벨링 대상 파일을 선택하세요."

        self.offscreen_render = rendering.OffscreenRenderer(640, 480)
        img = self.offscreen_render.render_to_image()
        o3d.io.write_image("test.png", img, 9)





    def _on_x_rot(self, new_val):
        try:
            self.move( 0, 0, 0, new_val * np.pi / 180, 0, 0)
        except:
            self._on_error("라벨링 대상 물체를 선택하세요 (error at _on_x_rot).")
        self._x_rot.int_value = 0      

    def _on_y_rot(self, new_val):
        try:
            self.move( 0, 0, 0, 0, new_val * np.pi / 180, 0)
        except:
            self._on_error("라벨링 대상 물체를 선택하세요 (error at _on_y_rot).")
        self._y_rot.int_value = 0     

    def _on_z_rot(self, new_val):
        try:
            self.move( 0, 0, 0, 0, 0, new_val * np.pi / 180)
        except:
            self._on_error("라벨링 대상 물체를 선택하세요 (error at _on_z_rot).")
        self._z_rot.int_value = 0     

    def _on_inst_value_changed(self, new_val):
        idx = self._meshes_used.selected_index
        obj_name = self._annotation_scene.get_objects()[idx].obj_name
        self._annotation_scene.get_objects()[idx].obj_instance = int(new_val)
        self._annotation_scene.get_objects()[idx].obj_name = "obj_" + obj_name.split("_")[1] + "_" + str(int(new_val))
        meshes = self._annotation_scene.get_objects()  # update list after adding current object
        meshes = [i.obj_name for i in meshes]
        self._meshes_used.set_items(meshes)
        self._meshes_used.selected_index = idx
        if self.settings.show_mesh_names:
            self._update_and_show_mesh_name()
        self._log.text = "\t인스턴스 아이디를 변경했습니다."

    def _on_filedlg_button(self):
        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "파일 선택",
                                 self.window.theme)
        filedlg.add_filter(".png .jpg", "RGB Image (.png, .jpg)")
        filedlg.add_filter("", "모든 파일")
        filedlg.set_on_cancel(self._on_filedlg_cancel)
        filedlg.set_on_done(self._on_filedlg_done)
        self.window.show_dialog(filedlg)

    def _on_filedlg_cancel(self):
        self.window.close_dialog()

    def _on_filedlg_done(self, rgb_path):
        self._fileedit.text_value = rgb_path
        dataset_path = str(Path(rgb_path).parent.parent.parent.parent)
        split_and_type = basename(str(Path(rgb_path).parent.parent.parent))
        self.scenes = Dataset(dataset_path, split_and_type)
        try:
            start_scene_num = int(basename(str(Path(rgb_path).parent.parent)))
            start_image_num = int(basename(rgb_path)[:-4])
            self.scene_num_lists = sorted([int(basename(x)) for x in glob.glob(dirname(str(Path(rgb_path).parent.parent)) + self.spl + "*") if os.path.isdir(x)])
            self.current_scene_idx = self.scene_num_lists.index(start_scene_num)
            self.image_num_lists = sorted([int(basename(x).split(".")[0]) for x in glob.glob(dirname(str(Path(rgb_path))) + self.spl + "*.png")])
            self.current_image_idx = self.image_num_lists.index(start_image_num)
            if os.path.exists(self.scenes.scenes_path) and os.path.exists(self.scenes.objects_path):
                self.update_obj_list()
                self.scene_load(self.scenes.scenes_path, start_scene_num, start_image_num)
                self._progress.value = (self.current_image_idx + 1) / len(self.image_num_lists) # 25% complete
                self._progress_str.text = "진행률: {:.1f}% [{}/{}]".format(
                    100 * (self.current_image_idx + 1) / len(self.image_num_lists), 
                    self.current_image_idx + 1, len(self.image_num_lists))
            self.window.close_dialog()
            self._log.text = "\t라벨링 대상 파일을 불러왔습니다."
        except Exception as e:
            print(e)
            self._on_error("잘못된 경로가 입력되었습니다. (error at _on_filedlg_done)")
            self._log.text = "\t올바른 파일 경로를 선택하세요."



    def _update_scene_numbers(self):
        self._scene_number.text = "작업 폴더: " + f'{self._annotation_scene.scene_num:06}'
        self._image_number.text = "이미지: " + f'{self._annotation_scene.image_num:06}'

    def move(self, x, y, z, rx, ry, rz):
        self._annotation_changed = True
        objects = self._annotation_scene.get_objects()
        active_obj = objects[self._meshes_used.selected_index]
        # translation or rotation
        if x != 0 or y != 0 or z != 0:
            h_transform = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
        else:  # elif rx!=0 or ry!=0 or rz!=0:
            center = active_obj.obj_geometry.get_center()
            rot_mat_obj_center = active_obj.obj_geometry.get_rotation_matrix_from_xyz((rx, ry, rz))
            T_neg = np.vstack((np.hstack((np.identity(3), -center.reshape(3, 1))), [0, 0, 0, 1]))
            R = np.vstack((np.hstack((rot_mat_obj_center, [[0], [0], [0]])), [0, 0, 0, 1]))
            T_pos = np.vstack((np.hstack((np.identity(3), center.reshape(3, 1))), [0, 0, 0, 1]))
            h_transform = np.matmul(T_pos, np.matmul(R, T_neg))
        active_obj.obj_geometry.transform(h_transform)
        center = active_obj.obj_geometry.get_center()
        self._scene.scene.remove_geometry(active_obj.obj_name)
        self._scene.scene.add_geometry(active_obj.obj_name, active_obj.obj_geometry,
                                        self.settings.annotation_active_obj_material,
                                        add_downsampled_copy_for_fast_rendering=True)
                                    
        # update values stored of object
        active_obj.transform = np.matmul(h_transform, active_obj.transform)

        if self.settings.show_coord_frame:
            self._add_coord_frame("obj_coord_frame", size=0.1)
            self._add_coord_frame("world_coord_frame")
        if self.settings.show_mesh_names:
            self._update_and_show_mesh_name()


    def _transform(self, event):
        if event.key == gui.KeyName.ESCAPE:
            self._on_generate()
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key == gui.KeyName.LEFT_SHIFT or event.key == gui.KeyName.RIGHT_SHIFT:
            if event.type == gui.KeyEvent.DOWN:
                self._left_shift_modifier = True
            elif event.type == gui.KeyEvent.UP:
                self._left_shift_modifier = False
            return gui.Widget.EventCallbackResult.HANDLED

        # if ctrl is pressed then increase translation and angle values
        if event.key == gui.KeyName.LEFT_CONTROL or event.key == gui.KeyName.RIGHT_CONTROL:
            if event.type == gui.KeyEvent.DOWN:
                if not self.upscale_responsiveness:
                    self.dist = self.dist * 15
                    self.deg = self.deg * 15
                    self.upscale_responsiveness = True
            elif event.type == gui.KeyEvent.UP:
                if self.upscale_responsiveness:
                    self.dist = self.dist / 15
                    self.deg = self.deg / 15
                    self.upscale_responsiveness = False
            return gui.Widget.EventCallbackResult.HANDLED

        if event.key == gui.KeyName.R and event.type == gui.KeyEvent.DOWN:
            self._on_refine()
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key == gui.KeyName.T and event.type == gui.KeyEvent.DOWN:
            self._on_initial_viewpoint()
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key == gui.KeyName.F and event.type == gui.KeyEvent.DOWN:
            self._on_generate()
            return gui.Widget.EventCallbackResult.HANDLED      
        if event.key == gui.KeyName.V and event.type == gui.KeyEvent.DOWN:
            is_open = self._images_panel.get_is_open()
            self._images_panel.set_is_open(not is_open)
            self.window.set_needs_layout()
            return gui.Widget.EventCallbackResult.HANDLED      
        if event.key == gui.KeyName.ONE and event.type == gui.KeyEvent.DOWN:
            if self._responsiveness.double_value >= 2:
                self._responsiveness.double_value -= 1
            else:
                self._responsiveness.double_value = 1
            self.dist = 0.0004 * self._responsiveness.double_value
            self.deg = 0.2 * self._responsiveness.double_value
            return gui.Widget.EventCallbackResult.HANDLED      
        if event.key == gui.KeyName.TWO and event.type == gui.KeyEvent.DOWN:
            if self._responsiveness.double_value <= 19:
                self._responsiveness.double_value += 1
            else:
                self._responsiveness.double_value = 20
            self.dist = 0.0004 * self._responsiveness.double_value
            self.deg = 0.2 * self._responsiveness.double_value
            return gui.Widget.EventCallbackResult.HANDLED      
        # if no active_mesh selected print error
        if self._meshes_used.selected_index == -1:
            self._on_error("라벨링 대상 물체를 선택하세요 (error at _transform)")
            return gui.Widget.EventCallbackResult.HANDLED

        # Translation
        if not self._left_shift_modifier:
            self._log.text = "\t물체 위치를 조정 중 입니다."
            if event.key == gui.KeyName.D:
                self.move( self.dist, 0, 0, 0, 0, 0)
            elif event.key == gui.KeyName.A:
                self.move( -self.dist, 0, 0, 0, 0, 0)
            elif event.key == gui.KeyName.S:
                self.move( 0, self.dist, 0, 0, 0, 0)
            elif event.key == gui.KeyName.W:
                self.move( 0, -self.dist, 0, 0, 0, 0)
            elif event.key == gui.KeyName.Q:
                self.move( 0, 0, self.dist, 0, 0, 0)
            elif event.key == gui.KeyName.E:
                self.move( 0, 0, -self.dist, 0, 0, 0)
        # Rotation - keystrokes are not in same order as translation to make movement more human intuitive
        else:
            self._log.text = "\t물체 방향을 조정 중 입니다."
            if event.key == gui.KeyName.E:
                self.move( 0, 0, 0, 0, 0, self.deg * np.pi / 180)
            elif event.key == gui.KeyName.Q:
                self.move( 0, 0, 0, 0, 0, -self.deg * np.pi / 180)
            elif event.key == gui.KeyName.A:
                self.move( 0, 0, 0, 0, self.deg * np.pi / 180, 0)
            elif event.key == gui.KeyName.D:
                self.move( 0, 0, 0, 0, -self.deg * np.pi / 180, 0)
            elif event.key == gui.KeyName.S:
                self.move( 0, 0, 0, self.deg * np.pi / 180, 0, 0)
            elif event.key == gui.KeyName.W:
                self.move( 0, 0, 0, -self.deg * np.pi / 180, 0, 0)

        return gui.Widget.EventCallbackResult.HANDLED

    def _on_mouse(self, event):
        
        # We could override BUTTON_DOWN without a modifier, but that would
        # interfere with manipulating the scene.
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.ALT):

            try:
                objects = self._annotation_scene.get_objects()
                active_obj = objects[self._meshes_used.selected_index]
            except IndexError:
                self._on_error("라벨링 대상 물체를 선택하세요. (error at _on_mouse)")
                return gui.Widget.EventCallbackResult.HANDLED

            def depth_callback(depth_image):
                # Coordinates are expressed in absolute coordinates of the
                # window, but to dereference the image correctly we need them
                # relative to the origin of the widget. Note that even if the
                # scene widget is the only thing in the window, if a menubar
                # exists it also takes up space in the window (except on macOS).
                x = event.x - self._scene.frame.x
                y = event.y - self._scene.frame.y
                # Note that np.asarray() reverses the axes.
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                    pass
                else:
                    target_xyz = self._scene.scene.camera.unproject(
                        event.x, event.y, depth, self._scene.frame.width,
                        self._scene.frame.height)
                    target_xyz = np.array(target_xyz)
                    # self._annotation_changed = True
                    objects = self._annotation_scene.get_objects()
                    active_obj = objects[self._meshes_used.selected_index]
                    h_transform = np.eye(4)
                    h_transform[:3, 3] = target_xyz - active_obj.obj_geometry.get_center()
                    active_obj.obj_geometry.transform(h_transform)
                    center = active_obj.obj_geometry.get_center()
                    self._scene.scene.remove_geometry(active_obj.obj_name)
                    self._scene.scene.add_geometry(active_obj.obj_name, active_obj.obj_geometry,
                                                self.settings.annotation_active_obj_material,
                                                add_downsampled_copy_for_fast_rendering=True)
                    # update values stored of object
                    active_obj.transform = np.matmul(h_transform, active_obj.transform)
                    if self.settings.show_coord_frame:
                        self._add_coord_frame("obj_coord_frame", size=0.1)
                        self._add_coord_frame("world_coord_frame")
                    if self.settings.show_mesh_names:
                        self._update_and_show_mesh_name()
            self._scene.scene.scene.render_to_depth_image(depth_callback)
            self._log.text = "\t물체 위치를 조정 중 입니다."
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_selection_changed(self, a, b):
        self._log.text = "\t라벨링 대상 물체를 변경합니다."
        objects = self._annotation_scene.get_objects()
        for obj in objects:
            self._scene.scene.remove_geometry(obj.obj_name)
            self._scene.scene.add_geometry(obj.obj_name, obj.obj_geometry,
                                        self.settings.annotation_obj_material,
                                        add_downsampled_copy_for_fast_rendering=True)
        active_obj = objects[self._meshes_used.selected_index]
        self._scene.scene.remove_geometry(active_obj.obj_name)
        self._scene.scene.add_geometry(active_obj.obj_name, active_obj.obj_geometry,
                                    self.settings.annotation_active_obj_material,
                                    add_downsampled_copy_for_fast_rendering=True)
        self.inst_id_edit.set_value(int(active_obj.obj_name.split("_")[-1]))
        self._apply_settings()

    def _on_refine(self):
        self._log.text = "\t자동 정렬 중입니다."
        self._annotation_changed = True

        # if no active_mesh selected print error
        if self._meshes_used.selected_index == -1:
            self._on_error("라벨링 대상 물체를 선택하세요. (error at _on_refine)")
            return gui.Widget.EventCallbackResult.HANDLED

        target = self._annotation_scene.annotation_scene
        objects = self._annotation_scene.get_objects()
        active_obj = objects[self._meshes_used.selected_index]
        source = active_obj.obj_geometry

        trans_init = np.identity(4)
        threshold = 0.004
        radius = 0.002
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        reg = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
                                                          o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                          o3d.pipelines.registration.ICPConvergenceCriteria(
                                                              max_iteration=50))
        if np.sum(np.abs(reg.transformation[:3, 3])) < 0.2:
            active_obj.obj_geometry.transform(reg.transformation)
            self._scene.scene.remove_geometry(active_obj.obj_name)
            self._scene.scene.add_geometry(active_obj.obj_name, active_obj.obj_geometry,
                                        self.settings.annotation_active_obj_material,
                                        add_downsampled_copy_for_fast_rendering=True)
            active_obj.transform = np.matmul(reg.transformation, active_obj.transform)

            if self.settings.show_coord_frame:
                self._add_coord_frame("obj_coord_frame", size=0.1)
                self._add_coord_frame("world_coord_frame")
            if self.settings.show_mesh_names:
                self._update_and_show_mesh_name()
            self._log.text = "\t자동 정렬을 완료했습니다."
        else:
            self._log.text = "\t자동 정렬에 실패했습니다. 물체 위치를 조정한 후 다시 시도하세요."

    def _on_generate(self):
        self._log.text = "\t라벨링 결과를 저장 중입니다."
        if self._annotation_scene is None: # shsh
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_generate)")
            return

        image_num = self._annotation_scene.image_num
        model_names = self.load_model_names()
        json_6d_path = os.path.join(self.scenes.scenes_path, f"{self._annotation_scene.scene_num:06}", "scene_gt.json")

        if os.path.exists(json_6d_path):
            with open(json_6d_path, "r") as gt_scene:
                try:
                    gt_6d_pose_data = json.load(gt_scene)
                except json.decoder.JSONDecodeError as e:
                    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = json_6d_path.replace(".json", "_backup_{}.json".fomrat(date_time))
                    shutil.copy(json_6d_path, backup_path)
                    print(e)
                    gt_6d_pose_data = {}
        else:
            gt_6d_pose_data = {}

        # wrtie/update "scene_gt.json"
        try:
            with open(json_6d_path, 'w+') as gt_scene:
                view_angle_data = list()
                for obj in self._annotation_scene.get_objects():
                    transform_cam_to_object = obj.transform
                    translation = list(transform_cam_to_object[0:3, 3] * 1000)  # convert meter to mm
                    model_names = self.load_model_names()
                    obj_id = int(obj.obj_name.split("_")[1])  # assuming object name is formatted as obj_000001
                    inst_id = int(obj.obj_name.split("_")[2])
                    obj_data = {
                        "cam_R_m2c": transform_cam_to_object[0:3, 0:3].tolist(),  # rotation matrix
                        "cam_t_m2c": translation,  # translation
                        "obj_id": obj_id,
                        "inst_id": inst_id
                    }
                    view_angle_data.append(obj_data)
                gt_6d_pose_data[str(image_num)] = view_angle_data
                json.dump(gt_6d_pose_data, gt_scene)
            self._log.text = "\t라벨링 결과를 저장했습니다."
        except Exception as e:
            date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            json_6d_path = os.path.join(self.scenes.scenes_path, f"{self._annotation_scene.scene_num:06}", "scene_gt_backup_{}.json".format(date_time))
            with open(json_6d_path, 'w+') as gt_scene:
                view_angle_data = list()
                for obj in self._annotation_scene.get_objects():
                    transform_cam_to_object = obj.transform
                    translation = list(transform_cam_to_object[0:3, 3] * 1000)  # convert meter to mm
                    model_names = self.load_model_names()
                    obj_id = int(obj.obj_name.split("_")[1])  # assuming object name is formatted as obj_000001
                    inst_id = int(obj.obj_name.split("_")[2])
                    obj_data = {
                        "cam_R_m2c": transform_cam_to_object[0:3, 0:3].tolist(),  # rotation matrix
                        "cam_t_m2c": translation,  # translation
                        "obj_id": obj_id,
                        "inst_id": inst_id
                    }
                    view_angle_data.append(obj_data)
                gt_6d_pose_data[str(image_num)] = view_angle_data
                json.dump(gt_6d_pose_data, gt_scene)
            self._log.text = "\t라벨링 결과를 저장했습니다."
        self._annotation_changed = False

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

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_show_coord_frame(self, show):
        self.settings.show_coord_frame = show
        self._apply_settings()

    def _on_show_mesh_names(self, show):
        self.settings.show_mesh_names = show
        self._apply_settings()

    def _on_highlight_obj(self, light):
        self.settings.highlight_obj = light
        if light:
            self._log.text = "\t 라벨링 대상 물체를 강조합니다."
            self.settings.annotation_obj_material.base_color = [0.9, 0.3, 0.3, 1.0]
            self.settings.annotation_active_obj_material.base_color = [0.3, 0.9, 0.3, 1.0]
        elif not light:
            self._log.text = "\t 라벨링 대상 물체를 강조하지 않습니다."
            self.settings.annotation_obj_material.base_color = [0.9, 0.9, 0.9, 1.0]
            self.settings.annotation_active_obj_material.base_color = [0.9, 0.9, 0.9, 1.0]

        self._apply_settings()

        # update current object visualization
        if self._annotation_scene is None: # shsh
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_highlight_obj)")
            return
        meshes = self._annotation_scene.get_objects()
        for mesh in meshes:
            self._scene.scene.modify_geometry_material(mesh.obj_name, self.settings.annotation_obj_material)
        active_obj = meshes[self._meshes_used.selected_index]
        self._scene.scene.modify_geometry_material(active_obj.obj_name, self.settings.annotation_active_obj_material)


    def _on_transparency(self, transparency): #shsh
        
        self._log.text = "\t 투명도 값을 변경합니다."
        self.settings.transparency = transparency
        if self._annotation_scene is None:
            return
        self.settings.annotation_obj_material.base_color = [0.9, 0.3 + 0.6*transparency, 0.3 + 0.6*transparency, 1]
        self.settings.annotation_active_obj_material.base_color = [0.3 + 0.6*transparency, 0.9, 0.3 + 0.6*transparency, 1]

        objects = self._annotation_scene.get_objects()
        for obj in objects:
            self._scene.scene.remove_geometry(obj.obj_name)
            self._scene.scene.add_geometry(obj.obj_name, obj.obj_geometry,
                                            self.settings.annotation_obj_material,
                                            add_downsampled_copy_for_fast_rendering=True)

        active_obj = objects[self._meshes_used.selected_index]
        self._scene.scene.remove_geometry(active_obj.obj_name)
        self._scene.scene.add_geometry(active_obj.obj_name, active_obj.obj_geometry,
                                        self.settings.annotation_active_obj_material,
                                        add_downsampled_copy_for_fast_rendering=True)                       
        self._apply_settings()


    def _on_point_size(self, size):
        self.settings.scene_material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_responsiveness(self, responsiveness):
        self.dist = 0.0004 * responsiveness
        self.deg = 0.2 * responsiveness

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_about(self):
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")
        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("6D Object Pose Annotator by GIST AILAB.\nCopyright (c) 2022 Seunghyeok Back\nGwangju Institute of Science and Technology (GIST)\nshback@gm.gist.ac.kr"))
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

    def _on_about_ok(self):
        self.window.close_dialog()

    def _obj_instance_count(self, mesh_to_add, meshes):
        types = [i[:-2] for i in meshes]  # remove last 3 character as they present instance number (OBJ_INSTANCE)
        equal_values = [i for i in range(len(types)) if types[i] == mesh_to_add]
        count = 0
        if len(equal_values):
            indices = np.array(meshes)
            indices = indices[equal_values]
            indices = [int(x[-1]) for x in indices]
            count = max(indices) + 1
            # TODO change to fill the numbers missing in sequence
        return count

    def _add_mesh(self):
        if self._annotation_scene is None: # shsh
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _add_mesh)")
            return
        if len(self._meshes_available.selected_value) == 0:
            self._on_error("라벨링 물체를 선택하세요. (error at _add_mesh)")
            return   

        self._log.text = "\t 라벨링 물체를 추가합니다."
        meshes = self._annotation_scene.get_objects()
        meshes = [i.obj_name for i in meshes]
        object_geometry = o3d.io.read_point_cloud(
            os.path.join(self.scenes.objects_path,  self._meshes_available.selected_value + '.ply'))
        object_geometry.points = o3d.utility.Vector3dVector(
            np.array(object_geometry.points) / 1000)  # convert mm to meter
        init_trans = np.identity(4)
        center = self._annotation_scene.annotation_scene.get_center()
        center[2] -= 0.2
        init_trans[0:3, 3] = center
        object_geometry.transform(init_trans)
        new_mesh_instance = self._obj_instance_count(self._meshes_available.selected_value, meshes)
        new_mesh_name = str(self._meshes_available.selected_value) + '_' + str(new_mesh_instance)
        self._scene.scene.add_geometry(new_mesh_name, object_geometry, self.settings.annotation_obj_material,
                                       add_downsampled_copy_for_fast_rendering=True)
        self._annotation_scene.add_obj(object_geometry, new_mesh_name, new_mesh_instance, transform=init_trans)
        if self.settings.show_mesh_names:
            self.mesh_names.append(self._scene.add_3d_label(center, f"{new_mesh_name}"))

        meshes = self._annotation_scene.get_objects()  # update list after adding current object
        meshes = [i.obj_name for i in meshes]
        self._meshes_used.set_items(meshes)
        self._meshes_used.selected_index = len(meshes) - 1

    def _remove_mesh(self):
        if self._annotation_scene is None: # shsh
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _remove_mesh)")
            return
        if not self._annotation_scene.get_objects():
            self._on_error("라벨링 대상 물체를 선택하세요. (error at _remove_mesh)")
            return
        self._log.text = "\t 라벨링 물체를 삭제합니다."
        meshes = self._annotation_scene.get_objects()
        active_obj = meshes[self._meshes_used.selected_index]
        self._scene.scene.remove_geometry(active_obj.obj_name)  # remove mesh from scene
        self._annotation_scene.remove_obj(self._meshes_used.selected_index)  # remove mesh from class list
        # update list after adding removing object
        meshes = self._annotation_scene.get_objects()  # get new list after deletion
        meshes = [i.obj_name for i in meshes]
        self._meshes_used.set_items(meshes)
        if self.settings.show_mesh_names:
            self._update_and_show_mesh_name()

    def _make_point_cloud(self, rgb_img, depth_img, cam_K):
        # convert images to open3d types
        rgb_img_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        depth_img_o3d = o3d.geometry.Image(depth_img)

        # convert image to point cloud
        intrinsic = o3d.camera.PinholeCameraIntrinsic(rgb_img.shape[0], rgb_img.shape[1],
                                                      cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img_o3d, depth_img_o3d,
                                                                  depth_scale=1, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        return pcd

    def scene_load(self, scenes_path, scene_num, image_num):

        self._annotation_changed = False

        self._scene.scene.clear_geometry()
        geometry = None

        scene_path = os.path.join(scenes_path, f'{scene_num:06}')

        camera_params_path = os.path.join(scene_path, 'scene_camera.json')
        with open(camera_params_path) as f:
            data = json.load(f)
            cam_K = data[str(image_num)]['cam_K']
            self.cam_K = np.array(cam_K).reshape((3, 3))
            depth_scale = data[str(image_num)]['depth_scale']

        self.rgb_path = os.path.join(scene_path, 'rgb', f'{image_num:06}' + '.png')
        rgb_img = cv2.imread(self.rgb_path)
        self.depth_path = os.path.join(scene_path, 'depth', f'{image_num:06}' + '.png')
        depth_img = cv2.imread(self.depth_path, -1)
        depth_img = np.float32(depth_img) / depth_scale / 1000
        H, W, _ = rgb_img.shape
        ratio = 640 / W
        _rgb_img = cv2.resize(rgb_img.copy(), (640, int(H*ratio)))
        _rgb_img = o3d.geometry.Image(cv2.cvtColor(_rgb_img, cv2.COLOR_BGR2RGB))
        self._rgb_proxy.set_widget(gui.ImageWidget(_rgb_img))

        try:
            geometry = self._make_point_cloud(rgb_img, depth_img, self.cam_K)
        except Exception as e:
            print(e)
            print("Failed to load scene.")

        if geometry is not None:
            print("[Info] Successfully read scene ", scene_num)
            geometry = geometry.voxel_down_sample(0.002)
            if not geometry.has_normals():
                geometry.estimate_normals()
            geometry.normalize_normals()
        else:
            print("[WARNING] Failed to read points")
        self._scene.scene.add_geometry("annotation_scene", geometry, self.settings.scene_material,
                                        add_downsampled_copy_for_fast_rendering=True)
        self.bounds = geometry.get_axis_aligned_bounding_box()
        self._on_initial_viewpoint()

        self._annotation_scene = AnnotationScene(geometry, scene_num, image_num)
        self._meshes_used.set_items([])  # clear list from last loaded scene

        # load values if an annotation already exists
        scene_gt_path = os.path.join(self.scenes.scenes_path, f"{self._annotation_scene.scene_num:06}",
                                        'scene_gt.json')
        if os.path.exists(scene_gt_path):
            with open(scene_gt_path) as scene_gt_file:
                try:
                    data = json.load(scene_gt_file)
                except json.decoder.JSONDecodeError:
                    self._on_error("저장된 라벨링 파일을 불러오지 못했습니다. (error at _scene_load)")
                    return
                if str(image_num) in data.keys():
                    scene_data = data[str(image_num)]
                    active_meshes = list()
                    for i, obj in enumerate(scene_data):
                        # add object to annotation_scene object
                        obj_geometry = o3d.io.read_point_cloud(
                            os.path.join(self.scenes.objects_path, 'obj_' + f"{int(obj['obj_id']):06}" + '.ply'))
                        obj_geometry.points = o3d.utility.Vector3dVector(
                            np.array(obj_geometry.points) / 1000)  # convert mm to meter
                        model_name = 'obj_' + f'{ + obj["obj_id"]:06}'
                        if "inst_id" in obj.keys():
                            obj_instance = int(obj["inst_id"])
                        else:
                            obj_instance = self._obj_instance_count(model_name, active_meshes)
                        obj_name = model_name + '_' + str(obj_instance)
                        translation = np.array(np.array(obj['cam_t_m2c']), dtype=np.float64) / 1000  # convert to meter
                        orientation = np.array(np.array(obj['cam_R_m2c']), dtype=np.float64)
                        transform = np.concatenate((orientation.reshape((3, 3)), translation.reshape(3, 1)), axis=1)
                        transform_cam_to_obj = np.concatenate(
                            (transform, np.array([0, 0, 0, 1]).reshape(1, 4)))  # homogeneous transform

                        self._annotation_scene.add_obj(obj_geometry, obj_name, obj_instance, transform_cam_to_obj)
                        # adding object to the scene
                        obj_geometry.translate(transform_cam_to_obj[0:3, 3])
                        center = obj_geometry.get_center()
                        obj_geometry.rotate(transform_cam_to_obj[0:3, 0:3], center=center)
                        if i == 0:
                            self._scene.scene.add_geometry(obj_name, obj_geometry, self.settings.annotation_active_obj_material,
                                                            add_downsampled_copy_for_fast_rendering=True)
                        else:
                            self._scene.scene.add_geometry(obj_name, obj_geometry, self.settings.annotation_obj_material,
                                                            add_downsampled_copy_for_fast_rendering=True)
                        active_meshes.append(obj_name)
                    self._meshes_used.set_items(active_meshes)
        self._update_scene_numbers()

        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def update_obj_list(self):
        model_names = self.load_model_names()
        self._meshes_available.set_items(model_names)

    def load_model_names(self):
        self.obj_ids = sorted([int(os.path.basename(x)[5:-4]) for x in glob.glob(self.scenes.objects_path + self.spl + '*.ply')])
        model_names = ['obj_' + f'{ + obj_id:06}' for obj_id in self.obj_ids]
        return model_names

    def _on_initial_viewpoint(self):
        if self.bounds is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_initial_viewpoint)")
            return
        self._log.text = "\t 처음 시점으로 이동합니다."
        self._scene.setup_camera(60, self.bounds, self.bounds.get_center())
        center = np.array([0, 0, 0])
        eye = center + np.array([0, 0, -0.5])
        up = np.array([0, -1, 0])
        self._scene.look_at(center, eye, up)
        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _check_changes(self):
        if self._annotation_changed:
            self._on_error("라벨링 결과를 저장하지 않았습니다. 저장하지 않고 넘어가려면 버튼을 다시 눌러주세요.")
            self._annotation_changed = False
            return True
        else:
            return False

    def _on_next_scene(self):
        if self._check_changes():
            return
        if self.current_scene_idx is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_next_scene)")
            return
        if self.current_scene_idx >= len(self.scene_num_lists) - 1:
            self._on_error("다음 라벨링 폴더가 존재하지 않습니다.")
            return
        self._log.text = "\t 다음 라벨링 폴더로 이동했습니다."
        self.current_scene_idx += 1
        self.scene_load(self.scenes.scenes_path, self.current_scene_idx, 0)  # open next scene on the first image

    def _on_previous_scene(self):
        if self._check_changes():
            return
        if self.current_scene_idx is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_previous_scene)")
            return
        if self.current_scene_idx <= 0:
            self._on_error("이전 라벨링 폴더가 존재하지 않습니다.")
            return
        self.current_scene_idx -= 1
        self._log.text = "\t 이전 라벨링 폴더로 이동했습니다."
        self.scene_load(self.scenes.scenes_path, self.current_scene_idx, 0)  # open next scene on the first image

    def _on_next_image(self):
        if self._check_changes():
            return
        if self.current_image_idx is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_next_image)")
            return
        if self.current_image_idx  >= len(self.image_num_lists) - 1:
            self._on_error("다음 이미지가 존재하지 않습니다.")
            return
        self._log.text = "\t 다음 이미지로 이동했습니다."
        self.current_image_idx += 1
        self.scene_load(self.scenes.scenes_path, self._annotation_scene.scene_num, self.image_num_lists[self.current_image_idx])
        self._progress.value = (self.current_image_idx + 1) / len(self.image_num_lists) # 25% complete
        self._progress_str.text = "진행률: {:.1f}% [{}/{}]".format(
            100 * (self.current_image_idx + 1) / len(self.image_num_lists), 
            self.current_image_idx + 1, len(self.image_num_lists))

    def _on_previous_image(self):
        if self._check_changes():
            return
        if self.current_image_idx is None:
            self._on_error("라벨링 대상 파일을 선택하세요. (error at _on_previous_image)")
            return
        if self.current_image_idx < 0:
            self._on_error("이전 이미지가 존재하지 않습니다.")
            return
        self._log.text = "\t 이전 이미지로 이동했습니다."
        self.current_image_idx -= 1
        self.scene_load(self.scenes.scenes_path, self._annotation_scene.scene_num, self.image_num_lists[self.current_image_idx])
        self._progress.value = (self.current_image_idx + 1) / len(self.image_num_lists) # 25% complete
        self._progress_str.text = "진행률: {:.1f}% [{}/{}]".format(
            100 * (self.current_image_idx + 1) / len(self.image_num_lists), 
            self.current_image_idx + 1, len(self.image_num_lists))

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
