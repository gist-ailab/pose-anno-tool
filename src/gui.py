import dearpygui.dearpygui as dpg

dpg.create_context()

width, height, channels, data = dpg.load_image("/home/seung/Workspace/custom/6DPoseAnnotator/tmp/obj_list.png")

with dpg.texture_registry(show=True):
    dpg.add_static_texture(width, height, data, tag="texture_tag")

def button_callback(sender, app_data):
    print(f"sender is: {sender}")
    print(f"app_data is: {app_data}")

with dpg.window(label="Tutorial"):
    dpg.add_image("texture_tag")
    dpg.add_button(label="1", callback=button_callback)


dpg.create_viewport(title='Custom Title', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()