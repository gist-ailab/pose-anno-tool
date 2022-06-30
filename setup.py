import sys
from cx_Freeze import setup, Executable
import scipy
import os

includefiles_list=[]
scipy_path = os.path.dirname(scipy.__file__)
includefiles_list.append(scipy_path)

# Dependencies are automatically detected, but it might need fine tuning.
# "packages": ["os"] is used as example only
build_exe_options = {"includes": ["OpenGL.platform.win32", "matplotlib", "os", "numpy", "numpy.core._methods", "matplotlib", "random", "tkinter","PyQt5.QtCore","PyQt5.QtGui", "PyQt5.QtWidgets","ctypes", "pyrender", "PyQt5", "glfw", "OpenGL", "pyglet"], 
                     "packages": ["OpenGL.platform.win32", "matplotlib", "os", "numpy", "numpy.core._methods", "matplotlib", "random", "tkinter","PyQt5.QtCore","PyQt5.QtGui", "PyQt5.QtWidgets","ctypes", "pyrender", "PyQt5", "glfw", "pyglet"], 
                     "include_files": [r'C:\\Users\\user\\Anaconda3\\Library\\plugins\\platforms'],
                     "excludes": []}

os.environ['TCL_LIBRARY'] = r'C:\\Users\\user\\Anaconda3\\envs\\pose-anno\\tcl\\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\\Users\\user\\Anaconda3\\envs\\pose-anno\\tcl\\tcl8.6'

# base="Win32GUI" should be used only for Windows GUI app
base = None
# if sys.platform == "win32":
#     base = "Win32GUI"


setup(
    name="gist-ailab-pose-annotator",
    version="0.1",
    description="GIST AILAB 6D Object Pose Annotation Tool",
    author="Seunghyeok Back, GIST AILAB, shback@gm.gist.ac.kr",
    options={"build_exe": build_exe_options},
    executables=[Executable("annotator.py", 
                            copyright="MIT License, Seunghyeok Back, GIST AILAB",
                            icon="./assets/icon.ico",
                            base=base)],
)