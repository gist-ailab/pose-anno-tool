import sys
from cx_Freeze import setup, Executable
import scipy
import os

includefiles_list=[]
scipy_path = os.path.dirname(scipy.__file__)
includefiles_list.append(scipy_path)

# Dependencies are automatically detected, but it might need fine tuning.
# "packages": ["os"] is used as example only
build_exe_options = {"includes": ["os", "numpy", "numpy.core._methods", "random", "tkinter", "ctypes"], 
                     "packages": ["os", "numpy", "numpy.core._methods", "random", "tkinter", "ctypes"], 
                     "include_files": [r'C:\\Users\\user\\Anaconda3\\Library\\plugins\\platforms'],
                     "excludes": []}

os.environ['TCL_LIBRARY'] = r'C:\\Users\\user\\Anaconda3\\envs\\pose-anno\\tcl\\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\\Users\\user\\Anaconda3\\envs\\pose-anno\\tcl\\tcl8.6'

base = None

setup(
    name="gist-ailab-object-viewer",
    version="0.1",
    description="GIST AILAB Object Viewer",
    author="Seunghyeok Back, GIST AILAB, shback@gm.gist.ac.kr",
    options={"build_exe": build_exe_options},
    executables=[Executable("object_viewer.py", 
                            copyright="MIT License, Seunghyeok Back, GIST AILAB",
                            icon="./lib/icon.ico",
                            base=base)],
)