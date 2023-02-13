
import sys
from cx_Freeze import setup, Executable
import scipy
import os

includefiles_list=[]
scipy_path = os.path.dirname(scipy.__file__)
includefiles_list.append(scipy_path)

# Dependencies are automatically detected, but it might need fine tuning.
# "packages": ["os"] is used as example only
build_exe_options = {"includes": ["os", "numpy", "numpy.core._methods", "chumpy", "random", "tkinter", "ctypes", "torch", "open3d"
                                    ], 
                     "packages": ["os", "numpy", "numpy.core._methods", "chumpy", "random", "tkinter", "ctypes", "torch", "open3d"
                                    ], 
                    #  "include_files": [r'C:\\Users\\user\\minionda3\\Library\\plugins\\platforms'],
                     "excludes": []}

base = None

setup(
    name="gist-ailab-pose-annotator",
    version="0.1",
    description="GIST AILAB 3D Hand Pose Annotation Tool",
    author="Raeyoung Kang, GIST AILAB, raeyo@gm.gist.ac.kr",
    options={"build_exe": build_exe_options},
    executables=[Executable("hand_pose_annotator.py", 
                            copyright="MIT License, Raeyoung Kang, GIST AILAB",
                            icon="./lib/icon.ico",
                            base=base)],
)