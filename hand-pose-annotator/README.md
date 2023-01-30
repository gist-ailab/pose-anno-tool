# Hand Pose Annotator

<img src="./hand_pose_annotator.png" height="400">

### How to Use
```
# install requirements
$ conda create -n pose-anno-3d python=3.7
$ conda activate pose-anno-3d
$ pip install numpy open3d==0.15.2 PyYAML opencv-python==4.5.3.56 Cython

# install torch - cpu
$ (linux) pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu
$ (window) pip3 install torch

# install mano
$ pip install git+https://github.com/hassony2/chumpy.git
$ pip install git+https://github.com/hassony2/manopth

# run
$ python hand_pose_annotator.py

# build
$ pip install --upgrade cx_Freeze
$ python setup.py build

```
