# Hand Pose Annotator

<img src="./lib/hand_pose_annotator.png" height="400">


# TODO:
- [ ] Save Debug log
- [ ] Labeled Object Mesh Model
- [ ] Image View with Current Label

## Install and Run

In windows, Microsoft Visual C++ 14.0 is required. [link](https://www.microsoft.com/ko-KR/download/details.aspx?id=48159) [link](https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/)

```
# install requirements
$ conda create -n pose-anno python=3.7
$ conda activate pose-anno
$ pip install -r requirements.txt
$ pip install numpy matplotlib==2.2.5 glumpy pyrender imgviz pyglet
$ pip install open3d==0.15.2
$ pip install opencv-python==4.5.3.56

# install torch
$ (linux) pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
$ (window) pip3 install torch torchvision torchaudio

# install mano
$ pip install git+https://github.com/hassony2/chumpy.git
$ pip install git+https://github.com/hassony2/manopth


$ python hand_pose_annotator.py
$ pip install --upgrade cx_Freeze
$ python setup.py build
```

## Keyboard Action
```
# Convert Labeling Hand
TAB: convert hand

# Labeling stage change
F1: Change Labeling Stage to "Root Translation and Rotation"
F2: Change Labeling Stage to "Hand Tip Translation"
F3: Change Labeling Stage to "Hand Detail Translation"


# Translation(If Shift pressed -> Rotation)
Q: +Z
W: +Y
E: -Z
A: -X
S: -Y
D: +X

# Reset
R: Reset hand model pose
HOME: Reset Target(Guide) Pose to Current Hand model Pose

# Control Joint Change (Labeling Stage 2 and 3)
1: 엄지
2: 검지
3: 중지
4: 약지
5: 소지
# for stage 3 
Page Up
Page Down

```




