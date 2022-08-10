# Hand Pose Annotator

<img src="./lib/hand_pose_annotator.png" height="400">


# TODO:
- [X] Move Each Hand by ALT + Click
- [X] Move Each Hand by Keyboard
- [X] Optimze Hand Pose by End-Tips
- [X] Optimze Hand Pose by Detail fingers

- [ ] Edit log
- [ ] CTRL+Z -> control joint back



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
1: Change Labeling Stage to "Root Translation and Rotation"
2: Change Labeling Stage to "Hand Tip Translation"
3: Change Labeling Stage to "Hand Detail Translation"


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
Z: 엄지
X: 검지
C: 중지
V: 약지
B: 소지
# for stage 3 
U - I - O - P (root to tip)

```




