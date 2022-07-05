# Object Pose Annotator

<img src="./lib/object_pose_annotator.png" height="400">


# TODO:
- [X] Show object coordinate
- [X] Add progressive bar
- [X] Add smart view controller
- [X] Move to initial viewpoint
- [X] Add transparency
- [X] Show original image
- [X] Fix camera viewpoint
- [X] Show mask image
- [X] Add score visualizer
- [X] Add world coordinate
- [X] Show object model images
- [X] Visualize the key direction
- [X] Add instance label
- [X] Add logger
- [X] Test on window
- [X] Cx_freeze
- [ ] Automatic save and backup


## Install and Run

In windows, Microsoft Visual C++ 14.0 is required. [link](https://www.microsoft.com/ko-KR/download/details.aspx?id=48159) [link](https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/)

```
$ conda create -n pose-anno python=3.7
$ conda activate pose-anno
$ pip install -r requirements.txt
$ pip install numpy matplotlib==2.2.5 open3d glumpy pyrender imgviz pyglet
$ pip install opencv-python==4.5.3.56
$ pip install git+https://github.com/SeungBack/bop_toolkit

$ python annotator.py

$ pip install --upgrade cx_Freeze

$ python setup.py build
```

