# 6DPoseAnnotator

An interactive 6 degree-of-freedom pose annotation tool using point cloud processings.

<!-- <img src="./example.png" width="5000px"> -->

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
- [ ] Fix key error
- [ ] Automatic save and backup
- [ ] Add logger
- [ ] Add shortcut helper
- [ ] Keyboard shortcuts
- [X] Test on window
- [ ] Pyinstaller


## Install and Run

In windows, Microsoft Visual C++ 14.0 is required. [link](https://www.microsoft.com/ko-KR/download/details.aspx?id=48159)

```
$ conda create -n pose-anno python=3.7
$ conda activate pose-anno
$ pip install matplotlib==2.2.5 opencv-python open3d cython pyrender imgviz pypng triangle glumpy==1.1.0 PyOpenGL glfw scipy==1.5.1 kiwisolver==1.3.1
$ pip install git+https://github.com/thodan/bop_toolkit.git
$ python annotator.py

$ pip install pyinstaller
$ pyinstall -F -w annotator.py
```

