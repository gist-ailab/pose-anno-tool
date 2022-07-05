# Object Viewer

<img src="./lib/object_viewer.png" height="400">


# TODO:
- [ ] Adjust far plane 
- [ ] Support texture.png

## Install and Run

In windows, Microsoft Visual C++ 14.0 is required. [link](https://www.microsoft.com/ko-KR/download/details.aspx?id=48159) [link](https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/)

```
$ conda create -n pose-anno python=3.7
$ conda activate pose-anno
$ pip install -r requirements.txt
$ pip install numpy matplotlib==2.2.5 open3d glumpy pyrender imgviz pyglet
$ pip install opencv-python==4.5.3.56
$ pip install git+https://github.com/SeungBack/bop_toolkit

$ python object_viewer.py

$ python setup.py build
```

