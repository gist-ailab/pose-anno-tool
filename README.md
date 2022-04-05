# 6DPoseAnnotator

An interactive 6 degree-of-freedom pose annotation tool using point cloud processings.

<img src="./data/6DoFAnnotation.gif" width="320px">

## Requirements
- [open3d==0.11.1](http://www.open3d.org/)
- [opencv-python==4.4.0.44](https://opencv.org/)

## 6D pose annotation with mouse and keyboard commands

Type:
```
$ python 6DoFPoseAnnotator.py
```

You can use following commands: 

- Left click - Translation to the mouse pointer
- "1" - Rotation around roll axis.
- "2" - Rotation around pitch axis.
- "3" - Rotation around yaw axis.
- "i" - Pose refinement by ICP algorithm (Coarse mode).
- "f" - Pose refinement by ICP algorithm (Fine mode).
- "q" - Quit

![2DView](./data/screenshot_2d_view.png)

When you type "q", a final transformation matrix, "trans.json", and a transformed point cloud, "cloud_rot.ply", are saved.

### Starting from specific initial transformation
By using option "--init", you can choose initial transformation matrix to be apply.

Try:
```
$ python 6dpose_annotator.py --init data/trans.json
```

