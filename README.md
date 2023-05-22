
[toc]
- [数据集](#数据集)
  - [The YOLOv8 Instance Segmentation Label Format](#the-yolov8-instance-segmentation-label-format)
  - [The YOLOv8 Instance Segmentation Dataset File Format](#the-yolov8-instance-segmentation-dataset-file-format)
- [推理时间](#推理时间)



使用YOLO进行建筑物实例分割

> 选择YOLO而不是Mask-RCNN的原因是：  
> - YOLO帧率高
> - CubeSLAM也使用YOLO

# 数据集

## [The YOLOv8 Instance Segmentation Label Format](https://docs.ultralytics.com/datasets/segment/)

1、每个图片对应一个同名的txt文件，即 `name.png/jpg + name.txt` ；  

2、在txt文件中，每一行表示一个单独的物体；  

3、每一行包括类别和遮罩：

```yaml
<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

- 第一个数字表示类别；  
- 后面的数字表示物体遮罩多边形的每个顶点的XY坐标；  
  > 在YOLO中，X坐标从左到右，Y坐标从上到下，原点为图片左上角。  
  
  > 所有的坐标都使用归一化坐标：Y相对于图片高度缩放到0~1，X相对于图片宽度缩放到0~1。

## [The YOLOv8 Instance Segmentation Dataset File Format](https://docs.ultralytics.com/datasets/segment/)

```yaml
train: <path-to-training-images>
val: <path-to-validation-images>

nc: <number-of-classes>
names: [ <class-1>, <class-2>, ..., <class-n> ]
# name 也可以定义为如下格式：
# names:
#  0: person
#  1: bicycle
```
---
https://docs.ultralytics.com/datasets/segment/coco/#dataset-structure   
coco.yaml
```yaml
# Ultralytics YOLO 🚀, AGPL-3.0 license
# COCO 2017 dataset http://cocodataset.org by Microsoft
# Example usage: yolo train data=coco.yaml
# parent
# ├── ultralytics
# └── datasets
#     └── coco  ← downloads here (20.1 GB)


# Train/val/test sets as 
# 1) dir: path/to/imgs, 
# 2) file: path/to/imgs.txt, or 
# 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco  # dataset root dir
train: train2017.txt  # train images (relative to 'path') 118287 images
val: val2017.txt  # val images (relative to 'path') 5000 images
test: test-dev2017.txt  # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794

# Classes
names:
  0: person
  1: bicycle
  2: car
# ...
```
---
https://docs.ultralytics.com/datasets/segment/coco8-seg/#introduction  
coco8-seg.yaml

```yaml
# Ultralytics YOLO 🚀, AGPL-3.0 license
# COCO8-seg dataset (first 8 images from COCO train2017) by Ultralytics
# Example usage: yolo train data=coco8-seg.yaml
# parent
# ├── ultralytics
# └── datasets
#     └── coco8-seg  ← downloads here (1 MB)


# Train/val/test sets as 
# 1) dir: path/to/imgs, 
# 2) file: path/to/imgs.txt, or 
# 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco8-seg  # dataset root dir
train: images/train  # train images (relative to 'path') 4 images
val: images/val  # val images (relative to 'path') 4 images
test:  # test images (optional)

# Classes
names:
  0: person
  1: bicycle
  2: car
# ...
```


# 推理时间

YoloV8使用`YOLOv8n-seg`模型需要100ms CPU时间；  
YoloV5使用`YOLOv5n-seg`模型需要65ms CPU时间。
