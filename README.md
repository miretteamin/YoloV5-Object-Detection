# **YoloV5 Object Detection**
## **Introduction**
This project represents the training of Yolov5 on COCO 2017 dataset once and on custom dataset.<br>
YOLO is an acronym for 'You only look once', it’s an object detection model, it provides images in a grid. It has 5 versions, in this report we used version 5 nano.

## **Dataset's Brief**
**COCO Dataset:**<br>
*Link:* https://drive.google.com/drive/folders/1-0AuOT57KFrB1xUkfqNKl973BIX552RJ <br>
We trained Yolov5n model on COCO dataset, which is dedicated for object detection segmentation, and captioning and published by Microsoft.
Coco stands for common Objects in Context of challenging, high quality images and here are some of its features:
- 330k labeled images
- 90 categories
- 5 captions per image<br>

We chose only 20 categories for our train: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow']<br>
And 3k images for training and 1k for validation.<br><br>
**Custom Dataset:**<br>
*Link:* https://drive.google.com/drive/folders/1gVhHKfMo81BqJ910Pqa9ayPNOmR8qETX <br>
We collected 110 images for 5 categories and labeled them in yolo format. The categories we chose were: bags, cups, mobiles, lamps and faces.

## **Yolo Architecture** 
### **Layers**
Yolov5 nano has 270 layers and 1790977 parameters.
1st element in each array represents where to get the number of input of channels -1 means the output of the last layer<br>
2nd element is the number of repeats for this block (it gets scaled based on yolov5 size)<br>
3rd element is the block type<br>
4th element is the args list for this block<br>
Conv → is standard convolution block with batch normalization and SiLU activation<br>
### **Activation Functions**
Repo authors decided to use SiLu in hidden layers and Sigmoid in final detection layer.
### **Optimization Function**
SGD (default optimizer, Model gives the option to use Adam but we didn’t use it)
### **Loss Function**
Binary Cross-Entropy with Logits Loss (used from PyTorch for loss calculation of class probability and object score) and Bounding Box Loss is 1-IOU Function
### **Grid Size**
Yolov5 use multiscale outputs stride 8, 16, 32 aka gives 3 different grid size for each image so it can differentiate big and small objects in any size (as Yolo doesn’t need to take one fixed size of image).<br>
Example: Input image size is 640x640 then it will generate 3 grids (640/8, 640/16 and 640/32) = 80x80, 40x40 and 20x20
### **Number of anchors**
3 anchors per layers and 3 different values for each grid.

## **Some Notes**
- We modified detect.py in the cloned repo, so we added the modified detect.py in the folder.
- Yaml files contain paths for images in the datasets uploaded on the drive we gave its link before.
