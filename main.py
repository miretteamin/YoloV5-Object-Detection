# %pip install  --upgrade PyYaml

## Renaming Files

# import os
# import glob
# i = 100
# for path in glob.glob("/content/drive/MyDrive/newCoco/CustomDataset"):
#     for img_path in glob.glob(os.path.join(path,"*.txt")):
#         i+=1
#         nb = img_path[-6:-4]
#         nb = (int(nb) - 20)+100
#         new_name = "image"+ str(nb)+".txt"
#         os.rename(img_path, os.path.join(path, new_name))
#         print(img_path)
#     break


## Modifying Text files

# for i in range(100):
#   with open(f'/content/drive/MyDrive/newCoco/CustomDataset/image{i+1}.txt', 'r') as file :
#         lines = [line.split() for line in  file.readlines()]
  # print(lines)
  # lines[0][0] = str(i//20)
  # lines = [" ".join(l) for l in lines]
  # print(lines)
  # # break
  # with open(f'/content/drive/MyDrive/newCoco/CustomDataset/image{i+1}.txt', "w") as f:
  #     f.writelines(lines)

# Load COCO Dataset

# # # from fiftyone import ViewField as F
# dataset = foz.load_zoo_dataset(
#     "coco-2017",
#     split="train",
#     max_samples=3000,
#     shuffle=True,
#     classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow'] 
# )

# test = foz.load_zoo_dataset(
#     "coco-2017",
#     split="validation",
#     max_samples=1000,
#     shuffle=True,
#     classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow'] 
# )

# # # bbox is scaled by the width and height

# ourclasses = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow']
# datadict = dataset.to_dict()
# testdict = test.to_dict()

# %cd /content/
# !pwd
# !mkdir /content/drive/MyDrive/newCoco/coco2017TrainImages
# !mkdir /content/drive/MyDrive/newCoco/coco2017ValidationImages
# import os
# import shutil

# def putDataInFormat(dataset:dict,split:str):
#   splitFiles = []
#   cats = []
#   for sample in dataset["samples"]:
#       imageName = sample["filepath"].split("/")[-1]
#       # !cp sample["filepath"] 
#       lines = []

#       for dect in sample["ground_truth"]["detections"]:
#           if  dect["label"] not in  ourclasses:
#             continue
#           category = ourclasses.index(dect["label"])
          

#           cats.append(dect["label"])
#           width , height  = sample["metadata"]["width"],sample["metadata"]["height"]
#           x0 = dect["bounding_box"][0] * width
#           dectWidth = dect["bounding_box"][2] * width
#           y0 = dect["bounding_box"][1] * height
#           dectHeight = dect["bounding_box"][3] * height
#           xcenter = (x0 + 0.5 * dectWidth)/ width
#           ycenter = (y0 + 0.5 * dectHeight)/ height
#           dectLine = f"{category} {xcenter} {ycenter} {dectWidth/width} {dectHeight/height}\n"
#           lines.append(dectLine)

#       if len(lines) == 0 :
#         continue
#       shutil.copyfile(sample["filepath"],os.path.join( f"/content/drive/MyDrive/newCoco/coco2017{split}Images",imageName))
#       with open(os.path.join(f"/content/drive/MyDrive/newCoco/coco2017{split}Images",imageName.split(".")[-2]+".txt"), "w")  as f:
#         f.writelines(lines)
#       splitFiles.append(os.path.join(f"/content/drive/MyDrive/newCoco/coco2017{split}Images",imageName)+"\n")
#   with open(f"/content/drive/MyDrive/newCoco/{split}Coco2017.txt","w") as f:
#     f.writelines(splitFiles)

# putDataInFormat(datadict,"Train")
# putDataInFormat(testdict,"Validation")

from google.colab import drive
drive.mount('/content/drive')

%git clone https://github.com/ultralytics/yolov5 

# Training on COCO dataset
%cd /content/yolov5
!python train.py  --batch 64 --epochs 50 --data /content/drive/MyDrive/newCoco/coco2017.yaml --cfg /content/yolov5/models/yolov5n.yaml --weights '' --device 0

# Saving Weights
%cd /content/yolov5
!python val.py --batch-size  1 --data /content/drive/MyDrive/newCoco/val.yaml  --weights '/content/drive/MyDrive/newCoco/best.pt' --device 0 --save-conf --save-txt

# Training on custom dataset
%cd /content/yolov5
!python train.py  --batch 16 --epochs 100 --data /content/drive/MyDrive/newCoco/custom.yaml --cfg /content/yolov5/models/yolov5n.yaml --weights '/content/yolov5/runs/train/exp/weights/last.pt' --device 0

# Predict
%cd /content/yolov5
!python detect.py --source /content/drive/MyDrive/newCoco/CustomDataset/image60.jpg  --weights /content/drive/MyDrive/newCoco/bestCustom.pt  --conf 0.02


# IOU 
from utils.metrics import bbox_iou
import os 
import torch
import numpy as np
expPath ='/content/yolov5/runs/val/exp8/labels'


imagesObjects = []
for imgtxt in os.listdir(expPath):
  with open(os.path.join(expPath,imgtxt),'r') as f:
    outputs = [ [float(i) for i in  l.split(" ")] for l in f.readlines()]
  outputs = np.array(outputs)
  outputs = outputs[np.flip(np.argsort(outputs[:,5]))]
  
  with open(os.path.join("/content/drive/MyDrive/newCoco/coco2017ValidationImages",imgtxt),'r') as f:
    realTargets = [ [float(i) for i in  l.split(" ")] for l in f.readlines()]
  realTargets  = np.array(realTargets)
  allIou =[]
  for o in outputs:
    if o[5] < 0.25 : continue
    maxIou = -1
    for t in realTargets:
      if t[0] != o[0] : continue
      iou = bbox_iou(torch.tensor(o[1:5]).unsqueeze(0),torch.tensor(t[1:]).unsqueeze(0))
      maxIou = max(iou.item(),maxIou)
    if maxIou != -1 :
      allIou.append(list(o)[:-1]+[maxIou])
  imagesObjects.append(allIou)