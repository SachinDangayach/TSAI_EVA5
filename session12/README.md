# EVA5 Session 12 Assignment by Sachin Dangayach

## ResNet18 for TinyImageNet

**GIT Link for the package**: https://github.com/SachinDangayach/TSAI_EVA5/tree/master/session11

**Collab Link for TinyImageNet Training**: https://colab.research.google.com/drive/1aX_8aenrFAWGhaQ0ohsWyaN-_I7VZZmn?usp=sharing

## Data Preparations for YOLO

**Collab Link for Data Preparations for YOLO**: https://colab.research.google.com/drive/1aX_8aenrFAWGhaQ0ohsWyaN-_I7VZZmn?usp=sharing

**GIT Link for annotated dataset with 50+ classes**: https://github.com/SachinDangayach/TSAI_EVA5/tree/master/session12/dataset/DataSet

**GIT Link for JSON file**: https://github.com/SachinDangayach/TSAI_EVA5/blob/master/session12/dataset/DataSet/s12_dataset.json


## Assignment A: TinyImagenet training

**A. Target**
1. Download this TINY IMAGENET dataset.
2. Train ResNet18 on this dataset (70/30 split) for 50 Epochs. Target 50%+ Validation Accuracy.
3. Submit Results. Of course, you are using your own package for everything.

**B. Results**

1. Parameters: 11,271,432

2. Best Training Accuracy in 30 epochs: 98.81%

3. Best Test Accuracy in 30 epochs: 93.22 %

**C. Analysis**

I have used one cycle learning with max learning rate of 0.02 and minimum of 0.002. Max LR is reached in 11 epochs. I have used augmentations (horizontal flip, resizing and random cropping, rotation and cutout with normalization) to regularize the training. Model could achieve required accuracy in

**D. Loss and Accuracy curves**

![alt text](https://github.com/SachinDangayach/TSAI_EVA5/blob/master/session11/Loss_Accuracy_Plot.png)

## Assignment B: Data Preparations for YOLO

## Target
1. Download 50 (min) images each of people wearing hardhat, vest, mask and boots.
2. Use these labels (same spelling and small letters):
    1. hardhat
    2. vest
    3. mask
    4. boots
3. Use this to annotate bounding boxes around the hardhat, vest, mask and boots.
4. Download JSON file.
5. Describe the contents of this JSON file in FULL details (you don't need to describe all 10 instances, anyone would work).
6. Refer to this tutorial . Find out the best total numbers of clusters. Upload link to your Colab File uploaded to GitHub.

## Description of Jason-
1. Key: Example img_001.jpg116338. This is unique key (concatination of image name and image size)
2. filename: Name of the image file
3. size: Size of image
4. regions: It consists of shape_attributes and region_attributes (both described below)
  1. shape_attributes: It is collection of attributes describing the bonding boxes. Below are the fields.
    1. name: There are 6 different options in vgg tool for region selection. As we have selected for rectangular option, value is rect in jason file
    2. b. x: We consider origin as top left corner of image. x is horizontal distance of bounding box top left corner from origin (Image's top left)
    3. c. y: We consider origin as top left corner of image. y is vertical distance of bounding box top left corner from origin (Image's top left)
    4. width: Width of bounding box
    5. height: Height of bounding box
  2. region_attributes: It consists of region_attributes. We have only one region attribute named class
    1. class: it is one of the four class values (hardhat, mask, boots, vest)
5. file_attributes: We have not used this attribute and this is empty for all images.
