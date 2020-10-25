# EVA5 Session 13 Assignment by Sachin Dangayach

## Object Detection Through YOLO

**GIT Link for the package**:

https://github.com/SachinDangayach/TSAI_EVA5/tree/master/session13

## Part 1: Object Detection Through YOLOv3OpenCV

**YOLOv3OpenCV code on Github.**:

https://github.com/SachinDangayach/TSAI_EVA5/blob/master/session13/session13_A/s13_A.ipynb

**Image annotated by OpenCV YOLO inference**: 📸

![alt text](https://github.com/SachinDangayach/TSAI_EVA5/blob/master/session13/session13_A/IMG_0013_BB.jpg)

### Steps-
1. Get the code from https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/
2. Get the weights from # !wget https://pjreddie.com/media/files/yolov3.weights and config file yolo3.cfg
3. Run the code and pass the image with object which is there in COCO data set
4. Save the annotated image

## Part 2: Get the annotated dataset from link below-
https://drive.google.com/file/d/1EqtOpF7cS74C56EaVQoKNkQmpT6_HFL2/view?usp=sharing

## Part 3: Object Detection Through YoloV3 training

**GitHub project**:
https://github.com/SachinDangayach/TSAI_EVA5/tree/master/session13/YoloV3


**Colab Link**:
https://colab.research.google.com/drive/1Eej5awd6erlSvq7j-znc7a2RWnFFji8I?usp=sharing


**YouTube video (your object annotated by your YoloV3 trained model)** 🎥
https://www.youtube.com/watch?v=MOcmQjWxyTk

## Steps
1. Clone the repo https://github.com/theschoolofai/YoloV3
2. Go inside data/customdata folder and do the following changes-
    1. Copy annotated images from step to into images folder
    2. Copy labels into labels folder
    3. Create test.txt and train.txt containing the path(for example ./data/customdata/images/img_010.jpg) for the images used for test and train
    4. Create test.shapes and train.shapes file containing the sizes for test and train images
    5. Put the classes names in the custom.names file ( hardhat, vest, mask, boots)
    6. Create custom.data file containing paths for file with below mentioned information
        classes=4
        train=data/customdata/train.txt
        valid=data/customdata/test.txt
        names=data/customdata/custom.names
3. In the config folder named cfg/ change the yolo3-custom.cfg file with following values
        burn_in=16
        max_batches = 5000
        policy=steps
        steps=4000,4500
        as we are training on three resolution with 4 classes, replace filters=255 to filters=27 at three places and classes = 4
4.  Download weights 'yolov3-spp-ultralytics.pt' from https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0
        as described in README.md and place in weights folder
5.  Once this setup is done, copy the repo in google drive and mount it to colab. Use the https://colab.research.google.com/drive/1Eej5awd6erlSvq7j-znc7a2RWnFFji8I?usp=sharing to run the code with 50 epochs and batch size 16.
6. Use ffmpeg to break a downloaded video clip( with hardhat, vests etc.) into images
7. Once training in complete, use the !python detect.py --conf-thres 0.1 --output out_out --source ppe_images/ppe_input_images/ to get the images in folder
8. Download and combine the annotated images into a video and upload the video on YouTube and Enjoy 😊
