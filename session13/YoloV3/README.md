# YoloV3
cloned repo

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

________
<!-- YoloV3 Simplified for training on Colab with custom dataset.

_A Collage of Training images_
![image](https://github.com/theschoolofai/YoloV3/blob/master/output/train.png)


We have added a very 'smal' Coco sample imageset in the folder called smalcoco. This is to make sure you can run it without issues on Colab.

Full credit goes to [this](https://github.com/ultralytics/yolov3), and if you are looking for much more detailed explainiation and features, please refer to the original [source](https://github.com/ultralytics/yolov3).

You'll need to download the weights from the original source.
1. Create a folder called weights in the root (YoloV3) folder
2. Download from: https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0
3. Place 'yolov3-spp-ultralytics.pt' file in the weights folder:
  * to save time, move the file from the above link to your GDrive
  * then drag and drop from your GDrive opened in Colab to weights folder
4. run this command
`python train.py --data data/smalcoco/smalcoco.data --batch 10 --cache --epochs 25 --nosave`

For custom dataset:
1. Clone this repo: https://github.com/miki998/YoloV3_Annotation_Tool
2. Follow the installation steps as mentioned in the repo.
3. For the assignment, download 500 images of your unique object.
4. Annotate the images using the Annotation tool.
```
data
  --customdata
    --images/
      --img001.jpg
      --img002.jpg
      --...
    --labels/
      --img001.txt
      --img002.txt
      --...
    custom.data #data file
    custom.names #your class names
    custom.txt #list of name of the images you want your network to be trained on. Currently we are using same file for test/train
```
5. As you can see above you need to create **custom.data** file. For 1 class example, your file will look like this:
```
  classes=1
  train=data/customdata/custom.txt
  test=data/customdata/custom.txt
  names=data/customdata/custom.names
```
6. As you it a poor idea to keep test and train data same, but the point of this repo is to get you up and running with YoloV3 asap. You'll probably do a mistake in writing to custom.txt file. This is how our file looks like (please note the .s and /s):
```
./data/customdata/images/img001.jpg
./data/customdata/images/img002.jpg
./data/customdata/images/img003.jpg
...
```
7. You need to add custom.names file as you can see above. For our example, we downloaded images of Walle. Our custom.names file look like this:
```
walle
```
8. Walle above will have a class index of 0.
9. For COCO's 80 classes, VOLOv3's output vector has 255 dimensions ( (4+1+80)*3). Now we have 1 class, so we would need to change it's architecture.
10. Copy the contents of 'yolov3-spp.cfg' file to a new file called 'yolov3-custom.cfg' file in the data/cfg folder.
11. Search for 'filters=255' (you should get entries entries). Change 255 to 18 = (4+1+1)*3
12. Search for 'classes=80' and change all three entries to 'classes=1'
13. Since you are lazy (probably), you'll be working with very few samples. In such a case it is a good idea to change:
  * burn_in to 100
  * max_batches to 5000
  * steps to 4000,4500
14. Don't forget to perform the weight file steps mentioned in the sectio above.
15. Run this command `python train.py --data data/customdata/custom.data --batch 10 --cache --cfg cfg/yolov3-custom.cfg --epochs 3 --nosave`

As you can see in the collage image above, a lot is going on, and if you are creating a set of say 500 images, you'd get a bonanza of images via default augmentations being performed.


**Results**
After training for 300 Epochs, results look awesome!

![image](https://github.com/theschoolofai/YoloV3/blob/master/output/download.jpeg) -->
