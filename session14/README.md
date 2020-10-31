# EVA5 Session 14 (15-A) Assignment by Sachin Dangayach

## RCNN Family: RCNN, Fast-RCNN, FasterRCNN & MaskRCNN

**GIT Link for the package**:

https://github.com/SachinDangayach/TSAI_EVA5/tree/master/session14

**Google Drive Link for Depth Maps Images dataset**:

https://drive.google.com/file/d/1ALKKsABUnbI0FodCvdFRRGwcfFUVirvd/view?usp=sharing

**Google Drive Link for planer images dataset**:
https://drive.google.com/file/d/1ycFC7INzTajJFmEV22ns-v813-wfJBta/view?usp=sharing

**Bounding boxes images from last assignment**
https://drive.google.com/file/d/1pPr9k23ChvLd-I0K8WZnuc_yZZriOHty/view?usp=sharing

## Assignment Explanation

After implementing YoloV3 in the last Assignment, where we first annotated the
images with bounding boxes for four classes (hardhat, mask, vests and boots), we
tuned the model and got the images annotated by YoloV3 to create a cool video which
was upload to YouTube, we are heading towards the most complex Assignment.

We focused on the part A ( Data Preparations ) in this assignment where we followed
following steps

We started with the dataset from assignment 13 (helmet, mask, PPE, and boot)

steps-
0. Image dataset with bounding boxes for the classes: From last assignment
1. Generate Images dataset with dataset depth map - Clone the repo https://github.com/intel-isl/MiDaS
2. Follow the instructions as mentioned in the repo (place) the images of dataset in input folder,
download the required weights file and execute the run.py file to get the **depth maps** of the images.

Google Drive Link for Depth Maps Images: https://drive.google.com/file/d/1ALKKsABUnbI0FodCvdFRRGwcfFUVirvd/view?usp=sharing

3. We now moved to the next step for generating the planner images and for this we referred to the below mentioned repo-
https://github.com/NVlabs/planercnn

4. As initial setup was complex, I took help from the code shared in the group, and did changes in evaluate.py like removing
the file types which are not required (.npy). Changes to let the entire dataset run (changed the sampleIndex to 5000 as we have
  3500+ images in dataset)

5. Commented the lines of code to get only the input and the planner segmentation imaged only generate as otherwise the dataset
was getting huge with unwanted images.

6. Ran the evaluate.py for all the images in the dataset, zipped the output and stored in google drive, for which link is shared above.
I have uploaded the changed evaluate.py and visualize_utils.py file in the git repo for this assignment. colab link is placed below.

  Link to colab file: https://colab.research.google.com/drive/1i6zoFy004cb8ZoXMt_8rsGLMQ9ntIhkJ?usp=sharing
