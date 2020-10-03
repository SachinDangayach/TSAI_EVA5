# EVA5 Session 9 Assignment by Sachin Dangayach

**Data Augmentations**
**GIT Link for the package**: https://github.com/SachinDangayach/TSAI_EVA5/tree/master/session9

**Collab Link**: https://colab.research.google.com/drive/1rYfl2zH4NhUtKDzq-FTqgRDDpDpi-M3J?usp=sharing

**A. Target**

1. Move your last code's transformations to Albumentations. Apply ToTensor,     HorizontalFlip, Normalize (at min) + More (for additional points)
2. Please make sure that your test_transforms are simple and only using ToTensor and Normalize
3. Implement GradCam function as a module.
4. Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality
5. Target Accuracy is 87%

**B. Results**

1.  Parameters: 11,173,962
2.  Best Training Accuracy in 25 epochs: 96.51%
3.  Best Test Accuracy in 25 epochs: 87.96%
4.  Total RF reached: 76*76 at the end of Conv block 4

**C. Analysis**

I have implemented Albumentations transforms for normalization ( by finding norm and std values for entire dataset ), Horizontal flip, Vertical flip, Rotations. This acts as a regularizer and now the model is not overfitting as much as it was earlier.
I have also implemented the grad cam functionality and results are displayed for few of the images

**D. Loss and Accuracy curves**

![alt text](https://github.com/SachinDangayach/TSAI_EVA5/blob/master/session9/Loss_Accuracy_curves.png)

**D. misclassified images**

![alt text](https://github.com/SachinDangayach/TSAI_EVA5/blob/master/session9/Misclassified.png)

**D. Grad Cam Results**

![alt text](https://github.com/SachinDangayach/TSAI_EVA5/blob/master/session9/GradCam.png)
