# EVA5 Session 10 Assignment by Sachin Dangayach

**Data Augmentations**
**GIT Link for the package**: https://github.com/SachinDangayach/TSAI_EVA5/tree/master/session10

**Collab Link**: https://colab.research.google.com/drive/1K7NP69N0nlpY6uwYxe9c65iC9svZxwcD?usp=sharing

Learninig Rate

**A. Target**
1. Pick your last code
2. Make sure  to Add CutOut to your code. It should come from your transformations (albumentations)
3. Use this repo: https://github.com/davidtvs/pytorch-lr-finder
  1. Move LR Finder code to your modules
  2. Implement LR Finder (for SGD, not for ADAM)
  3. Implement ReduceLROnPlatea: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau

4. Find best LR to train your model
5. Use SDG with Momentum
6. Train for 50 Epochs.
7. Show Training and Test Accuracy curves
8. Target 88% Accuracy.
9. Run GradCAM on the any 25 misclassified images. Make sure you mention what is the prediction and what was the ground truth label.
10. Submit


**B. Results**
1. Parameters: 11,173,962
2. Best Training Accuracy in 30 epochs: 98.08 %
3. Best Test Accuracy in 30 epochs: 91.34 %
4. Total RF reached: 76*76 at the end of Conv block 4

**C. Analysis**

I have implemented Albumentations transforms for normalization ( by finding norm and std values for entire dataset ), Horizontal flip, Vertical flip Rotations and cut our. This acts as a regularizer and now the model is not overfitting to the extent it was earlier

I have implemented the LR finder from given Repo. I have chosen the LR where the loss is minimum which is 0.1 and not suggested by the code. I even tried with the code suggested LR by model was hardly training with that value.

I have also implemented the grad cam functionality and results are displayed for 30 images

**D. Loss and Accuracy curves**

![alt text](https://github.com/SachinDangayach/TSAI_EVA5/blob/master/session10/Loss_Accuracy_curves.png)

**D. misclassified images**

![alt text](https://github.com/SachinDangayach/TSAI_EVA5/blob/master/session10/Misclassified.png)

**D. Grad Cam Results**

![alt text](https://github.com/SachinDangayach/TSAI_EVA5/blob/master/session10/GradCam.png)
