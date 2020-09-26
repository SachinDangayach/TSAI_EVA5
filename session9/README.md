# EVA5 Session 7 Assignment by Sachin Dangayach

**Advanced Convolutions**
**GIT Link for the package**: [https://github.com/SachinDangayach/EVA5](https://github.com/SachinDangayach/EVA5)

**Collab Link**: https://colab.research.google.com/drive/1S0wf6mBJGQhYpkI7p38leuicX_5Mto2u?usp=sharing

**A. Target**

1. Move your last code's transformations to Albumentations. Apply ToTensor,     HorizontalFlip, Normalize (at min) + More (for additional points)
2. Please make sure that your test_transforms are simple and only using ToTensor and Normalize
3. Implement GradCam function as a module.
4. Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality
5. Target Accuracy is 87%

**B. Results**

1.  Parameters: 11,173,962
2.  Best Training Accuracy in 20 epochs: 95.87%
3.  Best Test Accuracy in 20 epochs: 87.66%
4.  Total RF reached: 76*76 at the end of Conv block 4

**C. Analysis**

I have implemented Albumentations transforms for normalization ( by finding norm and std values for entire dataset ), Horizontal flip, Vertical flip, Rotations. This acts as a regularizer and now the model is not overfitting as earlier.
