# EVA5 Session 11 Assignment by Sachin Dangayach

**Super Convergence**
**GIT Link for the package**: https://github.com/SachinDangayach/TSAI_EVA5/tree/master/session11

**Collab Link for Cyclic Curve**: https://colab.research.google.com/drive/1CdN_rOeAq-ILGxwFpwf2O2pfL8jCSrxl?usp=sharing

**Collab Link for CLR**: https://colab.research.google.com/drive/1aX_8aenrFAWGhaQ0ohsWyaN-_I7VZZmn?usp=sharing


**A. Target**

1. Write a code which

> 1. Uses this new ResNet Architecture for Cifar10:

>> 1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]

>> 2. Layer1 -

>>> 1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]

>>> 2. R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]

>>> 3. Add(X, R1)

>>3. Layer 2 -

>>> 1. Conv 3x3 [256k]

>>> 2. MaxPooling2D

>>> 3. BN

>>> 4. ReLU

>> 4. Layer 3 -

>>> 1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]

>>> 2. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]

>>> 3. Add(X, R2)

>> 5. MaxPooling with Kernel Size 4

>> 6. FC Layer

>> 7. SoftMax

> 2. Uses One Cycle Policy such that:

>> 1. Total Epochs = 24

>> 2. Max at Epoch = 5

>> 3. LRMIN = FIND

>> 4. LRMAX = FIND

>> 5. NO Annihilation

>> 3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)

>>4. Batch size = 512

>>5. Target Accuracy: 90%.


**B. Results**

1. Parameters: 6,573,120

2. Best Training Accuracy in 24 epochs: 98.81%

3. Best Test Accuracy in 24 epochs: 93.22 %


**C. Analysis**


I have implemented the Davidnet and applied the required image augmentations. I used the LR finder to find the max learning rate with test range between .0001 to .02 and with 400 iteration (nearly 5 epochs with batch size of 512 while total 50000 images in train set). I found max LR to be .03. With max LR as 0.03 and min LR as Max Lr/ 10 = 0.003, with one cycle approach, we could train the network to reach above 90% accuracy in less than 24 epocs

**D. Loss and Accuracy curves**

![alt text](https://github.com/SachinDangayach/TSAI_EVA5/blob/master/session11/Loss_Accuracy_Plot.png)

**E. LR Range Test**

![alt text](https://github.com/SachinDangayach/TSAI_EVA5/blob/master/session11/LR_Range_test.png)

**F. CYCLIC TRIANGLE plot**

![alt text](https://github.com/SachinDangayach/TSAI_EVA5/blob/master/session11/Cyclic_Plot.png)

**G. Misclassified**

![alt text](https://github.com/SachinDangayach/TSAI_EVA5/blob/master/session11/Misclassified.png)
