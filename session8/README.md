# EVA5 Session 8 Assignment by Sachin Dangayach

**Receptive Fields & Network Architectures**

**Collab Link**: https://colab.research.google.com/drive/1_3-KnCt0YOnZlOC2Z2RBHmGUOpIx9N8F?usp=sharing

**A. Target**
[X] 1.  Go through this repository: https://github.com/kuangliu/pytorch-cifar
[X] 2.  Extract the ResNet18 model from this repository and add it to your API/repo
[X] 3.  Target is 85% accuracy. No limit on the number of epochs. Use default ResNet18 code (so params are fixed).


**B. Results**

1.  Parameters: 11,173,962
2.  Best Training Accuracy in 20 epochs: 95.73%
3.  Best Test Accuracy in 20 epochs: 85.59 %


**C. Analysis**

Model gets the desired accuracy in 10 epochs. There is training and testing accuracy gap is there. I used L2 regularizer to reduce the gap along with images transformations like rotations and horizontal shift. Gap is reduced but still scope of improvement is there.

**Accuracy and loss plots**
![alt text](https://github.com/SachinDangayach/TSAI_EVA5/blob/master/session8/session8_plots.jpeg?raw=true)

**Misclassified images**
![alt text](https://github.com/SachinDangayach/TSAI_EVA5/blob/master/session8/session8_mis_results.jpeg?raw=true)
