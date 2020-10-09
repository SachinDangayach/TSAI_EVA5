# Module to define model architecture for CIFAR10 data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""Class to define Model architecture for classification of cifar10"""
class Net(nn.Module):
    def __init__(self, dropout_value = 0.2):
        super(Net, self).__init__()

        self.prep_layer = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride = 1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU()
        )

        self.layer1= nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride = 1, padding=1, bias=False),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(128),
        nn.ReLU()
        )

        self.res1= nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride = 1, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride = 1, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        )

        self.layer2= nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride = 1, padding=1, bias=False),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(256),
        nn.ReLU()
        )

        self.layer3= nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride = 1, padding=1, bias=False),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(512),
        nn.ReLU()
        )

        self.res2= nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride = 1, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(),

        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride = 1, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        )

        self.maxpool = nn.MaxPool2d(4, 4)

        self.fc = nn.Linear(in_features = 512, out_features = 10, bias=False)

    def forward(self, x):

        prep = self.prep_layer(x)
        x1   = self.layer1(prep)
        r1   = self.res1(x1)
        layer1 = x1 + r1

        layer2 = self.layer2(layer1)

        x2 = self.layer3(layer2)
        r2 = self.res2(x2)
        layer3 = x2 + r2

        maxpool = self.maxpool(layer3)

        x = maxpool.view(maxpool.size(0),-1)

        fc = self.fc(x)

        return F.log_softmax(fc.view(-1,10), dim=-1)
