# Module to define model architecture for CIFAR10 data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""Class to define Model architecture for classification of cifar10_mean_std"""
class Net(nn.Module):
    def __init__(self, dropout_value = 0.2):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 1), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
            )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
            )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
            )

        # TRANSITION BLOCK 1
        self.transblock1 = nn.Sequential(
            nn.MaxPool2d(2, 2, 1)
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
            )

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        )

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        )

        # TRANSITION BLOCK 2
        self.transblock2 = nn.Sequential(
            nn.MaxPool2d(2, 2, 1)
        )

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64 out_channels=128, kernel_size=(1, 1), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
            )

        self.convblock8= nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        )

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        )

        # # OUTPUT BLOCK
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))   #nn.AvgPool2d(kernel_size=6),

        self.fc = nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),



    def forward(self, x):

        x1 = self.convblock1(x)
        x2 = self.convblock2(x1)
        x3 = self.convblock3(x1 + x2)
        x4 = self.transblock1(x1 + x2 + x3)
        x4 = self.convblock4(x4)
        x5 = self.convblock5(x4)
        x6 = self.convblock6(x4 + x5)
        x7 = self.convblock6(x4 + x5 +x6)
        x8 = self.transblock2(x5 + x6 + x7)
        x8 = self.convblock7(x8)
        x9 = self.convblock8(x8)
        x10 = self.convblock9(x8 + x9)
        x11 = self.convblock10(x8 + x9 + x10)
        x12 = self.gap(x11)
        x13 = self.fc(x12)
        x14 = x13.view(-1, 10)

        return F.log_softmax(x, dim=-1)
