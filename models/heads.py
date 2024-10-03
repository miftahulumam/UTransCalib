import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from models.realignment_layer import realignment_layer
import torch.nn.functional as F

from .basic_blocks import SingleConv, DoubleConv, Residual_DoubleConv

class global_regression_v1(nn.Module):
    def __init__(self, 
                 max_channel, 
                 depthwise=True, 
                 activation='nn.ReLU(inplace=True)', 
                 drop_rate=0.1, 
                 fc_drop_rate=0.3):
        super(global_regression_v1, self).__init__()

        self.conv = nn.Sequential(
            SingleConv(max_channel, max_channel, 2, depthwise, activation, drop_rate),
            SingleConv(max_channel, max_channel, 2, depthwise, activation, drop_rate),
            SingleConv(max_channel, max_channel, 2, depthwise, activation, drop_rate),
        )

        self.fc = nn.Sequential(
            nn.Linear(max_channel*3, 256),
            eval(activation),
            nn.Dropout(fc_drop_rate),
            nn.Linear(256, 64),
            eval(activation),
            nn.Dropout(fc_drop_rate)
        )

        self.fc_trans = nn.Sequential(
            nn.Linear(64, 16),
            eval(activation),
            nn.Linear(16, 3)
        )

        self.fc_rot = nn.Sequential(
            nn.Linear(64, 16),
            eval(activation),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x_trans = self.fc_trans(x)
        x_rot = self.fc_rot(x)

        return x_trans, x_rot