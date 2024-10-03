import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from models.realignment_layer import realignment_layer
import torch.nn.functional as F

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, depthwise=False, activation='nn.SiLU(inplace=True)', drop_rate=0.1):
        super(SingleConv, self).__init__()

        if depthwise:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(out_channels),
                eval(activation),
                nn.Dropout2d(drop_rate))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=1, bias=False),
                nn.BatchNorm2d(out_channels),
                eval(activation),
                nn.Dropout2d(drop_rate))

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, depthwise=False, activation='nn.SiLU(inplace=True)', drop_rate=0.1):
        super(DoubleConv, self).__init__()

        if not mid_channels:
            mid_channels = out_channels

        if depthwise: # depthwise separable conv to reduce parameters
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                eval(activation),
                nn.Dropout2d(drop_rate),

                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(out_channels),
                eval(activation),
                nn.Dropout2d(drop_rate)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                eval(activation),
                nn.Dropout2d(drop_rate),

                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                eval(activation),
                nn.Dropout2d(drop_rate)
            )

    def forward(self, x):
        return self.double_conv(x)
    
class Residual_DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, depthwise=False, activation=nn.ReLU, drop_rate=0.1):
        super(Residual_DoubleConv, self).__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            activation(inplace=True),
            nn.Dropout2d(drop_rate),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
            nn.Dropout2d(drop_rate)
        )

        self.double_conv_2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            activation(inplace=True),
            nn.Dropout2d(drop_rate),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
            nn.Dropout2d(drop_rate)
        )

    def forward(self, x):
        skip_1 = x
        x = self.double_conv_1(x)
        x = skip_1 + x
        
        skip_2 = x
        x = self.double_conv_2(x)
        x = skip_2 + x
        
        return x

class upsampling(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, depthwise=False, activation='nn.SiLU(inplace=True)', drop_rate=0.1):
        super(upsampling, self).__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2, depthwise, activation, drop_rate)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, depthwise=depthwise, activation=activation, drop_rate=drop_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

class Residual_upsampling(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, depthwise=False, activation=nn.ReLU, drop_rate=0.1):
        super(Residual_upsampling, self).__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = Residual_DoubleConv(in_ch, out_ch, in_ch // 2, depthwise, activation, drop_rate)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = Residual_DoubleConv(in_ch, out_ch, depthwise=depthwise, activation=activation, drop_rate=drop_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x) 