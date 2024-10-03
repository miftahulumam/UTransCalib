import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from models.realignment_layer import realignment_layer
import torch.nn.functional as F

from .basic_blocks import SingleConv, DoubleConv, Residual_DoubleConv

# FEATURE MATCHING // FEATURE FUSION  

# PS-Net

class cross_feature_fusion_module(nn.Module):
    def __init__(self, 
                 max_channel, 
                 fc_reduction,  
                 activation='nn.SiLU(inplace=True)'):
        super(cross_feature_fusion_module, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(max_channel*4, max_channel*4 // fc_reduction, bias=False),
            eval(activation),
            nn.Linear(max_channel*4 // fc_reduction, 4, bias=False),
            eval(activation),
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x1, x2, x3, x4):
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = torch.squeeze(self.avg_pool(x))
        x = torch.unsqueeze(self.fc(x), 2)

        a1, a2, a3, a4 = torch.chunk(x, 4, dim=1)
        a1 = torch.unsqueeze(a1, 1).expand_as(x1)
        a2 = torch.unsqueeze(a2, 1).expand_as(x2)
        a3 = torch.unsqueeze(a3, 1).expand_as(x3)
        a4 = torch.unsqueeze(a4, 1).expand_as(x4)
        
        # print(a1.shape, a2.shape, a3.shape, a4.shape)
        out = torch.mul(a1, x1) + torch.mul(a2, x2) + torch.mul(a3, x3) + torch.mul(a4, x4)

        return out
    
class cross_feature_fusion_3maps(nn.Module):
    def __init__(self, 
                 max_channel, 
                 fc_reduction,  
                 activation='nn.SiLU(inplace=True)'):
        super(cross_feature_fusion_3maps, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(max_channel*3, max_channel*3 // fc_reduction, bias=False),
            eval(activation),
            nn.Linear(max_channel*3 // fc_reduction, 3, bias=False),
            eval(activation),
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim=1)

        x = torch.squeeze(self.avg_pool(x))
        x = torch.unsqueeze(self.fc(x), 2)

        a1, a2, a3 = torch.chunk(x, 3, dim=1)
        a1 = torch.unsqueeze(a1, 1).expand_as(x1)
        a2 = torch.unsqueeze(a2, 1).expand_as(x2)
        a3 = torch.unsqueeze(a3, 1).expand_as(x3)
        
        # print(a1.shape, a2.shape, a3.shape, a4.shape)
        out = torch.mul(a1, x1) + torch.mul(a2, x2) + torch.mul(a3, x3)

        return out

class feature_fusion_full(nn.Module):
    def __init__(self, 
                 max_channel, 
                 fc_reduction, 
                 depthwise=True, 
                 activation='nn.SiLU(inplace=True)', 
                 drop_rate=0.1):
        super(feature_fusion_full, self).__init__()

        self.max_channel = max_channel

        self.single_downsampler = SingleConv(max_channel // 2, max_channel, 2, 
                                             depthwise, activation, drop_rate)
        
        self.double_downsampler = nn.Sequential(
            SingleConv(max_channel // 4, max_channel // 2, 2, depthwise, activation, drop_rate),
            SingleConv(max_channel // 2, max_channel, 2, depthwise, activation, drop_rate)
        )
        
        self.triple_downsampler = nn.Sequential(
            SingleConv(max_channel // 8, max_channel // 4, 2, depthwise, activation, drop_rate),
            SingleConv(max_channel // 4, max_channel // 2, 2, depthwise, activation, drop_rate),
            SingleConv(max_channel // 2, max_channel, 2, depthwise, activation, drop_rate)
            )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(max_channel, max_channel // fc_reduction, bias=False),
            eval(activation),
        )

        # parallel fc
        self.fc1 = nn.Sequential(
            nn.Linear(max_channel // fc_reduction, max_channel, bias=False),
            eval(activation),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(max_channel // fc_reduction, max_channel, bias=False),
            eval(activation),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(max_channel // fc_reduction, max_channel, bias=False),
            eval(activation),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(max_channel // fc_reduction, max_channel, bias=False),
            eval(activation),
        )

        self.softmax = nn.Softmax(dim=1)

        self.map_fusion = cross_feature_fusion_module(max_channel,
                                                      fc_reduction,
                                                      activation)
        

    def forward(self, x1, x2, x3, x4):

        x1 = self.triple_downsampler(x1)
        x2 = self.double_downsampler(x2)
        x3 = self.single_downsampler(x3)

        x = x1 + x2 + x3 + x4
        # print("check 1", x1.shape, x2.shape, x3.shape, x4.shape)
        y = torch.squeeze(self.avg_pool(x))
        # print("check 2", y.shape)
        y = self.fc(y) 
        # print("check 3", y.shape)
        y1 = torch.unsqueeze(self.fc1(y), 2)
        y2 = torch.unsqueeze(self.fc2(y), 2)
        y3 = torch.unsqueeze(self.fc3(y), 2)
        y4 = torch.unsqueeze(self.fc4(y), 2)
        # print("check 4", y1.shape, y2.shape, y3.shape, y4.shape)
        z = torch.cat((y1,y2,y3,y4), dim=2)
        # print("check 5", z.shape)
        z = self.softmax(z) #.view(b, c, 4)
        # print("check 5", z.shape)
        z1, z2, z3, z4 = torch.chunk(z, 4, 2)
        # print("check 6", z1.shape, z2.shape, z3.shape, z4.shape)
        z1 = torch.unsqueeze(z1, 2).expand_as(x1)
        z2 = torch.unsqueeze(z2, 2).expand_as(x2)
        z3 = torch.unsqueeze(z3, 2).expand_as(x3)
        z4 = torch.unsqueeze(z4, 2).expand_as(x4)
        # print("check 7", z1.shape, z2.shape, z3.shape, z4.shape)
        out1 = torch.mul(x1,z1)
        out2 = torch.mul(x2,z2)
        out3 = torch.mul(x3,z3)
        out4 = torch.mul(x4,z4)

        out = self.map_fusion(out1, out2, out3, out4)

        return out
    
class feature_fusion_3maps(nn.Module):
    def __init__(self, 
                 max_channel, 
                 fc_reduction, 
                 depthwise=True, 
                 activation='nn.SiLU(inplace=True)',
                 drop_rate=0.1):
        super(feature_fusion_3maps, self).__init__()

        self.max_channel = max_channel

        self.single_downsampler = SingleConv(max_channel // 2, max_channel, 2, depthwise, activation, drop_rate)
        
        self.double_downsampler = nn.Sequential(
            SingleConv(max_channel // 4, max_channel // 2, 2, depthwise, activation, drop_rate),
            SingleConv(max_channel // 2, max_channel, 2, depthwise, activation, drop_rate)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(max_channel, max_channel // fc_reduction, bias=False),
            eval(activation),
        )

        # parallel fc
        self.fc2 = nn.Sequential(
            nn.Linear(max_channel // fc_reduction, max_channel, bias=False),
            eval(activation),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(max_channel // fc_reduction, max_channel, bias=False),
            eval(activation),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(max_channel // fc_reduction, max_channel, bias=False),
            eval(activation),
        )

        self.softmax = nn.Softmax(dim=1)

        self.map_fusion = cross_feature_fusion_3maps(max_channel,
                                                      fc_reduction,
                                                      activation)
        

    def forward(self, x2, x3, x4):

        x2 = self.double_downsampler(x2)
        x3 = self.single_downsampler(x3)

        x = x2 + x3 + x4
        # print("check 1", x2.shape, x3.shape, x4.shape)
        y = torch.squeeze(self.avg_pool(x))
        # print("check 2", y.shape)
        y = self.fc(y) 
        # print("check 3", y.shape)
        y2 = torch.unsqueeze(self.fc2(y), 2)
        y3 = torch.unsqueeze(self.fc3(y), 2)
        y4 = torch.unsqueeze(self.fc4(y), 2)
        # print("check 4", y2.shape, y3.shape, y4.shape)
        z = torch.cat((y2,y3,y4), dim=2)
        z = self.softmax(z) #.view(b, c, 4)
        # print("check 5", z.shape)
        z2, z3, z4 = torch.chunk(z, 3, 2)
        # print("check 6", z2.shape, z3.shape, z4.shape)
        z2 = torch.unsqueeze(z2, 2).expand_as(x2)
        z3 = torch.unsqueeze(z3, 2).expand_as(x3)
        z4 = torch.unsqueeze(z4, 2).expand_as(x4)
        # print("check 7", z2.shape, z3.shape, z4.shape)
        out2 = torch.mul(x2,z2)
        out3 = torch.mul(x3,z3)
        out4 = torch.mul(x4,z4)

        out = self.map_fusion(out2, out3, out4)

        return out
    
class feature_fusion_2maps(nn.Module):
    def __init__(self, 
                 max_channel, 
                 fc_reduction, 
                 depthwise=False, 
                 activation='nn.SiLU(inplace=True)',
                 drop_rate=0.1):
        super(feature_fusion_2maps, self).__init__()

        self.max_channel = max_channel

        self.single_downsampler = SingleConv(max_channel // 2, max_channel, 2, depthwise, activation, drop_rate)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(max_channel, max_channel // fc_reduction, bias=False),
            eval(activation),
        )

        # parallel fc
        self.fc3 = nn.Sequential(
            nn.Linear(max_channel // fc_reduction, max_channel, bias=False),
            eval(activation),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(max_channel // fc_reduction, max_channel, bias=False),
            eval(activation),
        )

        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x3, x4):

        x3 = self.single_downsampler(x3)

        x = x3 + x4

        y = torch.squeeze(self.avg_pool(x))
        
        y = self.fc(y) 

        y3 = torch.unsqueeze(self.fc3(y), 2)
        y4 = torch.unsqueeze(self.fc4(y), 2)

        z = torch.cat((y3,y4), dim=2)
        z = self.softmax(z) #.view(b, c, 4)
        # print(z.shape)
        z3, z4 = torch.chunk(z, 2, 2)
        # print(z3.shape)
        z3 = torch.unsqueeze(z3, 2).expand_as(x3)
        z4 = torch.unsqueeze(z4, 2).expand_as(x4)

        out = torch.mul(x3,z3) + torch.mul(x4,z4)

        return out