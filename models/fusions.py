import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from models.realignment_layer import realignment_layer
import torch.nn.functional as F

from .basic_blocks import SingleConv, DoubleConv, Residual_DoubleConv
from .attentions import hydra_channel_attention, linear_channel_attention, spatial_attention

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

### proposed
class hybrid_attentive_fusion_lite(nn.Module):
    def __init__(self, 
                 channels = [64, 128, 256, 512], 
                 fusion_attn_repeat = None,
                 fc_reduction = 16, 
                 depthwise=True, 
                 activation='nn.SiLU(inplace=True)',
                 drop_rate=0.1):
        super(hybrid_attentive_fusion_lite, self).__init__()

        self.fusion_attn_repeat = fusion_attn_repeat

        self.single_downsampler = SingleConv(channels[2], channels[2], 2, 
                                             depthwise, activation, drop_rate)
        
        self.double_downsampler = nn.Sequential(
            SingleConv(channels[1], channels[1], 2, depthwise, activation, drop_rate),
            SingleConv(channels[1], channels[1], 2, depthwise, activation, drop_rate)
        )
        
        self.triple_downsampler = nn.Sequential(
            SingleConv(channels[0], channels[0], 2, depthwise, activation, drop_rate),
            SingleConv(channels[0], channels[0], 2, depthwise, activation, drop_rate),
            SingleConv(channels[0], channels[0], 2, depthwise, activation, drop_rate)
        )

        # fusion attention
        if self.fusion_attn_repeat == 0 or self.fusion_attn_repeat == None:
            self.spatial_attn = spatial_attention(sum(channels), reduction=fc_reduction)
            self.channel_attn = hydra_channel_attention(sum(channels))
        else:
            self.spatial_attn = nn.ModuleList()
            self.channel_attn = nn.ModuleList()

            for _ in range(self.fusion_attn_repeat):
                self.spatial_attn.append(spatial_attention(sum(channels), reduction=fc_reduction))
                self.channel_attn.append(hydra_channel_attention(sum(channels)))

    def forward(self, x1, x2, x3, x4):
        
        x1 = self.triple_downsampler(x1)
        x2 = self.double_downsampler(x2)
        x3 = self.single_downsampler(x3)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        B, C, H, W = x.shape

        if self.fusion_attn_repeat == 0 or self.fusion_attn_repeat == None:
            x_sp_attn = self.spatial_attn(x)

            x_ch_attn = x.view(B, C, H*W).permute(0, 2, 1)
            x_ch_attn = self.channel_attn(x_ch_attn).permute(0, 2, 1).reshape(B, C, H, W)

            out = torch.cat((x_sp_attn, x_ch_attn), dim=1)

        else:
            repeat_x = x

            for sa, ca in zip(self.spatial_attn, self.channel_attn):
                x_sp_attn = sa(repeat_x)

                x_ch_attn = repeat_x.view(B, C, H*W).permute(0, 2, 1)
                x_ch_attn = ca(x_ch_attn).permute(0, 2, 1).reshape(B, C, H, W)

                repeat_x = x_sp_attn + x_ch_attn
            
            out = repeat_x

        return out

### proposed
class hybrid_attentive_fusion(nn.Module):
    def __init__(self, 
                 channels = [64, 128, 256, 512], 
                 branch_attn_repeat = 1,
                 fusion_attn_repeat = None, # if none, concat; else, add
                 fc_reduction = 16, 
                 depthwise=True, 
                 activation='nn.SiLU(inplace=True)',
                 drop_rate=0.1):
        super(hybrid_attentive_fusion, self).__init__()

        self.branch_attn_repeat = branch_attn_repeat
        self.fusion_attn_repeat = fusion_attn_repeat

        self.single_downsampler = SingleConv(channels[2], channels[2], 2, 
                                             depthwise, activation, drop_rate)
        
        self.double_downsampler = nn.Sequential(
            SingleConv(channels[1], channels[1], 2, depthwise, activation, drop_rate),
            SingleConv(channels[1], channels[1], 2, depthwise, activation, drop_rate)
        )
        
        self.triple_downsampler = nn.Sequential(
            SingleConv(channels[0], channels[0], 2, depthwise, activation, drop_rate),
            SingleConv(channels[0], channels[0], 2, depthwise, activation, drop_rate),
            SingleConv(channels[0], channels[0], 2, depthwise, activation, drop_rate)
        )

        # branch attentions
        self.branch_1_attn = nn.ModuleList() 
        self.branch_2_attn = nn.ModuleList() 
        self.branch_3_attn = nn.ModuleList() 
        self.branch_4_attn = nn.ModuleList()

        if self.branch_attn_repeat == 0 or self.branch_attn_repeat == None:
            self.branch_1_attn.append(nn.Identity())
            self.branch_2_attn.append(nn.Identity())
            self.branch_3_attn.append(nn.Identity())
            self.branch_4_attn.append(nn.Identity())

        else:
            for _ in range(self.branch_attn_repeat):
                self.branch_1_attn.append(spatial_attention(channels[0], reduction=fc_reduction))
                self.branch_2_attn.append(spatial_attention(channels[1], reduction=fc_reduction))
                self.branch_3_attn.append(spatial_attention(channels[2], reduction=fc_reduction))
                self.branch_4_attn.append(spatial_attention(channels[3], reduction=fc_reduction))

        # fusion attention
        if self.fusion_attn_repeat == 0 or self.fusion_attn_repeat == None:
            self.spatial_attn = spatial_attention(sum(channels), reduction=fc_reduction)
            self.channel_attn = hydra_channel_attention(sum(channels))

        else:
            self.spatial_attn = nn.ModuleList()
            self.channel_attn = nn.ModuleList()

            for _ in range(self.fusion_attn_repeat):
                self.spatial_attn.append(spatial_attention(sum(channels), reduction=fc_reduction))
                self.channel_attn.append(hydra_channel_attention(sum(channels)))

    def forward(self, x1, x2, x3, x4):
        
        x1 = self.triple_downsampler(x1)
        x2 = self.double_downsampler(x2)
        x3 = self.single_downsampler(x3)

        # branch attentions
        for attn_1, attn_2, attn_3, attn_4 in zip(self.branch_1_attn, 
                                                  self.branch_2_attn,
                                                  self.branch_3_attn, 
                                                  self.branch_4_attn):
            x1 = attn_1(x1)
            x2 = attn_2(x2)
            x3 = attn_3(x3)
            x4 = attn_4(x4) 

        x = torch.cat((x1, x2, x3, x4), dim=1)

        # fusion attention
        B, C, H, W = x.shape

        if self.fusion_attn_repeat == 0 or self.fusion_attn_repeat == None:
            x_sp_attn = self.spatial_attn(x)

            x_ch_attn = x.view(B, C, H*W).permute(0, 2, 1)
            x_ch_attn = self.channel_attn(x_ch_attn).permute(0, 2, 1).reshape(B, C, H, W)

            out = torch.cat((x_sp_attn, x_ch_attn), dim=1)

        else:
            repeat_x = x

            for sa, ca in zip(self.spatial_attn, self.channel_attn):
                x_sp_attn = sa(repeat_x)

                x_ch_attn = repeat_x.view(B, C, H*W).permute(0, 2, 1)
                x_ch_attn = ca(x_ch_attn).permute(0, 2, 1).reshape(B, C, H, W)

                repeat_x = x_sp_attn + x_ch_attn
            
            out = repeat_x

        return out
