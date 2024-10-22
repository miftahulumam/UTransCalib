import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from models.realignment_layer import realignment_layer
import torch.nn.functional as F

from torchvision.models import resnet18, densenet121, mobilenet_v3_small

def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)
            
        if isinstance(module, old):
            ## simple module
            setattr(model, n, new)

    return model

class encoder_resnet(nn.Module):
    def __init__(self, pretrained = True, depth_branch = False, activation=None):
        super(encoder_resnet, self).__init__()
        
        resnet = resnet18(weights='IMAGENET1K_V1') if pretrained else resnet18()

        if activation is not None:
            new_act = eval(activation)
            resnet = replace_layers(resnet, nn.ReLU, new_act)

        if depth_branch:
            resnet.conv1 = nn.Conv2d(1, 
                                     64, 
                                     kernel_size=(7, 7), 
                                     stride=(2, 2), 
                                     padding=(3, 3), bias=False)

        self.stem_layers = nn.Sequential(*list(resnet.children())[:4])
        self.res_layer_1 = resnet.layer1
        self.res_layer_2 = resnet.layer2
        self.res_layer_3 = resnet.layer3
        self.res_layer_4 = resnet.layer4

    def forward(self, image):
        x = self.stem_layers(image)
        x1 = self.res_layer_1(x)
        x2 = self.res_layer_2(x1)
        x3 = self.res_layer_3(x2)
        x4 = self.res_layer_4(x3)

        return x1, x2, x3, x4
    
class encoder_densenet(nn.Module):
    def __init__(self, pretrained = True, depth_branch = False, activation = None):
        super(encoder_densenet, self).__init__()

        densenet = densenet121(weights='IMAGENET1K_V1').features if pretrained else densenet121().features

        if activation is not None:
            new_act = eval(activation)
            densenet = replace_layers(densenet, nn.ReLU, new_act)
        
        if depth_branch:
            self.stem_layers = nn.Sequential(
                                        nn.Conv2d(1, 64, 
                                                kernel_size=(7, 7), 
                                                stride=(2, 2), 
                                                padding=(3, 3), 
                                                bias=False),
                                        *list(
                                            densenet.children())[1:4])
        else:
            self.stem_layers = nn.Sequential(
                                        *list(
                                            densenet.children())[:4])
            
        transition_1 = nn.Sequential(*list(densenet.children())[5])
        transition_2 = nn.Sequential(*list(densenet.children())[7]) 
        transition_3 = nn.Sequential(*list(densenet.children())[9])
            
        self.dense_layer_1 = nn.Sequential(*list(densenet.children())[4:5],
                                           *list(transition_1.children())[:3])
        
        self.dense_layer_2 = nn.Sequential(*list(transition_1.children())[3:],
                                           *list(densenet.children())[6:7],
                                           *list(transition_2.children())[:3])
        
        self.dense_layer_3 = nn.Sequential(*list(transition_2.children())[3:],
                                           *list(densenet.children())[8:9],
                                           *list(transition_3.children())[:3])
                    
        self.dense_layer_4 = nn.Sequential(*list(transition_3.children())[3:],
                                           *list(densenet.children())[10:])

    def forward(self, image):
        x = self.stem_layers(image)
        x1 = self.dense_layer_1(x)
        x2 = self.dense_layer_2(x1)
        x3 = self.dense_layer_3(x2)
        x4 = self.dense_layer_4(x3)

        return x1, x2, x3, x4
    
class encoder_mobilenet_small(nn.Module):
    def __init__(self, pretrained = True, depth_branch = False, activation = None):
        super(encoder_mobilenet_small, self).__init__()

        mobilenet = mobilenet_v3_small(weights='IMAGENET1K_V1').features if pretrained else mobilenet_v3_small().features

        if activation is not None:
            new_act = eval(activation)
            mobilenet = replace_layers(mobilenet, nn.ReLU, new_act)
        
        if depth_branch:
            self.stem_layers = nn.Sequential(
                                        nn.Conv2d(1, 16, 
                                                  kernel_size=(3, 3), 
                                                  stride=(2, 2), 
                                                  padding=(1, 1), 
                                                  bias=False),
                                        nn.BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                                        nn.Hardswish(),
                                        *list(mobilenet.children())[1:2])                     
        else:
            self.stem_layers = nn.Sequential(
                                        *list(mobilenet.children())[:2])
            
        self.layer_1 = nn.Sequential(*list(mobilenet.children())[2:4])
        self.layer_2 = nn.Sequential(*list(mobilenet.children())[4:8])
        self.layer_3 = nn.Sequential(*list(mobilenet.children())[8:12])

    def forward(self, image):
        x1 = self.stem_layers(image)
        x2 = self.layer_1(x1)
        x3 = self.layer_2(x2)
        x4 = self.layer_3(x3)

        return x1, x2, x3, x4
            
        
# encoder made from scratch
class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch = None, 
                 reduction = 8, act_layer='nn.SiLU(inplace=True)'):
        super(conv_block, self).__init__()

        mid_ch = in_ch // reduction
        if out_ch is None:
            out_ch = in_ch
        
        # reduction conv
        self.pointwise_in = nn.Sequential(nn.Conv2d(in_ch, mid_ch,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    groups=1,
                                                    bias=False),
                                          nn.BatchNorm2d(mid_ch),
                                          eval(act_layer))
        
        # attns
        self.conv_branch_1 = nn.Sequential(nn.Conv2d(mid_ch, mid_ch,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     dilation=1,
                                                     groups=mid_ch,
                                                     bias=False),
                                          nn.BatchNorm2d(mid_ch),
                                          eval(act_layer))
        
        self.conv_branch_2 = nn.Sequential(nn.Conv2d(mid_ch, mid_ch,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=2,
                                                     dilation=2,
                                                     groups=mid_ch,
                                                     bias=False),
                                          nn.BatchNorm2d(mid_ch),
                                          eval(act_layer))
        
        self.conv_branch_3 = nn.Sequential(nn.Conv2d(mid_ch, mid_ch,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=3,
                                                     dilation=3,
                                                     groups=mid_ch,
                                                     bias=False),
                                          nn.BatchNorm2d(mid_ch),
                                          eval(act_layer))
        
        self.max_branch_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv2d(mid_ch, mid_ch,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=1,
                      bias=False),
            nn.BatchNorm2d(mid_ch),
            eval(act_layer)
            )
        
        self.max_branch_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=2),
            nn.Conv2d(mid_ch, mid_ch,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=1,
                      bias=False),
            nn.BatchNorm2d(mid_ch),
            eval(act_layer)
            )
    

        # expansion conv
        self.pointwise_out = nn.Sequential(nn.Conv2d(mid_ch*5, out_ch,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    groups=1,
                                                    bias=False),
                                           nn.BatchNorm2d(in_ch),
                                           eval(act_layer))

    def forward(self, x):
        res = x

        x = self.pointwise_in(x)

        x1 = self.conv_branch_1(x)
        x2 = self.conv_branch_2(x)
        x3 = self.conv_branch_3(x)
        x4 = self.max_branch_1(x)
        x5 = self.max_branch_2(x)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.pointwise_out(x)

        x = torch.cat((x, res), dim=1)
        
        return x
    
class transition_layers(nn.Module):
    def __init__(self, in_ch, out_ch = None, act_layer='nn.SiLU(inplace=True)'):
        super(transition_layers, self).__init__()

        if out_ch is None:
            out_ch = in_ch // 2

        self.transition = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 
                      kernel_size=(1,1), stride=(1,1), bias=False),
            nn.BatchNorm2d(out_ch),
            eval(act_layer),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x):
        return self.transition(x)

class encoder_lite(nn.Module):
    def __init__(self, 
                 depth_branch = False, 
                 stem_channels=64, 
                 act_layer='nn.SiLU(inplace=True)'):
        super(encoder_lite, self).__init__()

        if depth_branch:
            self.stem_layers = nn.Sequential(
                                        nn.Conv2d(1, stem_channels, 
                                                kernel_size=(7, 7), 
                                                stride=(2, 2), 
                                                padding=(3, 3), 
                                                bias=False),
                                        nn.BatchNorm2d(stem_channels),
                                        eval(act_layer),
                                        nn.MaxPool2d(kernel_size=3, 
                                                     stride=2, 
                                                     padding=1, 
                                                     dilation=1, 
                                                     ceil_mode=False)
            )
        else:
            self.stem_layers = nn.Sequential(
                                        nn.Conv2d(3, stem_channels, 
                                                kernel_size=(7, 7), 
                                                stride=(2, 2), 
                                                padding=(3, 3), 
                                                bias=False),
                                        nn.BatchNorm2d(stem_channels),
                                        eval(act_layer),
                                        nn.MaxPool2d(kernel_size=3, 
                                                     stride=2, 
                                                     padding=1, 
                                                     dilation=1, 
                                                     ceil_mode=False)
            )

        self.conv_block_1 = nn.Sequential(
            conv_block(stem_channels, act_layer=act_layer),
            conv_block(stem_channels*2, act_layer=act_layer)
        )
        self.transition_1 = transition_layers(stem_channels*4, act_layer=act_layer)

        self.conv_block_2 = nn.Sequential(
            conv_block(stem_channels*2, act_layer=act_layer),
            conv_block(stem_channels*4, act_layer=act_layer)
        )
        self.transition_2 = transition_layers(stem_channels*8, act_layer=act_layer)

        self.conv_block_3 = nn.Sequential(
            conv_block(stem_channels*4, act_layer=act_layer),
            conv_block(stem_channels*8, act_layer=act_layer)
        )
        self.transition_3 = transition_layers(stem_channels*16, act_layer=act_layer)
        
    def forward(self, x):
        x = self.stem_layers(x)
        x1 = x

        x = self.conv_block_1(x)
        x = self.transition_1(x)
        x2 = x

        x = self.conv_block_2(x)
        x = self.transition_2(x)
        x3 = x

        x = self.conv_block_3(x)
        x = self.transition_3(x)
        x4 = x
        
        return x1, x2, x3, x4