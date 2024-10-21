import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from models.realignment_layer import realignment_layer
import torch.nn.functional as F

from .basic_blocks import SingleConv, DoubleConv, Residual_DoubleConv
from .basic_blocks import upsampling, Residual_upsampling

class decoder(nn.Module):
    def __init__(self, 
                 in_channels=[64, 128, 256, 512], # from large dimension to small dimension
                 bilinear=True, 
                 depthwise=True, 
                 activation='nn.SiLU(inplace=True)', 
                 drop_rate=0.1):
        super(decoder, self).__init__()

        self.n_stage = len(in_channels) - 1

        self.up = nn.ModuleList()
        for i in range(self.n_stage):
            self.up.append(upsampling(
                            in_channels[-1-i] + in_channels[-2-i], 
                            in_channels[-2-i],
                            bilinear, depthwise, activation, drop_rate))
            
    def forward(self, input):
        assert len(input) == self.n_stage + 1

        output = [input[-1]] 
        small_map = input[-1]

        for i, up in enumerate(self.up):
            large_map = input[-2-i]
            out = up(small_map, large_map)
            output.insert(0, out)

            small_map = out
        
        assert len(input) == len(output)

        return output

class decoder_full(nn.Module):
    def __init__(self, 
                 in_ch, out_ch=None, 
                 mode='default', 
                 bilinear=True, 
                 depthwise=True, 
                 activation='nn.SiLU(inplace=True)', 
                 drop_rate=0.1):
        super(decoder_full, self).__init__()

        factor = 2 if bilinear else 1
        
        if out_ch is None:
            out_ch = in_ch // (4 * factor)

        if mode == 'residual':
            self.up1 = Residual_upsampling(in_ch + in_ch // factor, 
                                           in_ch // factor, 
                                           bilinear, depthwise, activation, drop_rate)
            self.up2 = Residual_upsampling(in_ch // factor + in_ch // (2 * factor), 
                                           in_ch // (2 * factor), 
                                           bilinear, depthwise, activation, drop_rate)
            self.up3 = Residual_upsampling(in_ch // (2 * factor) + out_ch, 
                                           out_ch, 
                                           bilinear, depthwise, activation, drop_rate)
        else:
            self.up1 = upsampling(in_ch + in_ch // factor, 
                                  in_ch // factor, 
                                  bilinear, depthwise, activation, drop_rate)
            self.up2 = upsampling(in_ch // factor + in_ch // (2 * factor), 
                                  in_ch // (2 * factor), 
                                  bilinear, depthwise, activation, drop_rate)
            self.up3 = upsampling(in_ch // (2 * factor) + out_ch, 
                                  out_ch, 
                                  bilinear, depthwise, activation, drop_rate)

    def forward(self, x1, x2, x3, x4):
        xd = x4
        xc = self.up1(x4, x3)
        xb = self.up2(xc, x2)
        xa = self.up3(xb, x1)

        return xa, xb, xc, xd
    
class decoder_3_stage(nn.Module):
    def __init__(self, 
                 in_ch, out_ch=None, 
                 mode='default', 
                 bilinear=True, 
                 depthwise=True, 
                 activation='nn.SiLU(inplace=True)', 
                 drop_rate=0.1):
        super(decoder_3_stage, self).__init__()

        factor = 2 if bilinear else 1
        
        if out_ch is None:
            out_ch = in_ch // (2 * factor)

        if mode == 'residual':
            self.up1 = Residual_upsampling(in_ch + in_ch // factor, 
                                           in_ch // factor, 
                                           bilinear, depthwise, activation, drop_rate)
            self.up2 = Residual_upsampling(in_ch // factor + out_ch, 
                                           out_ch, 
                                           bilinear, depthwise, activation, drop_rate)
        else:
            self.up1 = upsampling(in_ch + in_ch // factor, 
                                  in_ch // factor, 
                                  bilinear, depthwise, activation, drop_rate)
            self.up2 = upsampling(in_ch // factor + out_ch, 
                                  out_ch, 
                                  bilinear, depthwise, activation, drop_rate)

    def forward(self, x2, x3, x4):
        xd = x4
        xc = self.up1(x4, x3)
        xb = self.up2(xc, x2)

        return xb, xc, xd
    
class decoder_2_stage(nn.Module):
    def __init__(self, 
                 in_ch, out_ch=None, 
                 mode='default', 
                 bilinear=True, 
                 depthwise=False, 
                 activation='nn.SiLU(inplace=True)', 
                 drop_rate=0.1):
        super(decoder_2_stage, self).__init__()

        factor = 2 if bilinear else 1
        
        if out_ch is None:
            out_ch = in_ch // factor

        if mode == 'residual':
            self.up1 = Residual_upsampling(in_ch + out_ch, 
                                           out_ch, 
                                           bilinear, depthwise, activation, drop_rate)
        else:
            self.up1 = upsampling(in_ch + out_ch, 
                                  out_ch, 
                                  bilinear, depthwise, activation, drop_rate)

    def forward(self, x3, x4):
        xd = x4
        xc = self.up1(x4, x3)

        return xc, xd

class decoder_full_transformer(nn.Module):
    def __init__(self, 
                 in_ch, out_ch=None, 
                 mode='default', 
                 bilinear=True, 
                 depthwise=False, 
                 activation='nn.SiLU(inplace=True)', 
                 drop_rate=0.1):
        super(decoder_full_transformer, self).__init__()

        factor = 2 if bilinear else 1
        
        if out_ch is None:
            out_ch = in_ch // (4 * factor)

        if mode == 'residual':
            self.up1 = Residual_upsampling(in_ch + in_ch // factor, 
                                           in_ch // factor, 
                                           bilinear, depthwise, activation, drop_rate)
            self.up2 = Residual_upsampling(in_ch // factor + in_ch // (2 * factor), 
                                           in_ch // (2 * factor), 
                                           bilinear, depthwise, activation, drop_rate)
            self.up3 = Residual_upsampling(in_ch // (2 * factor) + out_ch, 
                                           out_ch, 
                                           bilinear, depthwise, activation, drop_rate)
        else:
            self.up1 = upsampling(in_ch + in_ch // factor, 
                                  in_ch // factor, 
                                  bilinear, depthwise, activation, drop_rate)
            self.up2 = upsampling(in_ch // factor + in_ch // (2 * factor), 
                                  in_ch // (2 * factor), 
                                  bilinear, depthwise, activation, drop_rate)
            self.up3 = upsampling(in_ch // (2 * factor) + out_ch, 
                                  out_ch, 
                                  bilinear, depthwise, activation, drop_rate)
            
        # self.trans1 = TransBlock()
        # self.trans2 = TransBlock()
        # self.trans3 = TransBlock()

    def forward(self, x1, x2, x3, x4):
        x1 = self.trans1(x1)
        x2 = self.trans2(x2)
        x3 = self.trans3(x3)

        xd = x4
        xc = self.up1(x4, x3)
        xb = self.up2(xc, x2)
        xa = self.up3(xb, x1)

        return xa, xb, xc, xd

