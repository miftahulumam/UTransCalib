import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from models.realignment_layer import realignment_layer
import torch.nn.functional as F

class hydra_channel_attention(nn.Module):
    def __init__(self, 
                 in_dim, 
                 emb_dim = None,
                 qkv_bias=False, 
                 norm_layer=nn.LayerNorm, 
                 attn_drop=0., proj_drop=0.):
        super(hydra_channel_attention, self).__init__()

        self.norm_layer = norm_layer(in_dim)

        self.emb_dim = in_dim if emb_dim is None else emb_dim

        self.Wq_Wk_Wv = nn.Linear(in_dim, self.emb_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(self.emb_dim, in_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, N, C = x.shape

        print(B, N, C)

        x = self.norm_layer(x)
        # print(x.shape)
        qkv = self.Wq_Wk_Wv(x).reshape(B, N, C, 3, 1).permute(3, 0, 2, 4, 1)
        q, k, v = qkv[0], qkv[1], qkv[2]

        phi_q = nn.functional.normalize(q, dim=1)
        phi_k = nn.functional.normalize(k, dim=1)

        kv = torch.sum(phi_k.mul(v), dim=-1)

        attn = phi_q*torch.unsqueeze(kv, dim=-1).expand_as(phi_q)
        # print(attn.shape)
        attn = torch.squeeze(attn).permute(0, 2, 1)

        x = self.proj(attn)
        x = self.proj_drop(x)

        return x
    
class linear_channel_attention(nn.Module):
    def __init__(self, 
                 in_dim, 
                 emb_dim = None,
                 qkv_bias=False, 
                 norm_layer=nn.LayerNorm, 
                 attn_drop=0., proj_drop=0.):
        super(linear_channel_attention, self).__init__()

        self.norm_layer = norm_layer(in_dim)

        self.emb_dim = in_dim if emb_dim is None else emb_dim

        self.Wq_Wk_Wv = nn.Linear(in_dim, self.emb_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(self.emb_dim, in_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        x = self.norm_layer(x)
        # print(x.shape)
        qkv = self.Wq_Wk_Wv(x).reshape(B, N, C, 3).permute(3, 0, 1, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]

        phi_q = torch.softmax(q, dim=2)
        phi_k = torch.softmax(k, dim=1)

        kv = torch.bmm(phi_k.transpose(1, 2), v)
        attn = torch.bmm(phi_q, kv)

        print(kv.shape, attn.shape)

        x = self.proj(attn)
        x = self.proj_drop(x)
        
        return attn
    
class spatial_attention(nn.Module):
    def __init__(self, in_ch, reduction, act_layer=nn.SiLU):
        super(spatial_attention, self).__init__()

        mid_ch = in_ch // reduction
        
        # reduction conv
        self.pointwise_in = nn.Sequential(nn.Conv2d(in_ch, mid_ch,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    groups=1,
                                                    bias=False),
                                          nn.BatchNorm2d(mid_ch),
                                          act_layer(inplace=True))
        
        # attns
        self.attn_branch_1 = nn.Sequential(nn.Conv2d(mid_ch, mid_ch,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     dilation=1,
                                                     groups=mid_ch,
                                                     bias=False),
                                          nn.BatchNorm2d(mid_ch),
                                          act_layer(inplace=True))
        
        self.attn_branch_2 = nn.Sequential(nn.Conv2d(mid_ch, mid_ch,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=2,
                                                     dilation=2,
                                                     groups=mid_ch,
                                                     bias=False),
                                          nn.BatchNorm2d(mid_ch),
                                          act_layer(inplace=True))
        
        self.attn_branch_3 = nn.Sequential(nn.Conv2d(mid_ch, mid_ch,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=3,
                                                     dilation=3,
                                                     groups=mid_ch,
                                                     bias=False),
                                          nn.BatchNorm2d(mid_ch),
                                          act_layer(inplace=True))
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # expansion conv
        self.pointwise_out = nn.Sequential(nn.Conv2d(mid_ch, in_ch,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    groups=1,
                                                    bias=False),
                                           nn.BatchNorm2d(in_ch),
                                           act_layer(inplace=True))
        
        # attns weights
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        attn = x

        attn = self.pointwise_in(attn)
        attn = (self.attn_branch_1(attn) + 
                self.attn_branch_2(attn) + 
                self.attn_branch_3(attn) +
                self.maxpool(attn) + 
                self.avgpool(attn))
        attn = self.pointwise_out(attn)
        attn = self.sigmoid(attn)

        x = attn*x + x

        return x