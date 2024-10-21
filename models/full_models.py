import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from models.realignment_layer import realignment_layer
import torch.nn.functional as F

from dotwiz import DotWiz

from .encoders import (encoder_densenet, 
                       encoder_resnet, 
                       encoder_mobilenet_small,
                       encoder_lite)

from .decoders import (decoder,
                       decoder_full, 
                       decoder_3_stage, 
                       decoder_2_stage)

from .fusions import (feature_fusion_full, 
                      feature_fusion_3maps, 
                      feature_fusion_2maps)

from .heads import global_regression_v1

class UTransCalib_model_resnet(nn.Module):
    def __init__(self, model_config):
        super(UTransCalib_model_resnet, self).__init__()

        activation = model_config.activation
        init_weights = model_config.init_weights

        self.rgb_encd = encoder_resnet(pretrained=True)
        self.depth_encd = encoder_resnet(pretrained=False, depth_branch=True)

        self.rgb_decdr = decoder_3_stage(in_ch=512, depthwise=True, activation=activation)
        self.depth_dcdr = decoder_3_stage(in_ch=512, depthwise=True, activation=activation)

        self.fusion_module = feature_fusion_3maps(max_channel=1024, 
                                                  fc_reduction=16,
                                                  depthwise=True,
                                                  activation=activation)
        
        self.global_regression = global_regression_v1(max_channel=1024,
                                                      depthwise=True)
        
        self.recalib = realignment_layer()

        if init_weights:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, rgb_im, depth_im, pcd_mis, T_mis_batch):
        _, x2_rgb, x3_rgb, x4_rgb = self.rgb_encd(rgb_im)
        _, x2_depth, x3_depth, x4_depth = self.depth_encd(depth_im)

        x2_rgb, x3_rgb, x4_rgb = self.rgb_decdr(x2_rgb, x3_rgb, x4_rgb)
        x2_depth, x3_depth, x4_depth = self.depth_dcdr(x2_depth, x3_depth, x4_depth)

        x2 = torch.cat((x2_rgb, x2_depth), dim=1)
        x3 = torch.cat((x3_rgb, x3_depth), dim=1)
        x4 = torch.cat((x4_rgb, x4_depth), dim=1)

        x_fused = self.fusion_module(x2, x3, x4)
        delta_t_pred, delta_q_pred = self.global_regression(x_fused)

        if delta_q_pred.ndim < 2:
            delta_q_pred = torch.unsqueeze(delta_q_pred, 0)
            delta_t_pred = torch.unsqueeze(delta_t_pred, 0)

        delta_q_pred = nn.functional.normalize(delta_q_pred)
        
        batch_T_pred, pcd_pred = self.recalib(pcd_mis, T_mis_batch, delta_q_pred, delta_t_pred)

        return pcd_pred, batch_T_pred, delta_q_pred, delta_t_pred

class UTransCalib_model_densenet(nn.Module):
    def __init__(self, model_config):
        super(UTransCalib_model_densenet, self).__init__()

        activation = model_config.activation
        init_weights = model_config.init_weights

        self.rgb_encd = encoder_densenet(pretrained=True)
        self.depth_encd = encoder_densenet(pretrained=False, depth_branch=True)

        self.rgb_decdr = decoder_full(in_ch=1024, depthwise=True, activation=activation)
        self.depth_dcdr = decoder_full(in_ch=1024, depthwise=True, activation=activation)

        self.fusion_module = feature_fusion_full(max_channel=2048, 
                                                 fc_reduction=16,
                                                 depthwise=True)
        
        self.global_regression = global_regression_v1(max_channel=2048,
                                                      depthwise=True)
        
        self.recalib = realignment_layer()

        if init_weights:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, rgb_im, depth_im, pcd_mis, T_mis_batch):
        x1_rgb, x2_rgb, x3_rgb, x4_rgb = self.rgb_encd(rgb_im)
        x1_depth, x2_depth, x3_depth, x4_depth = self.depth_encd(depth_im)

        x1_rgb, x2_rgb, x3_rgb, x4_rgb = self.rgb_decdr(x1_rgb, x2_rgb, x3_rgb, x4_rgb)
        x1_depth, x2_depth, x3_depth, x4_depth = self.depth_dcdr(x1_depth, x2_depth, x3_depth, x4_depth)

        x1 = torch.cat((x1_rgb, x1_depth), dim=1)
        x2 = torch.cat((x2_rgb, x2_depth), dim=1)
        x3 = torch.cat((x3_rgb, x3_depth), dim=1)
        x4 = torch.cat((x4_rgb, x4_depth), dim=1)

        x_fused = self.fusion_module(x1, x2, x3, x4)
        delta_t_pred, delta_q_pred = self.global_regression(x_fused)

        if delta_q_pred.ndim < 2:
            delta_q_pred = torch.unsqueeze(delta_q_pred, 0)
            delta_t_pred = torch.unsqueeze(delta_t_pred, 0)

        delta_q_pred = nn.functional.normalize(delta_q_pred)
        
        batch_T_pred, pcd_pred = self.recalib(pcd_mis, T_mis_batch, delta_q_pred, delta_t_pred)

        return pcd_pred, batch_T_pred, delta_q_pred, delta_t_pred


class UTransCalib_model_lite(nn.Module):
    def __init__(self, model_config):
        super(UTransCalib_model_lite, self).__init__()

        activation = model_config.activation
        init_weights = model_config.init_weights

        self.rgb_encd = encoder_lite(act_layer=activation)
        self.depth_encd = encoder_lite(depth_branch=True, act_layer=activation)

        self.rgb_decdr = decoder_3_stage(in_ch=512, depthwise=True, activation=activation)
        self.depth_dcdr = decoder_3_stage(in_ch=512, depthwise=True, activation=activation)

        self.fusion_module = feature_fusion_3maps(max_channel=1024, 
                                                  fc_reduction=16,
                                                  depthwise=True,
                                                  activation=activation)
        
        self.global_regression = global_regression_v1(max_channel=1024,
                                                      depthwise=True)
        
        self.recalib = realignment_layer()

        if init_weights:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, rgb_im, depth_im, pcd_mis, T_mis_batch):
        _, x2_rgb, x3_rgb, x4_rgb = self.rgb_encd(rgb_im)
        _, x2_depth, x3_depth, x4_depth = self.depth_encd(depth_im)

        x2_rgb, x3_rgb, x4_rgb = self.rgb_decdr(x2_rgb, x3_rgb, x4_rgb)
        x2_depth, x3_depth, x4_depth = self.depth_dcdr(x2_depth, x3_depth, x4_depth)

        x2 = torch.cat((x2_rgb, x2_depth), dim=1)
        x3 = torch.cat((x3_rgb, x3_depth), dim=1)
        x4 = torch.cat((x4_rgb, x4_depth), dim=1)

        x_fused = self.fusion_module(x2, x3, x4)
        delta_t_pred, delta_q_pred = self.global_regression(x_fused)

        if delta_q_pred.ndim < 2:
            delta_q_pred = torch.unsqueeze(delta_q_pred, 0)
            delta_t_pred = torch.unsqueeze(delta_t_pred, 0)

        delta_q_pred = nn.functional.normalize(delta_q_pred)
        
        batch_T_pred, pcd_pred = self.recalib(pcd_mis, T_mis_batch, delta_q_pred, delta_t_pred)

        return pcd_pred, batch_T_pred, delta_q_pred, delta_t_pred
    
class UTranscalib_densenet_attn(nn.Module):
    def __init__(self, model_config):
        super(UTranscalib_densenet_attn, self).__init__()

        activation = model_config.activation
        init_weights = model_config.init_weights

        self.rgb_encd = encoder_densenet(pretrained=True)
        self.depth_encd = encoder_densenet(pretrained=False, depth_branch=True)

        self.rgb_decdr = decoder_3_stage(in_ch=1024, depthwise=True, activation=activation)
        self.depth_dcdr = decoder_3_stage(in_ch=1024, depthwise=True, activation=activation)

        #####

        if init_weights:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, rgb_im, depth_im, pcd_mis, T_mis_batch):
        _, x2_rgb, x3_rgb, x4_rgb = self.rgb_encd(rgb_im)
        _, x2_depth, x3_depth, x4_depth = self.depth_encd(depth_im)

        x2_rgb, x3_rgb, x4_rgb = self.rgb_decdr(x2_rgb, x3_rgb, x4_rgb)
        x2_depth, x3_depth, x4_depth = self.depth_dcdr(x2_depth, x3_depth, x4_depth)

        x2 = torch.cat((x2_rgb, x2_depth), dim=1)
        x3 = torch.cat((x3_rgb, x3_depth), dim=1)
        x4 = torch.cat((x4_rgb, x4_depth), dim=1)

        return x2, x3, x4

class UTranscalib_mobilenet(nn.Module):
    def __init__(self, model_config):
        super(UTranscalib_mobilenet, self).__init__()

        activation = model_config.activation
        init_weights = model_config.init_weights

        self.rgb_encd = encoder_mobilenet_small(pretrained=True)
        self.depth_encd = encoder_mobilenet_small(pretrained=False, depth_branch=True)

        self.rgb_decdr = decoder(in_channels=[16, 24, 48, 576], activation=activation)
        self.depth_dcdr = decoder(in_channels=[16, 24, 48, 576], activation=activation)

        ########
        
        if init_weights:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, rgb_im, depth_im, pcd_mis, T_mis_batch):
        x1_rgb, x2_rgb, x3_rgb, x4_rgb = self.rgb_encd(rgb_im)
        x1_depth, x2_depth, x3_depth, x4_depth = self.depth_encd(depth_im)

        x1_rgb, x2_rgb, x3_rgb, x4_rgb = self.rgb_decdr([x1_rgb, x2_rgb, x3_rgb, x4_rgb])
        x1_depth, x2_depth, x3_depth, x4_depth = self.depth_dcdr([x1_depth, x2_depth, x3_depth, x4_depth])

        x1 = torch.cat((x1_rgb, x1_depth), dim=1)
        x2 = torch.cat((x2_rgb, x2_depth), dim=1)
        x3 = torch.cat((x3_rgb, x3_depth), dim=1)
        x4 = torch.cat((x4_rgb, x4_depth), dim=1)

        return x1, x2, x3, x4