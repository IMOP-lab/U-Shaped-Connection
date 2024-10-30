#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:04:06 2022

@author: leeh43
"""

from typing import Tuple

import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Union
import torch.nn.functional as F
from lib.utils.tools.logger import Logger as Log
from lib.models.tools.module_helper import ModuleHelper
from networks.UXNet_3D.uxnet_encoder import uxnet_conv
import torch
import einops

class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(out_channels//2,out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(out_channels//2,out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.AvgPool2d(2),
            DoubleConv2d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up2d(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2d(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv2d(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat((x2, x1), dim=1)
        return self.conv(x)

class OutConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)

class UNet2d(nn.Module):
    def __init__(self, in_cha=3, out_cha=3, features=[32,32,32], bilinear=True,first= False):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.first = first

        factor = 2 if bilinear else 1

        if self.first:
            self.inc = (DoubleConv2d(in_cha, features[0]))
        in_channels = features[0]
        # Creating the downsampling layers
        for feature in features[1:]:
            # print('1')
            self.downs.append(Down2d(in_channels, feature))
            in_channels = feature

        # Creating the upsampling layers
        for feature in reversed(features[:-1]):
            # print(in_channels,feature//factor)
            self.ups.append(Up2d(in_channels, feature//factor))
            in_channels = feature

        self.outc = (OutConv2d(features[0], out_cha))

    def forward(self, x):
        skip_connections = []
        if self.first:
            x = self.inc(x)
        for down in self.downs:
            skip_connections.append(x)
            x = down(x)

        for up, skip in zip(self.ups, reversed(skip_connections)):
            # print(x.shape,skip.shape)
            x = up(x, skip)

        x = self.outc(x)

        return x
        
class reshape_unet(nn.Module):
    def __init__(self,in_channels,out_channels,channels,first):#feat==channel
        super().__init__()
        self.unet = UNet2d(in_channels, out_channels, channels, False,first)

    def forward(self, x):
        sp = x.shape[-1]
        # x = einops.rearrange(x, 'B C D H W -> B (D H W) C') 
        x1 = einops.rearrange(x, 'B C D H W -> (B W) C D H')
        x1 =  self.unet(x1)
        x1 = einops.rearrange(x1, '(B W) C D H -> B C D H W',W = sp)
        # print(x1.shape,x.shape)
        return x1

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()

        Log.info('proj_dim: {}'.format(proj_dim))

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv3d(dim_in, dim_in, kernel_size=1),
                ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                nn.Conv3d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)

        
class UXNET2d(nn.Module):

    def __init__(
        self,
        in_chans=1, 
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:

        super().__init__()

        self.hidden_size = hidden_size

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.spatial_dims = spatial_dims
        self.uxnet_3d = uxnet_conv(
            in_chans= self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.encoder1 = reshape_unet(self.in_chans,self.feat_size[0],[self.feat_size[0],self.feat_size[1],self.feat_size[2],self.feat_size[3]],True)
        self.encoder2 = reshape_unet(self.feat_size[0], self.feat_size[1],[self.feat_size[0],self.feat_size[1],self.feat_size[2],self.feat_size[3]],False)
        self.encoder3 = reshape_unet(self.feat_size[1], self.feat_size[2],[self.feat_size[1],self.feat_size[2],self.feat_size[3]],False)
        self.encoder4 = reshape_unet(self.feat_size[2], self.feat_size[3],[self.feat_size[2],self.feat_size[3]],False)

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)
        # self.conv_proj = ProjectionHead(dim_in=hidden_size)


    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
    
    def forward(self, x_in):
        outs = self.uxnet_3d(x_in)
        # print("x_in",x_in.shape)
        enc1 = self.encoder1(x_in)
        # print("enc1",enc1.shape)
        x2 = outs[0]
        # print("x2",x2.shape)
        enc2 = self.encoder2(x2)
        # print("enc2",enc2.shape)

        x3 = outs[1]
        # print("x3",x3.shape)
        enc3 = self.encoder3(x3)
        # print("enc3",enc3.shape)
        x4 = outs[2]
        # print("x4",x4.shape)
        enc4 = self.encoder4(x4)
        # print("enc4",enc4.shape)
        enc_hidden = self.encoder5(outs[3])


        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)
        
        # feat = self.conv_proj(dec4)
        
        return self.out(out)