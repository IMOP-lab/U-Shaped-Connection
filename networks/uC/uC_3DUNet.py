import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
import pytorch_lightning as pl
from typing import Optional, Sequence, Union
import einops
from torch.nn import Softmax
from typing import Tuple
from collections import OrderedDict
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep

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
    def __init__(self, in_cha=3, out_cha=3, features=[32,32,32], bilinear=True, first = False, final = False):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.first = first
        self.final = final

        factor = 2 if bilinear else 1

        if self.first:
            self.inc = (DoubleConv2d(in_cha, features[0]))
        in_channels = features[0]
        # Creating the downsampling layers
        for feature in features[1:]:
            self.downs.append(Down2d(in_channels, feature))
            in_channels = feature

        # Creating the upsampling layers
        for feature in reversed(features[:-1]):
            # print(in_channels,feature//factor)
            self.ups.append(Up2d(in_channels, feature//factor))
            in_channels = feature

        if self.final:
            self.outc = (OutConv2d(features[0], out_cha))

    def forward(self, x):
        skip_connections = []
        if self.first:
            x = self.inc(x)

        for down in self.downs:
            skip_connections.append(x)
            x = down(x)
            # print('x1:',x.shape)

        for up, skip in zip(self.ups, reversed(skip_connections)):
            # print(x.shape,skip.shape)
            x = up(x, skip)
            # print('x2:',x.shape)

        if self.final:
            x = self.outc(x)

        return x

class DFi(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv1 = nn.Conv3d(dim * 2, dim, kernel_size=1, bias=False)
        self.conv2 = nn.Conv3d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv3 = nn.Conv3d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        output = torch.cat([x1, x2], dim=1)
        output = self.conv1(output)
        att = self.conv2(x1) + self.conv3(x2)
        output = output * self.sigmoid(att)
        return output

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return self._channels_last_norm(x)
        elif self.data_format == "channels_first":
            return self._channels_first_norm(x)
        else:
            raise NotImplementedError("Unsupported data_format: {}".format(self.data_format))

    def _channels_last_norm(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def _channels_first_norm(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

class TwoConv(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        super().__init__()

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)

class Down(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        feat :int = 96,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)

class UpCat(nn.Module):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.halves = halves
        if self.halves == False:
            self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)
        else :
            self.dfi = DFi(out_chns)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            
            if self.halves:
                x = self.dfi(x_0,x_e)
            else:
                x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x

class uC_skip(nn.Module):
    def __init__(self,in_channels,out_channels,channels,bilinear,first,final):  # feat==channel
        super().__init__()
        self.unet = UNet2d(in_channels, out_channels, channels, bilinear, first, final)

    def forward(self, x):
        sp = x.shape[-1]
        x1 = einops.rearrange(x, 'B C D H W -> (B W) C D H')
        x1 =  self.unet(x1)
        x1 = einops.rearrange(x1, '(B W) C D H -> B C D H W',W = sp)
        return x1

class uC_3DUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 64, 128, 256, 512, 32),  # list of feature maps
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        depths=[2, 2, 2, 2],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 512, # it It needs to be the same size as the largest feature map
        conv_block: bool = True,
        res_block: bool = True,
        norm_name: Union[Tuple, str] = "instance",
        dimensions: Optional[int] = None,
    ):
        super().__init__()
        
        if dimensions is not None:
            spatial_dims = dimensions
        fea = ensure_tuple_rep(features, 6)
        self.feat_size=fea

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout,feat=96)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout,feat=48)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout,feat=24)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout,feat=12)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample,halves = False)   # By default, layer 1 does not use DFi
        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)


        self.uC_skip0 = uC_skip(in_channels=fea[0],out_channels=fea[0],channels=(fea[0],fea[1],fea[2],fea[3],fea[4]),bilinear=False,first=False,final=False)
        self.uC_skip1 = uC_skip(in_channels=fea[1],out_channels=fea[1],channels=(fea[1],fea[2],fea[3],fea[4]),bilinear=False,first=False,final=False)
        self.uC_skip2 = uC_skip(in_channels=fea[2],out_channels=fea[2],channels=(fea[2],fea[3],fea[4]),bilinear=False,first=False,final=False)

    def forward(self, x: torch.Tensor):

        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        x0 = self.uC_skip0(x0)
        x1 = self.uC_skip1(x1)
        x2 = self.uC_skip2(x2)

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)
        
        logits = self.final_conv(u1)

        return logits