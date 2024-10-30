from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import InterpolateMode, UpsampleMode
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.upsample import UpSample
from dropblock import DropBlock3D, LinearScheduler
from torch.nn import Softmax
    
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
            # print('1')
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

        for up, skip in zip(self.ups, reversed(skip_connections)):
            # print(x.shape,skip.shape)
            x = up(x, skip)

        if self.final:
            x = self.outc(x)

        return x
        
class uC_skip(nn.Module):
    def __init__(self,in_channels,out_channels,channels,bilinear,first,final):#feat==channel
        super().__init__()
        self.unet = UNet2d(in_channels, out_channels, channels, bilinear, first, final)

    def forward(self, x):
        sp = x.shape[-1]
        x1 = einops.rearrange(x, 'B C D H W -> (B W) C D H')
        x1 =  self.unet(x1)
        x1 = einops.rearrange(x1, '(B W) C D H -> B C D H W',W = sp)
        return x1

class DWConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

def get_conv_layer(
    spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False
):
    return Convolution(
        spatial_dims, in_channels, out_channels, kernel_size=kernel_size, strides=stride, bias=bias, conv_only=True
    )

def get_upsample_layer(
    spatial_dims: int, in_channels: int, upsample_mode: UpsampleMode | str = "nontrainable", scale_factor: int = 2
):
    return UpSample(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=in_channels,
        scale_factor=scale_factor,
        mode=upsample_mode,
        interp_mode=InterpolateMode.LINEAR,
        align_corners=False,
    )

class ResBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: int = 3,
        act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)

        self.act = get_act_layer(act)
        self.conv1 = get_conv_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size
        )
        self.conv2 = get_conv_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size
        )

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity
        # print("x.ResBlock:",x.shape)
        return x
    
class uC_SegResNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,  # dimension(2,3)
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None, # dropout
        act: tuple | str = ("RELU", {"inplace": True}), # activate function
        norm: tuple | str = ("GROUP", {"num_groups": 8}), # Normalization
        blocks_down: tuple = (1, 2, 2, 3),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_prob = dropout_prob
        if dropout_prob is not None:
            self.dropout = nn.Dropout3d(dropout_prob)
            # self.dropout = DropBlock3D(block_size=8, drop_prob=self.dropout_prob)
            # self.dropout = LinearScheduler(
            #     DropBlock3D(drop_prob=dropout_prob,block_size=6),
            #     start_value=0,
            #     stop_value=dropout_prob,
            #     nr_steps=40000
            # )
        self.act = act 
        self.act_mod = get_act_layer(act)
        self.norm = norm
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.upsample_mode = UpsampleMode(upsample_mode)

        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters) 
        self.down_layers = self.make_down_layers()
        self.up_samples, self.up_layers = self.make_up_layers()
        self.conv_final = self.make_final_conv(out_channels)
        self.feat_size = [init_filters, init_filters*2, init_filters*4, init_filters*8]
        self.uc_skip1 = uC_skip(self.in_channels,self.feat_size[0],self.feat_size[0:],False,False,True)
        self.uc_skip2 = uC_skip(self.feat_size[0],self.feat_size[1],self.feat_size[1:],False,False,True)
        self.uc_skip3 = uC_skip(self.feat_size[1],self.feat_size[2],self.feat_size[2:],False,False,True)
        self.uc_skip4 = uC_skip(self.feat_size[2],self.feat_size[3],self.feat_size[3:],False,False,True)

    def make_down_layers(self):  
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm, act = (
            self.blocks_down, 
            self.spatial_dims, 
            self.init_filters, 
            self.norm, self.act
            )   # blocks_down=(1, 2, 2, 4),spatial_dims=3,init_filters=8,norm=GroupNorm,act=RELU
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i # 8,16,32,64
            # print(i,filters,layer_in_channels)
            pre_conv = (
                get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)   # Double the channels, reduce spatial size by half;channels=(8,16,32->16,32,64)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv, *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=act) for _ in range(item)] # number of ResBlock=(1,2,2,4);in_channels=(8,16,32,64),norm=GroupNorm,act=RELU
            )
            down_layers.append(down_layer)
        return down_layers

    def make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up) # upsample 3 times (1,1,1)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i) # (64,32,16)
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1), # channels/2,(64,32,16)->(32,16,8)
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode), # feature map size*2
                    ]
                )
            )
            up_layers.append(
                nn.Sequential(
                    *[
                        ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act) # resBlock threr times
                        for _ in range(blocks_up[i])
                    ]
                )
            )
        return up_samples, up_layers

    def make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        # print(x.shape)
        if self.dropout_prob is not None: 
            x = self.dropout(x)

        down_x = []

        for i,down in enumerate(self.down_layers):
            x = down(x)
            # print("down_x:",x.shape)

            if i == 0:
                uc_skip = self.uc_skip1(x)
                # print("down_x1:",unet2d_x.shape)
            elif i == 1:
                uc_skip = self.uc_skip2(x)
                # print("down_x2:",unet2d_x.shape)
            elif i == 2:
                uc_skip = self.uc_skip3(x)
                # print("down_x3:",unet2d_x.shape)
            elif i == 3:
                uc_skip = self.uc_skip4(x)
                # print("down_x4:",unet2d_x.shape)
                
            # if self.dropout_prob is not None:
            #     x = self.dropout(x)
            down_x.append(uc_skip) 
        # print(down_x[0].shape,down_x[1].shape,down_x[2].shape,down_x[3].shape)
        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)): # up corresponds to the up_samples layer (up sampling), upl corresponds to the up_layers layer (ResBlock)
            x = up(x) + down_x[i + 1] # upsampling, skip connections
            # print("x_upsampe:",x.shape)
            x = upl(x) # ResBlock
            # print("x_ResBlock2:",x.shape)

        x = self.conv_final(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:# assume the input is (4,1,96,96,96); 96->48->24->12->6
        x, down_x = self.encode(x)
        down_x.reverse() # channels->(64,32,16,8)

        x = self.decode(x, down_x)
        return x
    