"""
Network initialization library where you can add or modify any 3D segmentation network.

IIIIIIIIIII   M       M      OOOOOOOOO      PPPPPPPPP   
     I        MM     MM     O         O     P       P  
     I        M M   M M    O           O    P       P  
     I        M  M M  M   O             O   PPPPPPPPP  
     I        M   M   M    O           O    P          
     I        M       M     O         O     P          
IIIIIIIIIII   M       M      OOOOOOOOO      P    

Create on 2024-6-1 Saturday.   

@author: jjhuang and tyler
"""

from monai.networks.nets import BasicUNet
from monai.networks.nets.unet import UNet as Monai_UNet
from monai.networks.nets import VNet as Monai_VNet
from monai.networks.nets import SegResNet
from monai.networks.nets import UNETR
from highresnet import HighRes3DNet
from networks.UXNet_3D.network_backbone import UXNET
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from networks.SwinUNETR.SwinUNETR import SwinUNETR
from networks.SASAN.SASAN import SASAN
from networks.uC.uC_3DUNet import uC_3DUNet
from networks.uC.uC_SegResNet import uC_SegResNet        
from networks.uC.uC_TransBTS import uC_TransBTS
from networks.uC.uC_SwinUNETR import uC_SwinUNETR
from networks.uC.uC_3DUXNET import uC_3DUXNET

def get3dmodel(network, in_channel, out_classes):
    ## UNet
    if network == 'UNet':
        model = BasicUNet(in_channels=in_channel, out_channels=out_classes)
        
    ## Monai_UNet
    elif network == 'Monai_Unet':
        model = Monai_UNet(
            spatial_dims=3, 
            in_channels=in_channel, 
            out_channels=out_classes, 
            channels=(16, 32, 64, 128, 256), 
            strides=(2, 2, 2, 2))
        
    ## VNet
    elif network == 'Vnet':
        model = Monai_VNet(
            spatial_dims=3,
            in_channels=in_channel,
            out_channels=out_classes)
    
    ## SegResNet
    elif network == 'SegResNet':
        model = SegResNet(
            spatial_dims=3,
            in_channels=in_channel,
            out_channels=out_classes,
            init_filters=16,
            dropout_prob=0.5)
        
    ## UNETR
    elif network == 'UNETR':
        model = UNETR(
            in_channels=in_channel,
            out_channels=out_classes,
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0)
        
    ## HighRes3DNet
    elif network == 'HighRes3DNet':
        model = HighRes3DNet(
            in_channels=in_channel, 
            out_channels=out_classes)
        
    ## 3DUXNET
    elif network == '3DUXNET':
        model = UXNET(
            in_chans=in_channel,
            out_chans=out_classes,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3)
  
    ## nnFormer
    elif network == 'nnFormer':
        model = nnFormer(
            input_channels=in_channel, 
            num_classes=out_classes)      
        
    ## TransBTS
    elif network == 'TransBTS':
        _, model = TransBTS(img_dim=96,num_classes = out_classes , _conv_repr=True, _pe_type='learned')
        
    ## SwinUNETR 
    elif network == 'SwinUNETR':
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=in_channel,
            out_channels=out_classes,
            feature_size=48,
            use_checkpoint=False)
        
    ## SASAN 
    # mixed precision training is banned with SASAN
    elif network=="SASAN": 
        model = SASAN(
            in_channels = in_channel,
            out_channels = out_classes,
            depths=[2, 2, 2, 2],
            features= [32, 64, 128, 256, 512, 32],
            drop_path_rate=0,
            hidden_size= 512,
            layer_scale_init_value=1e-6)        

    elif network=="uC_3DUNet":
        model = uC_3DUNet(
            in_channels = in_channel,
            out_channels = out_classes,
            depths=[2, 2, 2, 2],
            features= [24, 48, 96, 192, 384, 32],
            drop_path_rate=0,
            hidden_size= 512,
            layer_scale_init_value=1e-6)
       
    elif network=="uC_SegResNet":
        model = uC_SegResNet(
            spatial_dims=3,
            in_channels=in_channel,
            out_channels=out_classes,
            init_filters=32,
            dropout_prob=None)
         
    elif network=="uC_TransBTS":
        _, model = uC_TransBTS(img_dim=96, num_classes=out_classes, _conv_repr=True, _pe_type='learned')
        
    elif network=="uC_SwinUNETR":
        model = uC_SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=in_channel,
            out_channels=out_classes,
            feature_size=48,
            use_checkpoint=False)
        
    elif network=="uC_3DUXNET":
        model = uC_3DUXNET(
            in_chans=in_channel,
            out_chans=out_classes,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3)
        
    return model
