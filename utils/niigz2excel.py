import argparse
from test_config import jisuan_excel
import os

parser = argparse.ArgumentParser(description='Hyperparameters for 3D medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='', required=True, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='', required=True, help='Output folder for both tensorboard and the best model')
parser.add_argument('--network', type=str, default='uC_3DUNet', help='Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET, uC_3DUNet}')
parser.add_argument('--out_classes', type=int, default=4)
parser.add_argument('--metrics_list', nargs='+', type=str, default=['pre', 'recall', 'spec', 'acc', 'iou', 'dice', 'assd', 'hd', 'hd95', 'voe', 'rand', 'adj_rand'], help='List of metrics to evaluate.')
args = parser.parse_args()

######  generate metrics excel #########
pre_path = os.path.join(args.output,'../final/')
tar_path = os.path.join(args.root,'./original_labelTs/')
res_path = os.path.join(args.output,'../'+args.network+'.xlsx')
jisuan_excel(pre_path,tar_path,res_path,args.metrics_list,args.out_classes)