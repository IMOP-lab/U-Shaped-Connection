"""
Code for testing on multiple GPUs.

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

import torch
import os
import numpy as np
import pytz
import argparse
import json
import datetime
import shutil
import random
import nibabel as nib
import concurrent.futures
from torch.utils.data.distributed import DistributedSampler
from scipy.ndimage import zoom 
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch
from dataloader.load_datasets_transforms import data_loader, data_transforms, infer_post_transforms
from model import get3dmodel
from config import misc
from colorama import Fore, Style

parser = argparse.ArgumentParser(description='inference hyperparameters for 3D image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='', required=True, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='', required=True, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='flare', required=True, help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')
## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='3DUXNET', required=True, help='Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET}')
parser.add_argument('--trained_weights', default='', required=True, help='Path of pretrained/fine-tuned weights')
parser.add_argument('--mode', type=str, default='test', help='Training or testing mode')
parser.add_argument('--sw_batch_size', type=int, default=4, help='Sliding window batch size for inference')
parser.add_argument('--overlap', type=float, default=0.5, help='Sub-volume overlapped percentage')
## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
parser.add_argument('--out_classes', type=int, default=4, help='Number of class')
parser.add_argument('--in_channel', type=int, default=1, help='Number of in_channel')
parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
parser.add_argument('--local_rank', type=int, help='local rank for dist')
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--testrank', nargs='+', type=int, help='input a list of integers')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

test_samples, out_classes = data_loader(args)   # test sets and classes
test_files = [
    {"image": image_name} for image_name in zip(test_samples['images'])
]

test_transforms = data_transforms(args)
post_transforms = infer_post_transforms(args, test_transforms, out_classes)

testrank=args.testrank
misc.init_distributed_mode(args)
print("args: " + str(args) + '\n')
if (args.distributed):
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(testrank[local_rank])
    args.gpu = torch.device("cuda", testrank[local_rank])
    print('local_rank: {}'.format(args.local_rank))
    print('world size: {}'.format(args.world_size))

## set seed
set_determinism(seed=0) 
seed = args.seed + misc.get_rank()
torch.manual_seed(seed)
np.random.seed(seed) 
random.seed(seed) 

test_ds = CacheDataset(
    data=test_files, transform=test_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)   # memory cache
test_sampler = DistributedSampler(test_ds, shuffle=False)   # distributed reading of test data
batch_sampler_val = torch.utils.data.BatchSampler(
            test_sampler, 1, drop_last=False)   # create a batch for test data
test_loader = DataLoader(test_ds, batch_sampler=batch_sampler_val, num_workers=args.num_workers, pin_memory=False)  # dataloader

## Load Networks
model = get3dmodel(args.network, in_channel=args.in_channel, out_classes=args.out_classes)
    
## get Beijing time
def get_beijing_time():
    beijing_tz = pytz.timezone('Asia/Shanghai')
    beijing_time = datetime.datetime.now(beijing_tz)
    return beijing_time.strftime("%Y-%m-%d %H:%M:%S")

## test starts, record time
start_time = get_beijing_time()
print(f"Test start time: {start_time}")

## remove the 'module.' at the beginning of the model weight.
weights=torch.load(args.trained_weights,map_location=torch.device('cuda'))
weights_dict = {}
for k, v in weights.items():    # accessing the model's k and v
    if k.startswith('module.'):
        new_k = k[len('module.'):]
    else:
        new_k = k
    weights_dict[new_k] = v
model.load_state_dict(weights_dict)
model.eval()    # Set the model to evaluation mode, disable the Dropout layer, and make BN use the running mean and variance calculated during the entire training period instead of the current batch.

model = model.to(args.gpu)
model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
print(Fore.YELLOW + '################Chosen Network Architecture: {}'.format(args.network)  + Style.RESET_ALL)

with torch.no_grad():   # disable gradient calculation
    for i, test_data in enumerate(test_loader):
        images = test_data["image"].cuda()
        roi_size = (96, 96, 96)
        test_data['pred'] = sliding_window_inference(
            images, roi_size, args.sw_batch_size, model, overlap=args.overlap
        )
        test_data = [post_transforms(i) for i in decollate_batch(test_data)]

### modify test result file location

# specify input and output paths
folder = args.output
output_folder = os.path.join(args.output,'../final/')
os.makedirs(output_folder, exist_ok=True)
json_path=os.path.join(args.root,'shapes.json')

# read shapes.json
with open(json_path, 'r', encoding='utf-8') as f:
    shapes = json.load(f)

# rename all files in test
def rename_files(filename):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    if filename.endswith('.nii.gz'):
        name, ext = os.path.splitext(filename)
        new_name = name.replace('_seg', '') + ext
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        os.rename(src, dst)

# iterate through all subfolders and extract all files to the parent folder directory
def move_and_rename_files(subdir):
    subdir_path = os.path.join(folder, subdir)
    if os.path.isdir(subdir_path):
        # remove all .nii.gz files from the subfolder to the parent folder
        for f in os.listdir(subdir_path):
            if f.endswith('.nii.gz'):
                f_path = os.path.join(subdir_path, f)
                shutil.move(f_path, folder)
                # rename file
                rename_files(f)

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(move_and_rename_files, os.listdir(folder))
    
# convert the .nii.gz file back to its original shape
def restore_nii(filename):
    if filename.endswith('.nii.gz'):
        input_path = os.path.join(folder, filename)
        output_path = os.path.join(output_folder, filename)
        original_shape = shapes[filename]
        # print(filename,original_shape)
        if original_shape:
            img = nib.load(input_path)
            scale = [o / n for o, n in zip(original_shape, img.shape)]
            data_restored = zoom(img.get_fdata(), zoom=scale,order=0)
            new_img = nib.Nifti1Image(data_restored, img.affine, img.header)
            nib.save(new_img, output_path)

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(restore_nii, os.listdir(folder))

# delete all empty subfolders
for subdir in os.listdir(folder):
    subdir_path = os.path.join(folder, subdir)
    if os.path.isdir(subdir_path):
        try:
            os.rmdir(subdir_path)
        except OSError:
            pass
        
# the test is over, record the time
end_time = get_beijing_time()
print(f"Test end time: {end_time}")