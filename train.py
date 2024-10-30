"""
Code for training and validation on multiple GPUs.

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
import torch.nn as nn
import numpy as np
import os
import re
import random
import argparse
import logging
import pytz
import einops
import torch.distributed as dist
from tqdm import tqdm
from datetime import datetime
from config import misc
from colorama import Fore, Style
from torch.utils.data.distributed import DistributedSampler
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from torch.cuda.amp import GradScaler, autocast
from dataloader.load_datasets_transforms import data_loader, data_transforms
from model import get3dmodel
from lr_scheduler import LinearWarmupCosineAnnealingLR
from loss.weightedDIceCELoss import weightedDiceCELoss
from loss.DiceLoss import DiceLoss

parser = argparse.ArgumentParser(description='Hyperparameters for 3D image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='', required=True, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='', required=True, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='', required=True, help='Dataset folder for all your datasets here')
## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='', help='Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET}')
parser.add_argument('--mode', type=str, default='train', help='Training or testing mode')
parser.add_argument('--pretrain', default=False, help='Have pretrained weights or not')
parser.add_argument('--pretrained_weights', default='', help='Path of pretrained weights')
parser.add_argument('--batch_size', type=int, default='1', help='Batch size for subject input')
parser.add_argument('--crop_sample', type=int, default='2', help='Number of cropped sub-volumes for each subject')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer types: Adam / AdamW')
parser.add_argument('--max_iter', type=int, default=80000, help='Maximum iteration steps for training')
parser.add_argument('--eval_step', type=int, default=1000, help='Per steps to perform validation')
parser.add_argument('--in_channel', type=int, default=1, help='In_channel of input images')
parser.add_argument('--out_classes', type=int, default='2', help='Batch size for subject input')
## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate to cache your dataset into CPUs')
parser.add_argument('--num_workers', type=int, default=12, help='Number of workers')
parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
parser.add_argument('--local_rank', type=int, help='local rank for dist')
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--testrank', nargs='+', type=int, help='input a list of integers')
parser.add_argument('--amp', action='store_true', help='Enable AMP')
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

train_samples, valid_samples, out_classes = data_loader(args) #Training sets, validation sets, and categories

train_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_samples['images'], train_samples['labels']) # a list saves the training set file path
]

val_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(valid_samples['images'], valid_samples['labels']) # a list saves the validation set file path
]

train_transforms, val_transforms = data_transforms(args)

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
torch.backends.cudnn.benchmark = True

## Train Pytorch Data Loader and Caching
print('Start caching datasets!')
train_ds = CacheDataset(
    data=train_files, transform=train_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)   # memory cache
sampler = DistributedSampler(train_ds, shuffle=True)    # distributed reading of train data
batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, args.batch_size, drop_last=True)   # create a batch for train data
train_loader = DataLoader(train_ds, batch_sampler=batch_sampler_train, num_workers=args.num_workers, pin_memory=True)   # dataloader

## Valid Pytorch Data Loader and Caching
val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
val_sampler = DistributedSampler(val_ds, shuffle=False) # memory cache
batch_sampler_val = torch.utils.data.BatchSampler(
            val_sampler, 1, drop_last=False)    # create a batch 1 to reduce the probability of gpu memory overflow
val_loader = DataLoader(val_ds, batch_sampler=batch_sampler_val, num_workers=args.num_workers, pin_memory=False)

## Load Networks
model = get3dmodel(args.network, in_channel=args.in_channel, out_classes=args.out_classes)

if os.path.exists(os.path.join(args.output,'best_metric_model.pth')):
    print("#########pretrained best metric model load ! ! !##########")
    print(os.path.join(args.output,'best_metric_model.pth'))
    weights=torch.load(os.path.join(args.output,'best_metric_model.pth'),map_location=torch.device('cuda'))
    weights_dict = {}
    for k, v in weights.items():     # accessing the model's k and v
        if k.startswith('module.'):
            new_k = k[len('module.'):] # remove the 'module.' at the beginning of the model weight.
        else:
            new_k = k
        weights_dict[new_k] = v
    model.load_state_dict(weights_dict)
    # model.eval()   # Set the model to evaluation mode, disable the Dropout layer, and make BN use the running mean and variance calculated during the entire training period instead of the current batch.

model = model.to(args.gpu)
model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)  # If you have a variable that is not participating in backpropagation, set find_unused_parameters to True.
print(Fore.YELLOW + '################Chosen Network Architecture: {}'.format(args.network)  + Style.RESET_ALL)

## load pre-trained model
if args.pretrain == 'True':
    print('Pretrained weight is found! Start to load weight from: {}'.format(args.pretrained_weights))
    model.load_state_dict(torch.load(args.pretrained_weights))

## Define Loss function
loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
print(Fore.GREEN + '##############Loss for training: {}'.format('DiceCELoss') + Style.RESET_ALL)

# loss_function = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
# print(Fore.GREEN + '##############Loss for training: {}'.format('DiceLoss') + Style.RESET_ALL)

# loss_function = weightedDiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
# print(Fore.GREEN + '##############Loss for training: {}'.format('weightedDiceCELoss') + Style.RESET_ALL)

## Define optimizer
if args.optim == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
elif args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=1e-8)

## Cosine annealing
if args.lrschedule == "warmup_cosine":
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=args.max_iter//100, max_epochs=args.max_iter
    )
elif args.lrschedule == "cosine_anneal":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iter)
## Learning rate adjustment strategy
else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.9, patience=5)    # maximizing dice, and the learning rate is reduced by 10% if there is no improvement within 5 cycles.
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)    # reduce the learning rate by 10% at regular intervals

root_dir = os.path.join(args.output)
t_dir = os.path.join(root_dir, 'tensorboard')
if dist.get_rank() == 0:
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)

dist.barrier()

## Remove ineffective weights each time validation is performed
def delete_low_dice_models(directory, dice_val_best, threshold=0.02):
    dice_threshold = dice_val_best - threshold
    for filename in os.listdir(directory):
        if filename.startswith("dice") and filename.endswith(".pth"):
            match = re.search(r"dice(\d+\.\d+)", filename)
            if match:
                dice_score = float(match.group(1))
                file_path = os.path.join(directory, filename)
                if dice_score < dice_threshold:
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)   
                        except FileNotFoundError:
                            logging.warning(f"File {filename} not found. Skipping deletion.")
                    else:
                        logging.warning(f"File {file_path} does not exist.")

def validation(epoch_iterator_val):
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            with autocast(enabled=args.amp):
                val_outputs = sliding_window_inference(val_inputs, (96,96,96), args.batch_size, model, overlap=0.5)
                
            ##  You should turn it on for wightedDiceCE Loss
            # loss_dict = loss_function(val_outputs, val_labels)  # get loss dictionary
            # loss_tuple = loss_dict[1]
            # print(f"Type of loss_dict: {type(loss_dict)}, Value of loss_dict: {loss_dict}")
            # lambda_dice = loss_tuple['lambda_dice']
            # dice_loss = loss_tuple['dice_loss']
            # lambda_ce = loss_tuple['lambda_ce']
            # ce_loss = loss_tuple['ce_loss']
            # logging.info(
            #     f'Validate Step {global_step}, lambda dice: {lambda_dice}, Dice Loss: {dice_loss.item()}, '
            #     f'lambda ce: {lambda_ce}, CE Loss: {ce_loss.item()}'
            # )
            ##  The end
            
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            current_dice = dice_metric.aggregate().item()
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, len(epoch_iterator_val), current_dice)
            )
    mean_dice_val = dice_metric.aggregate().item()
    # print(mean_dice_val)
    dice_metric.reset()
    # writer.add_scalar('Validation Dice score', mean_dice_val,global_step)
    return mean_dice_val

## get Beijing time
beijing_tz = pytz.timezone('Asia/Shanghai')

## setting log format
logging.basicConfig(filename=os.path.join(root_dir, 'training.log'), level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S %Z')

## beijing time is used for each log write
logging.Formatter.converter = lambda *args: datetime.now(tz=beijing_tz).timetuple()
args_str = ", ".join(f"{key}={value}" for key, value in vars(args).items())
logging.info("setting: " + args_str)

def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    if args.amp:
        scaler = GradScaler()
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        # print(x.shape,y.shape)

        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logit_map = model(x)
            total_loss = loss_function(logit_map, y)
            # total_loss, loss_dict = loss_function(logit_map, y) # You should turn it on for wightedDiceCE Loss
        if args.amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        
        epoch_loss += total_loss.item()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, total_loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            model.train()
            scheduler.step(dice_val)
                            
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(args.output,'best_metric_model.pth')) 
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
                logging.info(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {} Current Step: {}".format(
                        dice_val_best, dice_val, global_step
                    )
                )
            elif dice_val > 0.9:
                torch.save(model.state_dict(), os.path.join(root_dir, "dice{}_metric_model.pth".format(dice_val)))
                print(
                    "Model Was Not Best But Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
                logging.info(
                    "Model Was Not Best But Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {} Current Step: {}".format(
                        dice_val_best, dice_val, global_step
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
                logging.info(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {} Current Step: {}".format(
                        dice_val_best, dice_val, global_step
                    )
                )
            delete_low_dice_models(args.output, dice_val_best)
        # if scheduler is not None:
        #     scheduler.step()                
        # writer.add_scalar('Training Segmentation Loss', total_loss.data, global_step)
        global_step += 1
    return global_step, dice_val_best, global_step_best


max_iterations = args.max_iter
print('Maximum Iterations for training: {}'.format(str(args.max_iter)))
eval_num = args.eval_step
post_label = AsDiscrete(to_onehot=out_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )
