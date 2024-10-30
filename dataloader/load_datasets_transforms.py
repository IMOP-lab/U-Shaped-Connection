import numpy as np
import glob
import warnings
import os
warnings.filterwarnings('ignore')
from batchgenerators.utilities.file_and_folder_operations import *
from monai import transforms
from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    EnsureTyped,
    Invertd,
    KeepLargestConnectedComponentd,
    SaveImaged,
    Activationsd,
    Rand3DElastic,
    RandGaussianNoised,
    Rand3DElasticd,
    NormalizeIntensityd,
    ResizeD,
    RandAdjustContrastd,
    RandFlipd,
    RandScaleIntensityd,
    RandRotate90d
)
  
def data_loader(args):
    root_dir = args.root    # location of dataset
    dataset = args.dataset  # dataloader mode
    out_classes = args.out_classes  # catagory

    print('Start to load data from directory: {}'.format(root_dir))

    if args.mode == 'train':
        train_samples = {}
        valid_samples = {}

        ## Input training data
        train_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTr', '*.nii.gz')))   # glob to search, sorted to make sure same order
        train_label = sorted(glob.glob(os.path.join(root_dir, 'labelsTr', '*.nii.gz')))
        # print(train_img, train_label)
        train_img = train_img
        train_label = train_label
        train_samples['images'] = train_img
        train_samples['labels'] = train_label

        ## Input validation data
        valid_img = sorted(glob.glob(os.path.join(root_dir, 'imagesVal', '*.nii.gz')))
        valid_label = sorted(glob.glob(os.path.join(root_dir, 'labelsVal', '*.nii.gz')))
        # valid_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTs', '*.nii.gz')))
        # valid_label = sorted(glob.glob(os.path.join(root_dir, 'labelsTs', '*.nii.gz')))
        valid_samples['images'] = valid_img
        valid_samples['labels'] = valid_label


        print('Finished loading all training and validation samples from dataset: {}!'.format(dataset))
        print('Number of classes for segmentation: {}'.format(out_classes))

        return train_samples, valid_samples, out_classes

    elif args.mode == 'test':
        test_samples = {}

        ## Input test data
        test_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTs', '*.nii.gz')))
        test_samples['images'] = test_img

        print('Finished loading all test samples from dataset: {}!'.format(dataset))

        return test_samples, out_classes


def data_transforms(args):
    dataset = args.dataset
    if args.mode == 'train':
        crop_samples = args.crop_sample
    else:
        crop_samples = None

    if dataset == 'OIMHS':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                   
                ### Adjust voxel spacing of images and labels
                # Spacingd(keys=["image", "label"], pixdim=(
                #     1.0, 1.0, 2.0), mode=("bilinear", "nearest")),
                
                ### Adjust the spatial dimensions of images and labels
                # ResizeD(keys=["image", "label"], spatial_size=(-1, -1, 0), mode=('trilinear', 'nearest')),
                
                ## Adjust image intensity
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=300, b_min=0, b_max=1, clip=True
                ),
                # NormalizeIntensityd(keys='image',subtrahend=0,divisor=10),
                
                ### Remove irrelevant background
                CropForegroundd(keys=["image", "label"], source_key="image"),
                
                ### Various random data enhancement methods ###
                
                ### Randomly crop positive and negative samples, all non-zero pixels are considered as foreground
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96,96,96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                
                ### Random 3D elastic transformation
                Rand3DElasticd(
                    keys=["image", "label"],prob=0.5,
                    sigma_range=(5,10),
                    magnitude_range=(50, 150),
                    padding_mode='zeros',
                    mode = ['bilinear','nearest']
                ), 
                
                ### Random Flip
                RandFlipd(
                    keys=['image', 'label'],
                    prob=0.5,
                    spatial_axis=2,
                ),
                
                ### Random rotation and scaling
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, 
                    spatial_size=(96,96,96),
                    rotate_range=(0, 0, np.pi / 20),
                    scale_range=(0.15, 0.15, 0.15)),
                
                ### Contrast Enhancement  
                # RandAdjustContrastd( 
                #     keys=["image"],
                #     prob=0.5,
                #     gamma=(0.5, 1.5)),
                
                ### Random Gaussian noise
                # RandGaussianNoised(
                #     keys=['image'], 
                #     prob=0.5,
                #     mean=0.0, 
                #     std=0.1), 
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # Spacingd(keys=["image", "label"], pixdim=(
                    # 1.0, 1.0, 2.0), mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=300, b_min=0, b_max=1, clip=True
                ),
                # NormalizeIntensityd(keys='image',subtrahend=0,divisor=10),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                # Spacingd(keys=["image"], pixdim=(
                    # 1.0, 1.0, 2.0), mode=("bilinear")),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=300, b_min=0, b_max=1, clip=True
                ),
                # NormalizeIntensityd(keys='image',subtrahend=0,divisor=10),
                CropForegroundd(keys=["image"], source_key="image"),
            ]
        )

    elif dataset == 'FLARE':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # Spacingd(keys=["image", "label"], pixdim=(
                #     0.4, 0.4, 0.4), mode=("bilinear", "nearest")),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),      

                ### Various random data enhancement methods ###
                
                ### Randomly crop positive and negative samples, all non-zero pixels are considered as foreground
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96,96,96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                
                ### Random 3D elastic transformation
                Rand3DElasticd(
                    keys=["image", "label"],prob=0.5,
                    sigma_range=(5,10),
                    magnitude_range=(50, 150),
                    padding_mode='zeros',
                    mode = ['bilinear','nearest']
                ), 
                
                ### Random Flip
                RandFlipd(
                    keys=['image', 'label'],
                    prob=0.5,
                    spatial_axis=2,
                ),
                
                ### Random rotation and scaling
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, 
                    spatial_size=(96,96,96),
                    rotate_range=(0, 0, np.pi / 20),
                    scale_range=(0.15, 0.15, 0.15)),
                
                ### Contrast Enhancement  
                # RandAdjustContrastd( 
                #     keys=["image"],
                #     prob=0.5,
                #     gamma=(0.5, 1.5)),
                
                ### Random Gaussian noise
                # RandGaussianNoised(
                #     keys=['image'], 
                #     prob=0.5,
                #     mean=0.0, 
                #     std=0.1), 
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # Spacingd(keys=["image", "label"], pixdim=(
                #     1.0, 1.0, 1.2), mode=("bilinear", "nearest")),
                # ResizeD(keys=["image", "label"], spatial_size=(-1, -1, 0), mode=('trilinear', 'nearest')),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                # Spacingd(keys=["image"], pixdim=(
                #     1.0, 1.0, 1.2), mode=("bilinear")),
                # ResizeD(keys=["image"], spatial_size=(-1, -1, 0), mode=('trilinear')),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=275,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
            ]
        )

    elif dataset == 'ABCT1K':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # Spacingd(keys=["image", "label"], pixdim=(
                #     1.0, 1.0, 1.2), mode=("bilinear", "nearest")),
                # ResizeD(keys=["image", "label"], spatial_size=(-1, -1, 0), mode=('trilinear', 'nearest')),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=300,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),

                ### Various random data enhancement methods ###
                
                ### Randomly crop positive and negative samples, all non-zero pixels are considered as foreground
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96,96,96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                
                ### Random 3D elastic transformation
                Rand3DElasticd(
                    keys=["image", "label"],prob=0.5,
                    sigma_range=(5,10),
                    magnitude_range=(50, 150),
                    padding_mode='zeros',
                    mode = ['bilinear','nearest']
                ), 
                
                ### Random Flip
                RandFlipd(
                    keys=['image', 'label'],
                    prob=0.5,
                    spatial_axis=2,
                ),
                
                ### Random rotation and scaling
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, 
                    spatial_size=(96,96,96),
                    rotate_range=(0, 0, np.pi / 20),
                    scale_range=(0.15, 0.15, 0.15)),
                
                ### Contrast Enhancement  
                # RandAdjustContrastd( 
                #     keys=["image"],
                #     prob=0.5,
                #     gamma=(0.5, 1.5)),
                
                ### Random Gaussian noise
                # RandGaussianNoised(
                #     keys=['image'], 
                #     prob=0.5,
                #     mean=0.0, 
                #     std=0.1), 
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # Spacingd(keys=["image", "label"], pixdim=(
                #     1.0, 1.0, 1.2), mode=("bilinear", "nearest")),
                # ResizeD(keys=["image", "label"], spatial_size=(-1, -1, 0), mode=('trilinear', 'nearest')),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=300,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                # Spacingd(keys=["image"], pixdim=(
                    # 1.0, 1.0, 1.2), mode=("bilinear")),
                # ResizeD(keys=["image"], spatial_size=(-1, -1, 0), mode=('trilinear')),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-125, a_max=300,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
            ]
        )

    elif dataset == 'FeTA':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # Spacingd(keys=["image", "label"], pixdim=(
                #     0.4, 0.4, 0.4), mode=("bilinear", "nearest")),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=1000,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),      
                   
                ### Various random data enhancement methods ###
                
                ### Randomly crop positive and negative samples, all non-zero pixels are considered as foreground
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96,96,96),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                
                ### Random 3D elastic transformation
                Rand3DElasticd(
                    keys=["image", "label"],prob=0.5,
                    sigma_range=(5,10),
                    magnitude_range=(50, 150),
                    padding_mode='zeros',
                    mode = ['bilinear','nearest']
                ), 
                
                ### Random Flip
                RandFlipd(
                    keys=['image', 'label'],
                    prob=0.5,
                    spatial_axis=2,
                ),
                
                ### Random rotation and scaling
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, 
                    spatial_size=(96,96,96),
                    rotate_range=(0, 0, np.pi / 20),
                    scale_range=(0.15, 0.15, 0.15)),
                
                ### Contrast Enhancement  
                # RandAdjustContrastd( 
                #     keys=["image"],
                #     prob=0.5,
                #     gamma=(0.5, 1.5)),
                
                ### Random Gaussian noise
                # RandGaussianNoised(
                #     keys=['image'], 
                #     prob=0.5,
                #     mean=0.0, 
                #     std=0.1), 
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                # Spacingd(keys=["image", "label"], pixdim=(
                #     0.4, 0.4, 0.4), mode=("bilinear", "nearest")),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=1200,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                # Spacingd(keys=["image", "label"], pixdim=(
                #     0.4, 0.4, 0.4), mode=("bilinear", "nearest")),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=1200,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
            ]
        )

    elif dataset == 'BTCV':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),

                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,a_max=250,
                    b_min=0.0,b_max=1.0,clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                
                ### Various random data enhancement methods ###
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=1,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(keys=["image", "label"],spatial_axis=[0],prob=0.10,),
                RandFlipd(keys=["image", "label"],spatial_axis=[1],prob=0.10,),
                RandFlipd(keys=["image", "label"],spatial_axis=[2],prob=0.10,),
                RandRotate90d(keys=["image", "label"],prob=0.10,max_k=3,),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                
                ### Random rotation
                RandAffined(
                    keys=['image', 'label'],
                    # mode=('bilinear', 'nearest'),
                    mode=('nearest', 'nearest'),
                    prob=0.4, 
                    spatial_size=(96,96,96),
                    rotate_range=(0, 0, np.pi / 30),
                    scale_range=(0.1, 0.1, 0.1)),
                
                ### Random 3D elastic transformation
                Rand3DElasticd(
                    keys=["image", "label"],
                    prob=0.3,
                    sigma_range=(5, 10),
                    magnitude_range=(50, 150),
                    padding_mode='zeros',
                    mode = ['bilinear','nearest']
                ), 
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,a_max=250,
                    b_min=0.0,b_max=1.0,clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
            ]
        )
        
        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],a_min=-175,a_max=250,
                    b_min=0.0,b_max=1.0,clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                Spacingd(
                    keys=["image"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear"),
                ),
            ]
        )
    
    if args.mode == 'train':
        return train_transforms, val_transforms

    elif args.mode == 'test':
        return test_transforms


def infer_post_transforms(args, test_transforms, out_classes):

    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=test_transforms,
            orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
            # then invert `pred` based on this information. we can use same info
            # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
            # for example, may need the `affine` to invert `Spacingd` transform,
            # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
            # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
            # otherwise, no need this arg during inverting
            nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
            # to ensure a smooth output, then execute `AsDiscreted` transform
            to_tensor=True,  # convert to PyTorch Tensor after inverting
            device="cuda",  # test on gpus
        ),
        AsDiscreted(keys="pred", argmax=True),
        # KeepLargestConnectedComponentd(keys='pred', applied_labels=[1, 3, 4]), # 3D maximum connected area, properly enabling it can slightly improve model segmentation performance.
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=args.output,
                   output_postfix="seg", output_ext=".nii.gz", resample=True),
    ])

    return post_transforms