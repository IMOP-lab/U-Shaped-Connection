# Installation
We provide a detailed description of the environment settings required to replicate our project.

## Basic development environment tools
To reproduce our project, you need to install conda, cuda and cudnn firstly.

Install [conda](https://www.anaconda.com/download-success) (we use 23.7.4 version)

Install [cuda](https://developer.nvidia.com/cuda-toolkit-archive) >= 11.8.0 (we use 11.8.0 version)

Install [cudnn](https://developer.nvidia.com/rdp/cudnn-archive) (we use 8.9.2 for cuda11.x version)

## Conda environment setup
You can create and enter your own conda environment by:
"""
conda create -n 3dseg python=3.9.0
source activate 3dseg
"""

Alternatively, you can install the required conda environment from the 'environment.yaml' file by:
"""
conda create -f environment.yaml
"""
Notice that you need to replace 'prefix: /home/user/anaconda3/envs/3dseg' with your virtual environment path in the 'environment.yaml',and you can rename the created virtual environment.

Install [pytorch, torchvision, torchaudio](https://pytorch.org/get-started/previous-versions/) (it depends on your nvidia driver and your cuda version)
We use torch==2.0.0, torchvision==0.15.1, torchaudio==2.0.1 for cuda 11.8.
You can install by:
"""
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
"""

# Datasets format
You need to put the downloaded datasets in the ./datasets path by:

    ./datasets/
    ├── ABCT1K
    ├── FeTA
    ├── FLARE
    ├── OIMHS
    ├── BTCV
    ├── ...

The dataset is further divided into training sets, validation sets, and test sets in the following format:

    path to the dataset/
    ├── imagesTr
    ├── labelsTr
    ├── imagesVal
    ├── labelsVal
    ├── imagesTs
    ├── original_labelTs
    ├── shapes.json

Where original_labelTs holds all raw unlabeled data, shapes.json is each sample and its corresponding shape, for example: "train_000.nii.gz": [512, 512, 110].
Currently our framework only supports data in .nii.gz format,


# Training and testing

We currently provide run_gpu.sh to quickly train and test the model, and pretrained_test.sh to inference using the pre-trained weights provided in ./pretrained_models.