# U-Shaped Connection
**Official code implementation of 'Upping the Game: How 2D U-Net Skip Connection Flipping 3D Segmentation'**

### [Project page](https://github.com/IMOP-lab/U-Shaped Connection-Pytorch) | [Our laboratory home page](https://github.com/IMOP-lab)

# Methodology
<div align=center>
  <img src="https://github.com/IMOP-lab/U-Shaped-Connection/blob/main/figures/uC%203DU-Net.png">
</div>
<p align=center>
  Figure 1: Detailed module structure of the uC 3DU-Net.
</p>

**We introduce uC3D U-Net, which integrates U-Shaped Connection into a 3D U-Net backbone, augmented with a dual feature integration (DFi) module.**

## Installation
    python = 3.9.0
    pytorch = 2.0.0+cu118
    monai = 0.9.0
    numpy = 1.23.2
**For a full list of software packages and version numbers, please take a look at the experimental environment file 'environment.yaml'.**

# Experiment

## Quantitative Results
<div>
  <img src="https://github.com/IMOP-lab/U-Shaped-Connection-Pytorch/blob/main/tables/fig1.png"width=80% height=80%>
</div>
<p>
  Table 1: Quantitative results on the FALRE2021 and FeTA2021 datasets.
</p>

<div>
  <img src="https://github.com/IMOP-lab/U-Shaped-Connection-Pytorch/blob/main/tables/fig2.png"width=40% height=40%>
</div>
<p>
  Table 2: Quantitative results on the OIMHS dataset.
</p>

<div>
  <img src="https://github.com/IMOP-lab/U-Shaped-Connection-Pytorch/blob/main/tables/fig3.png"width=40% height=40%>
</div>
<p>
  Table 3: Quantitative results on the AbdomenCT-1K dataset.
</p>

## Qualitative Results
<div align=center>
  <img src="https://github.com/IMOP-lab/U-Shaped-Connection-Pytorch/blob/main/figures/flare2d.png"width=80% height=80%>
</div>
<p align=center>
  Fig. 2: Qualitative results of the uC’s impact on segmentation performance in 3DUX-Net and SegResNet models on the FALRE2021 dataset.
</p>

<div align=center>
  <img src="https://github.com/IMOP-lab/U-Shaped-Connection-Pytorch/blob/main/figures/image_2d.png"width=80% height=80%>
</div>
<p align=center>
  Fig. 3: Qualitative results of the segmentation performance on 3DUX-Net and 3DUXNET+uC with various channel depths on the OIMHS dataset.
</p>

## License
**This project is licensed under the [MIT license](https://github.com/IMOP-lab/U-Shaped Connection-Pytorch/blob/main/LICENSE).**

## Contributors
**The project was implemented with the help of the following contributors:**

Xingru Huang, Yihao Guo, Jian Huang, Tianyun Zhang, Hong He, Shaowei Jiang, Yaoqi Sun.



