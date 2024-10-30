<div align="center">
  <h1>Upping the Game: How 2D U-Net Skip Connection Flip 3D Segmentation</h1>
  <h2>NeurIPS 2024 (poster)</h2>
</div>

## [Project page](https://github.com/IMOP-lab/U-Shaped-Connection-Pytorch) | [Our laboratory home page](https://github.com/IMOP-lab)

## 📖 Abstract
In the present study, we introduce an innovative structure for 3D medical image segmentation that effectively integrates 2D U-Net-derived skip connections into the architecture of 3D convolutional neural networks (3D CNNs). Conventional 3D segmentation techniques predominantly depend on isotropic 3D convolutions for the extraction of volumetric features, which frequently engenders inefficiencies due to the varying information density across the three orthogonal axes in medical imaging modalities such as computed tomography (CT) and magnetic resonance imaging (MRI). This disparity leads to a decline in axial-slice plane feature extraction efficiency, with slice plane features being comparatively underutilized relative to features in the time-axial. To address this issue, we introduce the U-shaped Connection (uC), utilizing simplified 2D U-Net in place of standard skip connections to augment the extraction of the axial-slice plane features while concurrently preserving the volumetric context afforded by 3D convolutions. Based on uC, we further present uC 3DU-Net, an enhanced 3D U-Net backbone that integrates the uC approach to facilitate optimal axial-slice plane feature utilization. Through rigorous experimental validation on five publicly accessible datasets—FLARE2021, OIMHS, FeTA2021, AbdomenCT-1K, and BTCV, the proposed method surpasses contemporary state-of-the-art models. Notably, this performance is achieved while reducing the number of parameters and computational complexity. This investigation underscores the efficacy of incorporating 2D convolutions within the framework of 3D CNNs to overcome the intrinsic limitations of volumetric segmentation, thereby potentially expanding the frontiers of medical image analysis.

## 🚀Methodology
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
**For a full list of software packages and version numbers, please take a look at the experimental environment file [environment.yaml](https://github.com/IMOP-lab/U-Shaped-Connection/blob/main/environment.yaml).**

# Experiment

## Quantitative Results
<div>
  <img src="https://github.com/IMOP-lab/U-Shaped-Connection/blob/main/tables/fig1.png"width=80% height=80%>
</div>
<p>
  Table 1: Quantitative results on the FALRE2021 and FeTA2021 datasets.
</p>

<div>
  <img src="https://github.com/IMOP-lab/U-Shaped-Connection/blob/main/tables/fig2.png"width=40% height=40%>
</div>
<p>
  Table 2: Quantitative results on the OIMHS dataset.
</p>

<div>
  <img src="https://github.com/IMOP-lab/U-Shaped-Connection/blob/main/tables/fig3.png"width=40% height=40%>
</div>
<p>
  Table 3: Quantitative results on the AbdomenCT-1K dataset.
</p>

## Qualitative Results
<div align=center>
  <img src="https://github.com/IMOP-lab/U-Shaped-Connection/blob/main/figures/flare2d.png"width=80% height=80%>
</div>
<p align=center>
  Fig. 2: Qualitative results of the uC’s impact on segmentation performance in 3DUX-Net and SegResNet models on the FALRE2021 dataset.
</p>

<div align=center>
  <img src="https://github.com/IMOP-lab/U-Shaped-Connection/blob/main/figures/image_2d.png"width=80% height=80%>
</div>
<p align=center>
  Fig. 3: Qualitative results of the segmentation performance on 3DUX-Net and 3DUXNET+uC with various channel depths on the OIMHS dataset.
</p>

## License
**This project is licensed under the [MIT license](https://github.com/IMOP-lab/U-Shaped-Connection-Pytorch/blob/main/LICENSE).**

## Contributors
**The project was implemented with the help of the following contributors:**

Xingru Huang, Yihao Guo, Jian Huang, Tianyun Zhang, Hong He, Shaowei Jiang, Yaoqi Sun.



