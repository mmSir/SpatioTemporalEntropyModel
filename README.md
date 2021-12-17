# Spatio-Temporal Entropy Model

A Pytorch Reproduction of Spatio-Temporal Entropy Model (STEM) for end-to-end leaned video compression.

More details can be found in the following paper:

>[Spatiotemporal Entropy Model is All You Need for Learned Video Compression](https://arxiv.org/abs/2104.06083)  
>Alibaba Group, arxiv 2021.4.13  
>Zhenhong Sun, Zhiyu Tan, Xiuyu Sun, Fangyi Zhang, Dongyang Li, Yichen Qian, Hao Li

*Note that It Is Not An Official Implementation Code.*

The differences with the original paper are not limited to the following:
* The number of model channels are fewer.
* The Encoder/Decoder in original paper consists of conditional conv[<sup>1</sup>](#refer-anchor-1) to support various rate in one single model. And the architecture is the same 
as [2][<sup>2</sup>](#refer-anchor-2). However, I only use the single rate Encoder/Decoder with the same architecture as [2][<sup>2</sup>](#refer-anchor-2)

ToDo:
- [ ] 1. various rate model training and evaluation.

# Environment

* Python == 3.7.10
* Pytorch == 1.7.1
* CompressAI

# Dataset
I use the Vimeo90k Septuplet Dataset to train the models. The Dataset contains about 64612 training sequences and 7824 testing sequences. All sequence contains 7 frames.

The train dataset folder structure is as
```
.dataset/vimeo_septuplet/
│  sep_testlist.txt
│  sep_trainlist.txt
│  vimeo_septuplet.txt
│  
├─sequences
│  ├─00001
│  │  ├─0001
│  │  │      f001.png
│  │  │      f002.png
│  │  │      f003.png
│  │  │      f004.png
│  │  │      f005.png
│  │  │      f006.png
│  │  │      f007.png
│  │  ├─0002
│  │  │      f001.png
│  │  │      f002.png
│  │  │      f003.png
│  │  │      f004.png
│  │  │      f005.png
│  │  │      f006.png
│  │  │      f007.png
...
```

I evaluate the model on UVG & HEVC TEST SEQUENCE Dataset.
The test dataset folder structure is as
```
.dataset/UVG/
├─PNG
│  ├─Beauty
│  │      f001.png
│  │      f002.png
│  │      f003.png
...
│  │      f598.png
│  │      f599.png
│  │      f600.png
│  │      
│  ├─HoneyBee
│  │      f001.png
│  │      f002.png
│  │      f003.png
...
│  │      f598.png
│  │      f599.png
│  │      f600.png
│  │     
...
```
```
.dataset/HEVC/
├─BasketballDrill
│      f001.png
│      f002.png
│      f003.png
...
│      f098.png
│      f099.png
│      f100.png
│      
├─BasketballDrive
│      f001.png
│      f002.png
...
```

# Train Your Own Model
>python3 trainSTEM.py -d /path/to/your/image/dataset/vimeo_septuplet --lambda 0.01 -lr 1e-4 --batch-size 16 --model-save /path/to/your/model/save/dir --cuda --checkpoint /path/to/your/iframecompressor/checkpoint.pth.tar

# Evaluate Your Own Model
>python3 evalSTEM.py --checkpoint /path/to/your/iframecompressor/checkpoint.pth.tar --entropy-model-path /path/to/your/stem/checkpoint.pth.tar

Currently only support evaluation on UVG & HEVC TEST SEQUENCE Dataset.

# Result

| 测试数据集UVG | PSNR | BPP | PSNR in paper | BPP in paper |
| --- | --- | --- | --- | --- |
| SpatioTemporalPriorModel_Res | 36.104 |  0.087 | 35.95 | 0.080 |
| SpatioTemporalPriorModel | 36.053 |  0.080 | 35.95 | 0.082 |
| SpatioTemporalPriorModelWithoutTPM | None |  None | 35.95 | 0.100 |
| SpatioTemporalPriorModelWithoutSPM | 36.066 |  0.080 | 35.95 | 0.087 |
| SpatioTemporalPriorModelWithoutSPMTPM | 36.021 |  0.141 | 35.95 | 0.123 |

PSNR in paper & BPP in paper is estimated from Figure 6 in the original paper.

# More Informations About Various Rate Model Training
As stated in the original paper, they use a variable-rate auto-encoder to support various rate in one single model. I tried to train with [GainedVAE](https://github.com/mmSir/GainedVAE), which is also a various rate model. Some point can achieve comparable r-d performance while others may degrade. What's more, the interpolation result could have more performance degradation cases.
Probably we need Loss Modulator[<sup>3</sup>](#refer-anchor-3) for various rate model training. Read Oren Ripple's ICCV 2021 paper for more details.


# Acknowledgement

The framework is based on CompressAI, I add the model in compressai.models.spatiotemporalpriors.
And trainSTEM.py/evalSTEM.py is modified with reference to compressai_examples

# Reference
<div id="refer-anchor-1"></div>
[1] [Variable Rate Deep Image Compression With a Conditional Autoencoder](https://openaccess.thecvf.com/content_ICCV_2019/html/Choi_Variable_Rate_Deep_Image_Compression_With_a_Conditional_Autoencoder_ICCV_2019_paper.html) 
<div id="refer-anchor-2"></div>
[2] [Joint Autoregressive and Hierarchical Priors for Learned Image Compression](https://arxiv.org/abs/1809.02736) 
<div id="refer-anchor-3"></div>
[3] [ELF-VC Efficient Learned Flexible-Rate Video Coding](https://arxiv.org/abs/2104.14335) 

# Contact
Feel free to contact me if there is any question about the code or to discuss any problems with image and video compression. (mxh_wine@qq.com)
