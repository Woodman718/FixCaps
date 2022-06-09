# Abstract
The early detection of skin cancer substantially improves the five-year survival rate of patients. It is often difficult to distinguish early malignant tumors from skin images, even by expert dermatologists. Therefore, several classification methods of dermatoscopic images have been proposed, but they have been found to be inadequate or defective for skin cancer detection, and often require a large amount of calculations. This study proposes an improved capsule network called FixCaps for dermoscopic image classification. FixCaps has a larger receptive field than CapsNets by applying a high-performance large-kernel at the bottom convolution layer whose kernel size is as large as 31 $\times$ 31, in contrast to commonly used 9 $\times$ 9. The convolutional block attention module was used to reduce the losses of spatial information caused by convolution and pooling. The group convolution was used to avoid model underfitting in the capsule layer. The network can improve the detection accuracy and reduce a great amount of calculations, compared with several existing methods. The experimental results showed that FixCaps is better than IRv2-SA for skin cancer diagnosis, which achieved an accuracy of 96.49\% on the HAM10000 dataset.

https://doi.org/10.1109/ACCESS.2022.318122

Note：Here's a trick. Changing "308" to "310" or "312" in the test data augment "transforms.  resize ((308,308))" boost up approximately 0.5% accuracy when testing. 

#Results
1. Classification accuracy (%) on the HAM10000 test set.

Method	|Accuracy [%]	|Params(M) 	|FLOPs(G)
|:--------:|:-------------:|:-------------:|:-------------:|
GoogLeNet	|83.94	|5.98	|1.58
Inception V3	|86.82	|22.8	|5.73
MobileNet V3	|89.97	|1.53	|0.12
IRv2-SA	|93.47	|47.5	|25.46
FixCaps-DS	|96.13	|0.14	|0.08
FixCaps	|96.49	|0.5	|6.74

2. The accuracy is evaluated on the test set by using different LKC(large-kernel convolution).
![Alt](https://github.com/Woodman718/FixCaps/blob/main/Images/LKC.png#pic_center)

#Datasets

Available:
```
https://challenge.isic-archive.com/data/#2018
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
```

HAM10000 dataset:

```
Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)", 2018; https://arxiv.org/abs/1902.03368

Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018). 
```

Available: https://www.nature.com/articles/sdata2018161, https://arxiv.org/abs/1803.10417

#Related Work

a. IRv2-SA

```
S. K. Datta, M. A. Shaikh, S. N. Srihari, and M. Gao. "Soft-Attention Improves Skin Cancer Classification Performance," Computer Science, vol 12929. Springer, Cham. doi: 10.1007/978-3-030-87444-5_2, 2021.
```

https://github.com/skrantidatta/Attention-based-Skin-Cancer-Classification

b. SLA-StyleGAN

```
C. Zhao, R. Shuai, L. Ma, W. Liu, D. Hu and M. Wu, ``Dermoscopy Image Classification Based on StyleGAN and DenseNet201," in IEEE Access, vol. 9, pp. 8659-8679, 2021, doi: 10.1109/ACCESS.2021.3049600.
```

#Citation
If you use FixCaps for your research or aplication, please consider citation:

```
@ARTICLE{9791221,
  author={Lan, Zhangli and Cai, Songbai and He, Xu and Wen, Xinpeng},
  journal={IEEE Access}, 
  title={FixCaps: An Improved Capsules Network for Diagnosis of Skin Cancer}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/ACCESS.2022.3181225}}
```
