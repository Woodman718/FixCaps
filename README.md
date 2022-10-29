## Abstract
The early detection of skin cancer substantially improves the five-year survival rate of patients. It is often difficult to distinguish early malignant tumors from skin images, even by expert dermatologists. Therefore, several classification methods of dermatoscopic images have been proposed, but they have been found to be inadequate or defective for skin cancer detection, and often require a large amount of calculations. This study proposes an improved capsule network called FixCaps for dermoscopic image classification. FixCaps has a larger receptive field than CapsNets by applying a high-performance large-kernel at the bottom convolution layer whose kernel size is as large as 31 $\times$ 31, in contrast to commonly used 9 $\times$ 9. The convolutional block attention module was used to reduce the losses of spatial information caused by convolution and pooling. The group convolution was used to avoid model underfitting in the capsule layer. The network can improve the detection accuracy and reduce a great amount of calculations, compared with several existing methods. The experimental results showed that FixCaps is better than IRv2-SA for skin cancer diagnosis, which achieved an accuracy of 96.49\% on the HAM10000 dataset.

https://doi.org/10.1109/ACCESS.2022.3181225

Note: 
The augmented data of HAM10000 can be obtained as follows:
https://aistudio.baidu.com/aistudio/datasetdetail/151696

## Results
1. Classification accuracy (%) on the HAM10000 test set.

|Method |Accuracy [%] |Params(M)  |FLOPs(G)|
|:--------:|:-------------:|:-------------:|:-------------:|
GoogLeNet |83.94  |5.98 |1.58
Inception V3  |86.82  |22.8 |5.73
MobileNet V3  |89.97  |1.53 |0.12
IRv2-SA |93.47  |47.5 |25.46
FixCaps-DS  |96.13  |0.14 |0.08
FixCaps |96.49  |0.5  |6.74

2. The accuracy is evaluated on the test set by using different LKC(large-kernel convolution).

![LKC](https://github.com/Woodman718/FixCaps/blob/main/Images/LKC.jpg#pic_center)

3 Evaluation metrics of the FixCaps.

<table> 
 <tr><th>FixCaps-31</th><th>Distribution of the HAM10000 Dataset</th></tr> 
<tr><td> 

|  Type  | Precision | Recall |   F1  | Accuracy |
|:--------:|:-------:|:-------------:|:--------:|:----------:|
| akiec  |   0.88 | 0.957  | 0.917 |          |
|  bcc   |  0.9565| 0.846  | 0.898 |          |
|  bkl   |  0.8676| 0.894  | 0.881 |          |
|   df   |  0.5714| 0.667  | 0.615 |          |
|  mel   |  0.9394| 0.912  | 0.925 |          |
|   nv   |  0.9835| 0.986  | 0.985 |          |
|  vasc  |   1.0  |  0.7   | 0.824 |          |
|overall:|        |        |       |  0.9649  |

</td><td>
 
 ![dis_data](https://github.com/Woodman718/FixCaps/blob/main/Images/Dis_HAM10000_paper.png)
 
</td></tr> </table>

4 Generalization Performance

<table> 
 <tr><th>Robustness(FixCaps-29)<th>Distribution of the HAM10000 Dataset</th></tr> 
<tr><td> 

![dis_data](https://github.com/Woodman718/FixCaps/blob/main/Images/Size_Accuracy_29.png)

</td><td> 

![dis_data](https://github.com/Woodman718/FixCaps/blob/main/Images/Dis_HAM10000_GP.png)

</td></tr>
 <tr><th>FixCaps-29(Driver Version: 515.76)</th></th><th>Evaluation Metrics(RTX3070)</th></tr> 
<tr><td>
 
|  Type  | Precision | Recall |   F1  | Accuracy |
|:--------:|:-------:|:-------------:|:--------:|:----------:|
| akiec  |    1.0    |  1.0   |  1.0  |          |
|  bcc   |   0.9259  | 0.962  | 0.943 |          |
|  bkl   |   0.9344  | 0.864  | 0.898 |          |
|   df   |   0.4444  | 0.667  | 0.533 |          |
|  mel   |   0.931   | 0.794  | 0.857 |          |
|   nv   |   0.9776  | 0.989  | 0.984 |          |
|  vasc  |    1.0    |  0.8   | 0.889 |          |
|overall:|        |        |       |  0.9662  |
 
</td><td> 

| Method  |Accuracy[%]|Params(M)|FLOPs(G)| FPS |
|:--------:|:-------------:|:--------:|:--------:|:--------:|
| FixCaps_DS-18  |   95.894 | 0.13  | 0.03 |130.4|
| FixCaps_DS-24  |   94.08 | 0.13  | 0.05 |127.8|
| FixCaps_DS-31  |   94.324 | 0.14  | 0.07 |127.5|
| FixCaps_18   |  96.376 | 0.26  | 2.49 |130.9|
| FixCaps-21   |  96.014 | 0.30  | 3.33 |123.4|
| FixCaps-24   |  96.256 | 0.35  | 4.22 |121.0|
| FixCaps-29   |  96.618 | 0.46  | 5.99 |119.2|
| FixCaps-31   |  93.961 | 0.50  | 6.74 |114.7|
| FixCaps-33   |  94.806 | 0.55  | 7.52 |113.5|

</td></tr> 
 <tr><th>FixCaps-18(Ablation-CAM)</th><th>FixCaps_DS-18</th></tr> 
<tr><td> 

|  Type  | Precision | Recall |   F1  | Accuracy |
|:--------:|:-------:|:-------------:|:--------:|:----------:|
| akiec  |   0.8846  |  1.0   | 0.939 |          |
|  bcc   |    1.0    | 0.923  |  0.96 |          |
|  bkl   |   0.9104  | 0.924  | 0.917 |          |
|   df   |   0.3333  | 0.167  | 0.222 |          |
|  mel   |    0.95   | 0.559  | 0.704 |          |
|   nv   |   0.9735  | 0.995  | 0.984 |          |
|  vasc  |    1.0    |  1.0   |  1.0  |          |
|overall:|        |        |       |  0.9638  |

</td><td> 

|  Type  | Precision | Recall |   F1  | Accuracy |
|:--------:|:-------:|:-------------:|:--------:|:----------:|
| akiec  |    0.8    |  0.87  | 0.833 |          |
|  bcc   |    0.88   | 0.846  | 0.863 |          |
|  bkl   |   0.8939  | 0.894  | 0.894 |          |
|   df   |    0.5    |  0.5   |  0.5  |          |
|  mel   |   0.9565  | 0.647  | 0.772 |          |
|   nv   |   0.9792  | 0.992  | 0.986 |          |
|  vasc  |   0.9091  |  1.0   | 0.952 |          |
|overall:|        |        |       |  0.9589  |

</td></tr> </table>

```
Dataset:  https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
The COVID-19 Radiography Database consisted of 21165 images.
Among them, covid(3616),normal(10192),opacity(6012),viral(1345).
```

<table> 
<tr><th>Evaluation Metrics</th><th>Distribution of the COVID-19 Radiography Dataset</th></tr> 
<tr><td> 

|  Type  | Precision | Recall |  F1  | Accuracy |
|:--------:|:-------------:|:-------------:|:--------:|:----------:|
|  covid  |0.9918| 1.0 |0.996 |      |
|  normal |  1.0 |0.988|0.994 |      |
| opacity |0.9852|  1.0|0.993 |      |
|  viral  | 1.0  |  1.0| 1.0  |      |
|overall:|      |     |      |0.9943|

</td><td>
 
 ![dis_data](https://github.com/Woodman718/FixCaps/blob/main/Module/COVID-19/Dis_COVID-19_data.png)
 
</td></tr>
</table>

```
Source Data: http://dx.doi.org/10.5281/zenodo.1214456
Jakob Nikolas Kather, Johannes Krisam, et al., "Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study," PLOS Medicine, vol. 16, no. 1, pp. 1–22, 01 2019.
This is a slightly different version of the "NCT-CRC-HE-100K" image set: This set contains 100,000 images in 9 tissue classes at 0.5 MPP and was created from the same raw data as "NCT-CRC-HE-100K". 
However, no color normalization was applied to these images. Consequently, staining intensity and color slightly varies between the images. Please note that although this image set was created from the same data as "NCT-CRC-HE-100K", the image regions are not completely identical because the selection of non-overlapping tiles from raw images was a stochastic process.
```

<table> 
<tr><th>FixCaps-DS-18</th><th>NCT-CRC-HE-100K-NONORM</th></tr> 
<tr><td> 

|  Type  | Precision | Recall |  F1  | Accuracy |
|:--------:|:-------------:|:-------------:|:--------:|:----------:|
|  ADI   |   0.9952  | 0.997  | 0.996 |          |
|  BACK  |   0.9972  |  1.0   | 0.999 |          |
|  DEB   |   0.9965  | 0.988  | 0.992 |          |
|  LYM   |   0.9948  | 0.993  | 0.994 |          |
|  MUC   |   0.9932  | 0.987  |  0.99 |          |
|  MUS   |   0.9941  | 0.996  | 0.995 |          |
|  NORM  |   0.9853  | 0.995  |  0.99 |          |
|  STR   |   0.9801  |  0.99  | 0.985 |          |
|  TUM   |   0.9951  | 0.989  | 0.992 |          |
|overall:|           |        |       |0.9927|

</td><td>
 
 ![dis_data](https://github.com/Woodman718/FixCaps/blob/main/Images/Dis_NCT-CRC-HE-100K-NONORM.png)
 
</td></tr>
</table>

## Dataset

![Data](https://github.com/Woodman718/FixCaps/blob/main/Images/data.jpg#pic_center)

```
Example of Skin lesions in HAM10000 dataset.
Among them, BKL, DF, NV, and VASC are benign tumors, whereas AKIEC, BCC, and MEL are malignant tumors.

Available:
https://challenge.isic-archive.com/data/#2018
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
```

HAM10000 dataset:

```
Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)", 2018; https://arxiv.org/abs/1902.03368

Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018). 
Available: https://www.nature.com/articles/sdata2018161, https://arxiv.org/abs/1803.10417
```

## License

The dataset is released under a Creative Commons Attribution 4.0 License.
For more information, see https://creativecommons.org/licenses/by/4.0/ .

## Related Work

a. IRv2-SA

```
S. K. Datta, M. A. Shaikh, S. N. Srihari, and M. Gao. "Soft-Attention Improves Skin Cancer Classification Performance," 
Computer Science, vol 12929. Springer, Cham, 2021. doi: 10.1007/978-3-030-87444-5_2.

https://github.com/skrantidatta/Attention-based-Skin-Cancer-Classification
```

b. SLA-StyleGAN

```
C. Zhao, R. Shuai, L. Ma, W. Liu, D. Hu and M. Wu, ``Dermoscopy Image Classification Based on StyleGAN and DenseNet201," 
in IEEE Access, vol. 9, pp. 8659-8679, 2021, doi: 10.1109/ACCESS.2021.3049600.
```

## Citation

If you use FixCaps for your research or aplication, please consider citation:

```
@ARTICLE{9791221,
  author={Lan, Zhangli and Cai, Songbai and He, Xu and Wen, Xinpeng},
  journal={IEEE Access}, 
  title={FixCaps: An Improved Capsules Network for Diagnosis of Skin Cancer}, 
  year={2022},
  volume={10},
  number={},
  pages={76261-76267},
  doi={10.1109/ACCESS.2022.3181225}}
```
