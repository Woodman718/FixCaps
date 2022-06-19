## Abstract
The early detection of skin cancer substantially improves the five-year survival rate of patients. It is often difficult to distinguish early malignant tumors from skin images, even by expert dermatologists. Therefore, several classification methods of dermatoscopic images have been proposed, but they have been found to be inadequate or defective for skin cancer detection, and often require a large amount of calculations. This study proposes an improved capsule network called FixCaps for dermoscopic image classification. FixCaps has a larger receptive field than CapsNets by applying a high-performance large-kernel at the bottom convolution layer whose kernel size is as large as 31 $\times$ 31, in contrast to commonly used 9 $\times$ 9. The convolutional block attention module was used to reduce the losses of spatial information caused by convolution and pooling. The group convolution was used to avoid model underfitting in the capsule layer. The network can improve the detection accuracy and reduce a great amount of calculations, compared with several existing methods. The experimental results showed that FixCaps is better than IRv2-SA for skin cancer diagnosis, which achieved an accuracy of 96.49\% on the HAM10000 dataset.

https://doi.org/10.1109/ACCESS.2022.3181225

## Results
1. Classification accuracy (%) on the HAM10000 test set.

Method	|Accuracy [%]	|Params(M) 	|FLOPs(G)
|:--------:|:-------------:|:-------------:|:-------------:|
GoogLeNet	|83.94	|5.98	|1.58
Inception V3	|86.82	|22.8	|5.73
MobileNet V3	|89.97	|1.53	|0.12
IRv2-SA	|93.47	|47.5	|25.46
FixCaps-DS	|96.13	|0.14	|0.08
FixCaps	|96.49	|0.5	|6.74

```
pip install ipywidgets prettytable torchsummary tqdm seaborn matplotlib protobuf==3.16.0

#Calculate the Params (M) FLOPs (G)
pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git
pip install onnx

Failed to install the "onnx", official link as follows:
https://github.com/onnx/onnx
```

2. The accuracy is evaluated on the test set by using different LKC(large-kernel convolution).

![LKC](https://github.com/Woodman718/FixCaps/blob/main/Images/LKC.jpg#pic_center)

3 Evaluation metrics of the FixCaps.

<table> 
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

</td></tr> </table>

4 Generalization performance on the COVID-19 Radiography Database

<table> 
<tr><th>Evaluation Metrics</th><th>Distribution of the COVID-19 Radiography Database</th></tr> 
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
Dataset:  https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
The COVID-19 Radiography Database consisted of 21165 images.
Among them, covid(3616),normal(10192),opacity(6012),viral(1345).
```

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
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/ACCESS.2022.3181225}}
```
