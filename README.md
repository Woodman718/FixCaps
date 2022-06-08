# An Improved Capsules Network for Diagnosis of Skin Cancer
The early detection of skin cancer substantially improves the five-year survival rate of patients. It is often difficult to distinguish early malignant tumors from skin images, even by expert dermatologists. Therefore, several classification methods of dermatoscopic images have been proposed, but they have been found to be inadequate or defective for skin cancer detection, and often require a large amount of calculations. This study proposes an improved capsule network called FixCaps for dermoscopic image classification. FixCaps has a larger receptive field than CapsNets by applying a high-performance large-kernel at the bottom convolution layer whose kernel size is as large as 31 $\times$ 31, in contrast to commonly used 9 $\times$ 9. The convolutional block attention module was used to reduce the losses of spatial information caused by convolution and pooling. The group convolution was used to avoid model underfitting in the capsule layer. The network can improve the detection accuracy and reduce a great amount of calculations, compared with several existing methods. The experimental results showed that FixCaps is better than IRv2-SA for skin cancer diagnosis, which achieved an accuracy of 96.49\% on the HAM10000 dataset.

https://doi.org/10.1109/ACCESS.2022.3181225

#Results
1. The accuracy is evaluated on the test set by using different LKC(large-kernel convolution).
pass
2. Evaluation metrics of FixCaps and IRV2-SA for each skin lesion type on the test set.
pass

#Citation

@article{lan2022fixcaps, title={FixCaps: An Improved Capsules Network for Diagnosis of Skin Cancer}, author={Lan, Zhangli and Cai, Songbai and He, Xu and Wen, Xinpeng}, journal={IEEE Access}, year={2022},doi={10.1109/ACCESS.2022.3181225}}

#Datasets

https://challenge.isic-archive.com/data/#2018
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T


HAM10000 dataset:

Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)", 2018; https://arxiv.org/abs/1902.03368

Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018). 

Available: https://www.nature.com/articles/sdata2018161, https://arxiv.org/abs/1803.10417
