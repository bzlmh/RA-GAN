## RA-GAN: Region-Adaptive GAN for Binarization  on Degraded Document Images
<img width="600" alt="Figure1" src="frame/Overall%20Architecture.jpg">

## Description
This is the official implementation for our paper [RA-GAN: Region-Adaptive GAN for Binarization  on Degraded Document Image]. 
We have submitted our paper for publication and it is under review. The source codes will be updated here once the paper is accepted.

## Abstract
Binarization is a crucial pre-processing step for the visual analysis of degraded document images, which usually possess features like bleeding, stains, holes, and creases. 
Existing methods typically perform binarization by establishing a unified remote dependency between foreground and background pixels. 
However, two critical problems remain to be investigated: (1) the inability to accurately locate degraded regions, leading to inadequate adaptive feature learning, and (2) the failure to effectively utilize small variations at the edges of degraded regions. 
To address these issues, this paper proposes an image binarization method called RA-GAN, which can accurately locate degraded regions and perform fine-grained modeling. 
Specifically, RA-GAN comprises three main modules: the Gradient Gated Convolution (GGC) module, the Guided Attention (GA) module, and the Gated Feature Pyramid (GFP) module. 
The GGC module captures pixel mutation information from the neighborhood using a locally modulated convolution kernel and employs a gating mechanism to activate global information for adaptive supplementation, and the GA module identifies degraded regions in both spatial and channel dimensions.
Furthermore, the GFP module effectively restores the learned information using its unique structure. Comprehensive evaluations on five degraded document image datasets demonstrate that RA-GAN outperforms state-of-the-art methods across all metrics, with the lowest number of 
trainable parameters and computational costs. Additionally, model extension experiments and ablation studies further validate RA-GAN’s effectiveness and potential for future development.

## Models
## Prerequisites
- Linux (Ubuntu)/Windows
- Python >= 3.8
- NVIDIA GPU + CUDA CuDNN


## Model train/eval
- datasets referenced in this paper
  - [(H)DIBCO](https://vc.ee.duth.gr/dibco2019/)
  - [Label](https://www.dropbox.com/sh/gqqugvclzltfldt/AACNELpHwTW-1bHLZzipxQWja?dl=0)
  - [PHIDB](http://www.iapr-tc11.org/mediawiki/index.php/Persian_Heritage_Image_Binarization_Dataset_(PHIBD_2012))
  - [Bickly-diary](https://github.com/vqnhat/DSN-Binarization/files/2793688/original_gt_labeled.zip)
  - [PLM](http://amadi.univ-lr.fr/ICFHR2016_Contest/index.php/download-123)
- other datasets
  - [CHB](https://pan.baidu.com/s/1ymF08smuRhTo69dG_7gYUw?pwd=h4fu)(This is a dataset of image binarization of degenerate documents in ancient Chinese in Huizhou for readers to experiment)
  - [S-MS](http://tc11.cvc.uab.es/datasets/SMADI_1)
  - [LRDE-DBD](https://www.lrde.epita.fr/dload/olena/datasets/dbd/1.0/)
 - Patch per datasets
You can get the data set you need for training from the links below, and use the utils toolkit to capture image patches, 
as well as get edge maps and mask maps. With all this ready, 
you can run the training script (train.py).
 ```bash
 python3 crop.py
 python3 edge.py
 python3 ostuandsobel.py
 ```
- Train a model per datasets
```bash
python3 train.py
```

- Evaluate the model per datasets
<!--
(our pre-trained models are in ./pretrained_model)
- We plan to upload the pre-trained models on our Github page.
-->
```bash
python3 test.py
```

