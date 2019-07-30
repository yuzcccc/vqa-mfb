# MFB and MFH for VQA

**This project is deprecated! The Pytorch implementation of MFB(MFH)+CoAtt with pre-trained models, along with several state-of-the-art VQA models are maintained in our [OpenVQA](https://github.com/MILVLG/openvqa) project, which is much more convenient to use!**

This project is the implementation of the papers *[Multi-modal Factorized Bilinear Pooling with Co-Attention Learning for Visual Question Answering (MFB)](https://arxiv.org/abs/1708.01471)* and *[Beyond Bilinear: Generalized Multi-modal Factorized High-order Pooling for Visual Question Answering (MFH)](https://arxiv.org/abs/1708.03619)*. Compared with existing state-of-the-art approaches such as MCB and MLB, our MFB models achieved superior performance on the large-scale VQA-1.0 and VQA-2.0 datasets. Moreover, MFH, the high-order extention of MFB, is also proveided to report better VQA performance. The MFB(MFH)+CoAtt network architecture for VQA is illustrated in Figure 1. 

![Figure 1: The MFB+CoAtt Network architecture for VQA.](https://github.com/yuzcccc/mfb/raw/master/imgs/MFB-github.png)
<center>Figure 1: The MFB+CoAtt Network architecture for VQA.</center>

## Update Dec. 2nd, 2017
The 3rd-party pytorch implementation for MFB(MFH) is released [here](https://github.com/asdf0982/vqa-mfb.pytorch). Great thanks, Liam!

## Update Sep. 5th, 2017
Using the Bottom-up and Top-Down (BUTD) image features (the model with adaptive K ranges from [10,100]) [here](https://github.com/yuzcccc/bottom-up-attention), our single MFH+CoAtt+GloVe model achieved the overall accuracy **68.76%** on the test-dev set of VQA-2.0 dataset. With an ensemble of 8 models, we achieved the new state-of-the-art performance on the VQA-2.0 dataset's [leaderboard](https://evalai.cloudcv.org/web/challenges/challenge-page/1/leaderboard) with the overall accuracy **70.92%**. 

## Update Aug. 1st, 2019
Our solution for the VQA Challenge 2017 is updated! 

We proposed a **high-order** extention for MFB, i.e., the Multi-modal Factorized High-order Pooling (MFH). See the flowchart in Figure 2 and the implementations in `mfh_baseline` and `mfh-coatt-glove` folders. With an ensemble of 9 MFH+CoAtt+GloVe(+VG) models, **we won the 2nd place (tied with another team) in the VQA Challenge 2017**. The detailed information can be found in our paper (the second paper in the CITATION section on bottom of this page). 

![](https://github.com/yuzcccc/mfb/raw/master/imgs/MFH-github.png)
<center>Figure 2: The high-order MFH model which consists of p MFB blocks (without sharing parameters).</center>

## Prerequisites

Our codes is implemented based on the high-quality [vqa-mcb](https://github.com/akirafukui/vqa-mcb) project. The data preprocessing and and other prerequisites are the same with theirs. Before running our scripts to train or test MFB model, see the `Prerequisites` and `Data Preprocessing` sections in the README of vqa-mcb's project first. 

- The Caffe version required for our MFB is slightly different from the MCB. We add some layers, e.g., sum pooling, permute and KLD loss layers to the `feature/20160617_cb_softattention` branch of Caffe for MCB. Please checkout our caffe version [here](https://github.com/yuzcccc/caffe) and compile it. **Note that CuDNN is not compatible with sum pooling currently, you should switch it off to run the codes correctly**.

## Pretrained Models

We release the pretrained **single model** "MFB(or MFH)+CoAtt+GloVe+VG" in the papers. To the best of our knowledge, our MFH+CoAtt+GloVe+VG model report the best result (test-dev) with a single model on both the VQA-1.0 and VQA-2.0 datasets(train + val + visual genome). The corresponding results are shown in the table below. The results JSON files (results.zip for VQA-1.0) are also included in the model folders, which can be uploaded to the evaluation servers directly. **Note that the models are trained with a old version of GloVe in spacy. If you use the latest one, they maybe incosistent, leading to inferior performance. I suggest training the model from scratch by yourself.**

|   Datasets\Models    | MCB | MFB | MFH  | MFH (BUTD img features) |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| VQA-1.0   | 65.38%   |66.87% [BaiduYun](http://pan.baidu.com/s/1o8LURge)   | 67.72% [BaiduYun](http://pan.baidu.com/s/1c2neUv2) or [Dropbox](https://www.dropbox.com/s/qh1swgsq0na1bua/VQA1.0-mfh-coatt-glove-vg.zip?dl=0) | **69.82%** |
| VQA-2.0   | 62.33%<sup>1</sup>   |65.09% [BaiduYun](http://pan.baidu.com/s/1pLjtkSV)   | 66.12% [BaiduYun](http://pan.baidu.com/s/1pLLUvIN) or [Dropbox](https://www.dropbox.com/s/zld15405a69how6/VQA2.0-mfh-coatt-glove-vg.zip?dl=0) | **68.76%**<sup>2</sup> |

<sup>1</sup> the MCB result on VQA-2.0 is provided by the VQA Challenge organizer with does not introdunce the GloVe embedding.

<sup>2</sup> overall: 68.76, yes/no: 84.27, num: 49.56, other: 59.89

## Training from Scratch

We provide the scripts for training two MFB models from scratch, i.e., `mfb-baseline` and `mfb-coatt-glove` folders. Simply running the python scripts `train_*.py` to train the models from scratch. 

- Most of the hyper-parameters and configrations with comments are defined in the `config.py` file. 
- The solver configrations are defined in the `get_solver` function in the `train_*.py` scripts. 
- Pretrained GloVe word embedding model (the spacy library) is required to train the mfb-coatt-glove model. The installation instructions of spacy and GloVe model can be found [here](https://github.com/akirafukui/vqa-mcb/tree/master/train).

## Evaluation

To generate an answers JSON file in the format expected by the VQA evaluation code and VQA test server, you can use `eval/ensemble.py`. This code can also ensemble multiple models. Running `python ensemble.py` will print out a help message telling you what arguments to use.

## Licence

This code is distributed under MIT LICENSE. The released models are only allowed for non-commercial use.

## Citation

If the codes are helpful for your research, please cite

```
@article{yu2017mfb,
  title={Multi-modal Factorized Bilinear Pooling with Co-Attention Learning for Visual Question Answering},
  author={Yu, Zhou and Yu, Jun and Fan, Jianping and Tao, Dacheng},
  journal={IEEE International Conference on Computer Vision (ICCV)},
  pages={1839--1848},
  year={2017}
}

@article{yu2018beyond,
  title={Beyond Bilinear: Generalized Multimodal Factorized High-Order Pooling for Visual Question Answering},
  author={Yu, Zhou and Yu, Jun and Xiang, Chenchao and Fan, Jianping and Tao, Dacheng},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  volume={29},
  number={12},
  pages={5947--5959},
  year={2018}
}
```

## Concat

Zhou Yu  [yuz(AT)hdu.edu.cn]
