# Multi-modal Factorized Bilinear Pooling (MFB) for VQA

This project is the implementation of the paper **Multi-modal Factorized Bilinear Pooling with Co-Attention Learning for Visual Question Answering**. Compared with existing state-of-the-art approaches such as MCB and MLB, our MFB models achieved superior performance on the large-scale VQA-1.0 and VQA-2.0 datasets. The MFB+CoAtt network architecture for VQA is illustrated in Figure 1. 

![Figure 1: The MFB+CoAtt Network architecture for VQA.](https://github.com/yuzcccc/mfb/raw/master/imgs/MFB-github.png)
<center>Figure 1: The MFB+CoAtt Network architecture for VQA.</center>

## Update!
Our solution for the VQA Challenge 2017 is updated! 

We proposed a **high-order** extention for MFB, i.e., the Multi-modal Factorized High-order Pooling (MFH). See the flowchart in Figure 2 and the implementations in `mfh_baseline` and `mfh-coatt-glove` folders. With an ensemble of 9 MFH+CoAtt+GloVe(+VG) models, **we won the 2nd place (tied with another team) in the VQA Challenge 2017**. The detailed information can be found in our paper (the second paper in the CITATION section on bottom of this page). 

![](https://github.com/yuzcccc/mfb/raw/master/imgs/MFH-github.png)
<center>Figure 2: The high-order MFH model which consists of p MFB blocks (without sharing parameters).</center>

## Prerequisites

Our codes is implemented based on the high-quality [vqa-mcb](https://github.com/akirafukui/vqa-mcb) project. The data preprocessing and and other prerequisites are the same with theirs. Before running our scripts to train or test MFB model, see the `Prerequisites` and `Data Preprocessing` sections in the README of vqa-mcb's project first. 

- The Caffe version required for our MFB is slightly different from the MCB. We add some layers, e.g., sum pooling, permute and KLD loss layers to the `feature/20160617_cb_softattention` branch of Caffe for MCB. Please checkout our caffe version [here](https://github.com/yuzcccc/caffe) and compile it. **Note that CuDNN is not compatible with sum pooling currently, you should switch it off to run the codes correctly**.

## Pretrained Models

We release the pretrained **single model** "MFB(or MFH)+CoAtt+GloVe+VG" in the papers. To the best of our knowledge, our MFH+CoAtt+GloVe+VG model report the best result with a single model on both the VQA-1.0 and VQA-2.0 datasets. The corresponding results are shown in the table below. The results JSON files (results.zip for VQA-1.0) are also included in the model folders, which can be uploaded to the evaluation servers directly.

|   Datasets\Models    | MCB | MFB | MFH  |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| VQA-1.0   | 65.38%   |66.87% [BaiduYun](http://pan.baidu.com/s/1o8LURge)   | **67.72%** [BaiduYun](http://pan.baidu.com/s/1c2neUv2) or [Dropbox](https://www.dropbox.com/s/qh1swgsq0na1bua/VQA1.0-mfh-coatt-glove-vg.zip?dl=0) |
| VQA-2.0   | 62.33%*  |65.09% [BaiduYun](http://pan.baidu.com/s/1pLjtkSV)   | **66.12%** [BaiduYun](http://pan.baidu.com/s/1pLLUvIN) or [Dropbox](https://www.dropbox.com/s/zld15405a69how6/VQA2.0-mfh-coatt-glove-vg.zip?dl=0) |
* the MCB result on VQA-2.0 is provided by the VQA Challenge organizer
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
  author={Zhou, Yu and Jun, Yu and Jianping, Fan and Dacheng, Tao},
  journal={IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}

@article{yu2017beyond,
  title={Beyond Bilinear: Generalized Multi-modal Factorized High-order Pooling for Visual Question Answering},
  author={Zhou, Yu and Jun, Yu and Chenchao, Xiang and Jianping, Fan and Dacheng, Tao},
  journal={arXiv preprint arXiv:1708.03619},
  year={2017}
}
```

## Concat

Zhou Yu  [yuz(AT)hdu.edu.cn]
