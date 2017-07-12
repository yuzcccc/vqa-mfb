# Multi-modal Factorized Bilinear Pooling (MFB)

This project is the implementation of the paper **Multi-modal Factorized Bilinear Pooling with Co-Attention Learning for Visual Question Answering**. Compared with existing state-of-the-art approaches such as MCB and MLB, our MFB models achieved superior performance on the large-scale VQA-1.0 and VQA-2.0 datasets. if the codes in this project help your research, please cite our paper :) 

## Prerequisites

Our code is implemented based on the [vqa-mcb](https://github.com/akirafukui/vqa-mcb) project. The data preprocessing and and other prerequisites are the same with theirs. Before running our scripts to train or test MFB model, see the `Prerequisites` and `Data Preprocessing` sections in the README of vqa-mcb's project first. 

- The Caffe version required for our MFB is slightly different from the MCB. We add some layers, e.g., sum pooling, permute and KLD loss layers to the `feature/20160617_cb_softattention` branch of Caffe for MCB. Please checkout our caffe version [here](https://github.com/yuzcccc/caffe) and compile it. 

## Pre-trained Models

We release the "MCB+CoAtt+GloVe+VG" models from the paper, which achieves **66.87%** on real open-ended test-dev of VQA-1.0 dataset, and **65.09** on the test-dev of VQA-2.0 (VQA Challenge 2017) dataset
- [VQA-1.0 model](http://pan.baidu.com/s/1o8LURge) on the BaiDuYun
- [VQA-2.0 model](http://pan.baidu.com/s/1pLjtkSV) on the BaiduYun
- The results JSON files (results.zip for VQA-1.0) are also included in the model folders, which can be uploaded to the evaluation servers directly.

## Training from scratch

We provide the scripts for training two MFB models from scratch, i.e., `mfb-baseline` and `mfb-coatt-glove`. 

## Evaluate

To generate an answers JSON file in the format expected by the VQA evaluation code and VQA test server, you can use `eval/ensemble.py`. This code can also ensemble multiple models. Running `python ensemble.py` will print out a help message telling you what arguments to use.

## Citation
```
@article{zhou2017mfb,
  title={Multi-modal Factorized Bilinear Pooling with Co-Attention Learning for Visual Question Answering},
  author={Zhou, Yu and Jun, Yu and Jianping, Fan and Dacheng, Tao},
  journal={IEEE International Conference on Computer Vision (ICCV)},
  year={2017},
}
```
