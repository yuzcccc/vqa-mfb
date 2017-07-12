# Multi-modal Factorized Bilinear Pooling (MFB)

This project is the implementation of the paper **Multi-modal Factorized Bilinear Pooling with Co-Attention Learning for Visual Question Answering**. Compared with existing state-of-the-art approaches such as MCB and MLB, our MFB models achieved superior performance on the large-scale VQA-1.0 and VQA-2.0 datasets. 

## Prerequisites

Our code is implemented based on the [vqa-mcb](https://github.com/akirafukui/vqa-mcb) project. The data preprocessing and and other prerequisites are the same with theirs. Before running our scripts to train or test MFB model, see the `Prerequisites` and `Data Preprocessing` sections in the README of vqa-mcb's project first. 

- The Caffe version required for our MFB is slightly different from the MCB. We add some layers, e.g., sum pooling, permute and KLD loss layers to the `feature/20160617_cb_softattention` branch of Caffe for MCB. Please checkout our caffe version [here](https://github.com/yuzcccc/caffe) and compile it. Note that CuDNN is not compatible with sum pooling currently, you should switch it off to run the codes correctly.

## Pretrained Models

We release the "MFB+CoAtt+GloVe+VG" models from the paper, which achieves **66.87%** on real open-ended test-dev of VQA-1.0 dataset (65.38% for MCB), and **65.09%** on the test-dev of VQA-2.0 (VQA Challenge 2017) dataset. The results JSON files (results.zip for VQA-1.0) are also included in the model folders, which can be uploaded to the evaluation servers directly.
- [VQA-1.0 model](http://pan.baidu.com/s/1o8LURge) on the BaiDuYun
- [VQA-2.0 model](http://pan.baidu.com/s/1pLjtkSV) on the BaiduYun

## Training from Scratch

We provide the scripts for training two MFB models from scratch, i.e., `mfb-baseline` and `mfb-coatt-glove` folders. Simply running the python scripts `python train_mfb_baseline.py` or `python train_mfb_coatt_glove.py` will start training from scratch. 

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
@article{zhou2017mfb,
  title={Multi-modal Factorized Bilinear Pooling with Co-Attention Learning for Visual Question Answering},
  author={Zhou, Yu and Jun, Yu and Jianping, Fan and Dacheng, Tao},
  journal={IEEE International Conference on Computer Vision (ICCV)},
  year={2017},
}
```

## Concat

Zhou Yu  [yuz(AT)hdu.edu.cn]
