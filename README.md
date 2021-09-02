# IKTN
Code for [An Iterative Knowledge Transfer Network with Routing for Aspect-based Sentiment Analysis](https://arxiv.org/abs/2004.01935) of EMNLP21 Findings (early version)

## Introduction

The implementation is based on [IMN](https://github.com/ruidan/IMN-E2E-ABSA) and the dataset we used is also from IMN. Please download [Glove](http://nlp.stanford.edu/data/glove.840B.300d.zip) file and the bert-based feature used in this Repo is produced by [BERT](https://github.com/google-research/bert).

## Usage

Training and evaluating with the following scripts (the Hyper-parameters are shown in 'train.py' file and you may change them for better results): 

```
bash train.sh
```

## Requirements

+ Python 2.7
+ Keras 2.2.4
+ tensorflow 1.4.1
+ numpy 1.13.3
## Citation

If you find this project helps, please cite the following paper :)

```
@misc{liang2020iterative,
      title={An Iterative Knowledge Transfer Network with Routing for Aspect-based Sentiment Analysis}, 
      author={Yunlong Liang and Fandong Meng and Jinchao Zhang and Jinan Xu and Yufeng Chen and Jie Zhou},
      year={2020},
      eprint={2004.01935},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
