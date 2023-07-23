# Neural Active Learning on Heteroskedastic Distributions
An implementation for the paper **Understanding and Improving Neural Active Learning on Heteroskedastic Distributions**, published in ECAI 2023.

The proposed active learning pipeline seeks to facilitate machine learning under the presence of heteroskedastic noise by filtering out the noisy data samples from the pool of unlabelled data and selecting the most informative samples for training the model. To this end, our algorithm leverages the change in a model's state (loss and later-layer feature representations) to choose a diverse set of clean and challenging training samples from the unlabelled data pool. Further, we propose a simple fine-tuning mechanism that uses high-confidence samples in the unlabelled data pool to further improve feature learning and boost the model's performance.

![teaser](https://github.com/heteroskedastic-dist/Active-Learning-on-Heteroskedastic-Distributions/blob/main/teaser.png?raw=true)

This repository experiments with 3 synthetic noise settings - *noisy-blank*, *noisy-diverse*, and *noisy-class*. In each case, a large number of noisy examples are mixed with the clean examples in the unlabelled data pool.

<p align="center">
  <img src="https://github.com/heteroskedastic-dist/Active-Learning-on-Heteroskedastic-Distributions/blob/main/noise.png?raw=true" width="750">
</p>


## Dependencies
- Python 3.6.8
- Pytorch 1.4.0
- Scikit-learn


## Running an experiment

Use the script `run.py` to run an experiment. The command-line flags are:

`--alg` specifies the active learning algorithm to use

`--model` specifies the machine learning model to use for active learning

`--nQuery` sets the number of samples to be queried in each active learning iteration

`--data` specifies the dataset to perform active learning on

`--mult` multiplier specifying the size of noisy examples compared to the clean examples. For e.g. if `--mult=4`, the number of noisy examples will be 4x the number of clean examples.

`--mode` specifies the type of noise-setting

### Example runs

The command below runs an experiment with ResNet18 and *noisy-class* CIFAR10, querying batches of 1000 data points using the LHD algorithm. The number of noisy examples is 4x the number of clean examples.
```
python run.py --alg lhd --model resnet --nQuery 1000 --data CIFAR10 --mult 4 --mode class
```

To run an experiment using MLP and *noisy-diverse* SVHN, querying batches of 1000 data points using the BADGE algorithm:
```
python run.py --alg badge --model mlp --nQuery 1000 --data SVHN --mult 4 --mode diverse
```


## Citation
```
@misc{khosla2022neural,
      title={Neural Active Learning on Heteroskedastic Distributions}, 
      author={Savya Khosla and Chew Kin Whye and Jordan T. Ash and Cyril Zhang and Kenji Kawaguchi and Alex Lamb},
      year={2022},
      eprint={2211.00928},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
