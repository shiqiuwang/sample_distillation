# sample_distillation

source code on caltech256 dataset in paper:Hardness-Aware Sample Distillation for Efficient Model Training

## Instructions for use

### data preparation

The original dataset and pre-training related files are in the release. First, you need to clone the Caltech256 folder to the local, and then go to release to download the relevant data and files and unzip them to the root directory of the Caltech256 folder.

### Related code file introduction

focalloss.py :It is the implementation code of focalloss function

SPLD.py:it is the implementation code of Self-paced learning function

mcmc.py:it is the Implementation of Monte Carlo Algorithm

selected_by_mcmc.py: the source code of Sampling the original data set with Monte Carlo

spld_train.py:Use reverse self-paced learning and focalloss for model training on the original data set

alexnet_train_imgs_by_mcmc.py:Train alexnet with the sampled data set

### run

training the model with reverse self-learning,run the following command

```python
python spld_train.py
```



sampleing data, run the following command

```python
python selected_by_mcmc.py
```

Useing the sampled data to train the model run the following command

```python
python alexnet_train_imgs_by_mcmc.py
```


