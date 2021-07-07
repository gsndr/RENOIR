# autoencodeR-based neural nEtwork for INtrusiOn DetectIon Systems with tRiplet loss function (RENOIR)

The repository contains code refered to the work:

_Giuseppina Andresini, Annalisa Appice, Donato Malerba_

[Autoencoder-based Deep Metric Learning for Network Intrusion Detection](https://www.sciencedirect.com/science/article/pii/S002002552100462X)

Please cite our work if you find it useful for your research and work.

```
@article{ANDRESINI2021,
title = {Autoencoder-based deep metric learning for network intrusion detection},
journal = {Information Sciences},
year = {2021},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2021.05.016},
url = {https://www.sciencedirect.com/science/article/pii/S002002552100462X},
author = {Giuseppina Andresini and Annalisa Appice and Donato Malerba},
keywords = {Network intrusion detection, Deep metric learning, Triplet network, Autoencoder}
}

```




## Code requirements

The code relies on the following **python2.6+** libs.

Packages need are:
* [Tensorflow 1.13](https://www.tensorflow.org/) 
* [Keras 2.3](https://github.com/keras-team/keras) 
* [Pandas 0.23.4](https://pandas.pydata.org/)
* [Numpy 1.15.4](https://www.numpy.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)

## Data
The datasets used for experiments are accessible from [__DATASETS__](https://drive.google.com/drive/folders/1jLb9I80w16IU_AkWr9H5e7vTOcypQsLw?usp=sharing). 
The repository contains the dataset after the preprocessing phase (folder: "numeric") 
Corresponding labels: 
* 0: "Attacks"
* 1: "Normal"

Preprocessing phase is done mapping categorical feature and performing the Min Max scaler.

## How to use
Repository contains scripts of all experiments included in the paper:
* __main.py__ : script to run RENOIR 
To run the code the command is main.py NameOfDataset (es CICIDS2017, AAGM or KDDCUP99)
  
 Code contains models (autoencoder and classification) and datasets used for experiments in the work.
 
  

## Replicate the experiments

To replicate experiments reported in the work, you can use models and datasets stored in homonym folders.
Global variables are stored in __RENOIR.conf__  file 


```python
    N_CLASSES = 2
    PREPROCESSING1 = 0  #if set to 1 code execute preprocessing phase on original date
    LOAD_AUTOENCODER_N = 1 #if 1 the autoencoder for normal items  is loaded from models folder
    LOAD_AUTOENCODER_A = 1 #if 1 the autoencoder for attacks items  is loaded from models folder
    LOAD_NN = 1  #if 1 the classifier is loaded from models folder
    VALIDATION_SPLIT #the percentage of validation set used to train models
```

## Download datasets

[All datasets](https://drive.google.com/drive/folders/1jLb9I80w16IU_AkWr9H5e7vTOcypQsLw?usp=sharing)
