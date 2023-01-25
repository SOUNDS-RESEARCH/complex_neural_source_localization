# Complex neural source localization
This repository contains the code for the paper
"Deep complex-valued convolutional-recurrent networks for single source doa estimation" to be published at the
International Workshop on Acoustic Signal Enhancement (IWAENC) 2022.

https://hal.science/hal-03779970/document
https://ieeexplore.ieee.org/abstract/document/9914747


## Installation

To test the code without installing anything, we suggest running it using this [Kaggle notebook](https://www.kaggle.com/code/egrinstein/neural-doa-training-notebook). To install it locally, follow the instructions below.


### Requirements
* Python 3

run `pip install -r requirements.txt` to install the python libraries needed

Download the [Kaggle dataset](https://www.kaggle.com/datasets/egrinstein/dcase-2019-single-source) containing the data, and change the file 'config/dcase_2019_task3_dataset.yaml' to point at the correct train, validation and test datasets.

Then, change the working directory to this project and run `python train.py` or `make train` to start training the model. Every time you start training a model, a folder will be created in the outputs/ 


## Unit tests
To execute all unit tests, run either:

`pytest tests`
or
`make tests`
`
