# Complex neural source localization
This repository contains the code for the paper
"DEEP COMPLEX-VALUED CONVOLUTIONAL-RECURRENT NETWORKS FOR SINGLE SOURCE DOA ESTIMATION" to be published at the
International Workshop on Acoustic Signal Enhancement (IWAENC) 2022.

## Installation

To test the code without installing anything, we suggest running it using this [Kaggle notebook](https://www.kaggle.com/code/egrinstein/neural-doa-training-notebook). To install it locally, follow the instructions below.


### Requirements
* Python 3

run `pip install -r requirements.txt` to install the python libraries needed

## Unit tests
To execute all unit tests, run either:

`pytest tests`
or
`make tests`

## Generate synthetic data
Run `python -m datasets.generate_dataset` to generate synthetic data at the default generated_dataset/ directory
