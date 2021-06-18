# YOHO (You Only Hear Once): Towards End-to-End Sound Event Localization and Detection 

## Installation
### Requirements
* Python 3

run `pip install -r requirements.txt` to install the python libraries needed

## Unit tests
To execute all unit tests, run either:

`pytest tests`
or
`make tests`

## Data used
The data used comes from The [DCase 2021 Task 3 Competition](http://dcase.community/challenge2021/task-sound-event-localization-and-detection). In order to download the data, you can run the command `python -m datasets.dcase_2021.download` to download the files. I found out that multipart zip files are a demonic thing to work with Python, so unzipping those is unfortunately manual.

## Model History

The first model developed is located at `yoho/models/yoho_model`. This model attempts to perform sound event localization and detection using a single CRNN model. Although the loss function is reduced in the model, the precision/recall/f1 metrics stay awful, so I do not think it is a success.

The second model is located at `yoho/models/sed` tries to do something simpler: To say what signals are contained within each 1s frame. During training using a learning rate of 1e-5, the loss goes down steady and recall increases steadily, at the cost of the precision.