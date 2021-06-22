# Neural TDOA

## Installation
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


## Experiments

### Scenario 1
A first experiment was carried by placing two microphones and a source within a 5x3x3 room. The microphones were fixed at (1, 1, 1) and (2, 2, 1). In turn, the source was able to move horizontally, having therefore a position of (x, y, 1).

1000 samples were generated under this scenario, 700 for training and 300 for validation.
The model achieved 1% error in the validation set within 4 epochs, and then started overfitting