import logging
from neural_tdoa.models.tdoa_simple_cnn import TdoaSimpleCnn
import shutil

from datasets.dataset import TdoaDataset
from neural_tdoa.train import train
from neural_tdoa.loss import Loss
from neural_tdoa.callbacks import make_callbacks

from neural_tdoa.models.tdoa_simple_cnn import TdoaSimpleCnn
from neural_tdoa.models.tdoa_crnn10 import TdoaCrnn10
from neural_tdoa.models.tdoa_cnn14 import TdoaCnn14
from neural_tdoa.models.tdoa_dccrn import TdoaDCCRN


NUM_TRAIN_SAMPLES = 350
NUM_VALIDATION_SAMPLES = 150

VALIDATION_DIR = "tests/temp/validation_dataset_dir"
TRAIN_DIR = "tests/temp/train_dataset_dir"


def test_train(regenerate_datasets=False):
    _setup(regenerate_datasets)

    model = TdoaDCCRN()

    logging.info(f"Creating training dataset with {NUM_TRAIN_SAMPLES}")
    dataset_train = TdoaDataset(n_samples=NUM_TRAIN_SAMPLES, dataset_dir=TRAIN_DIR)

    logging.info(f"Creating validation dataset with {NUM_VALIDATION_SAMPLES}")
    dataset_val = TdoaDataset(n_samples=NUM_VALIDATION_SAMPLES, dataset_dir=VALIDATION_DIR)

    loss_function = Loss()

    train(model, loss_function, dataset_train, dataset_val,
          callbacks=make_callbacks(), batch_size=32)


def _setup(regenerate_datasets):
    if regenerate_datasets:
        shutil.rmtree(TRAIN_DIR, ignore_errors=True)
        shutil.rmtree(VALIDATION_DIR, ignore_errors=True)

if __name__ == "__main__":
    test_train(False)
