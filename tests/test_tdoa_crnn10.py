from datasets.settings import BASE_DATASET_CONFIG
import shutil

from datasets.dataset import TdoaDataset
from neural_tdoa.model import TdoaCrnn10


def test_tdoa_crnn10_with_stft():
    temp_dataset_path = "tests/temp/dataset"
    shutil.rmtree(temp_dataset_path, ignore_errors=True)

    model = TdoaCrnn10()

    dataset_config = BASE_DATASET_CONFIG
    dataset_config["n_training_samples"] = 1
    dataset_config["dataset_dir"] = temp_dataset_path
    dataset = TdoaDataset(dataset_config)
    
    sample = dataset[0]
    target = sample["targets"]

    model_output = model(sample["signals"].unsqueeze(0))

    assert model_output.shape == (1, 1)


def test_tdoa_crnn10_with_mfcc():
    temp_dataset_path = "tests/temp/dataset"
    shutil.rmtree(temp_dataset_path, ignore_errors=True)

    model = TdoaCrnn10(feature_type="mfcc")

    dataset_config = BASE_DATASET_CONFIG
    dataset_config["n_training_samples"] = 1
    dataset_config["dataset_dir"] = temp_dataset_path
    dataset = TdoaDataset(dataset_config)

    sample = dataset[0]
    target = sample["targets"]

    model_output = model(sample["signals"].unsqueeze(0))

    assert model_output.shape == (1, 1)
