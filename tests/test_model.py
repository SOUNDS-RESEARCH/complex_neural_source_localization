import shutil

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from datasets.dataset import TdoaDataset
from neural_tdoa.model import TdoaCrnn


def test_tdoa_crnn10_with_stft():
    _test_tdoa_crnn10("stft")


def test_tdoa_crnn10_with_mfcc():
    _test_tdoa_crnn10("mfcc")


def _test_tdoa_crnn10(feature_type):
    GlobalHydra.instance().clear()
    temp_dataset_path = "tests/temp/dataset"
    shutil.rmtree(temp_dataset_path, ignore_errors=True)

    initialize(config_path="../config", job_name="test_app")
    cfg = compose(config_name="config")
    cfg["training_dataset"]["training_dataset_dir"] = temp_dataset_path
    cfg["training_dataset"]["n_training_samples"] = 1
    cfg["model"]["feature_type"] = feature_type

    model = TdoaCrnn(cfg["model"], cfg["training_dataset"])
    dataset = TdoaDataset(cfg["training_dataset"])
    
    sample = dataset[0]
    _ = sample[1]

    model_output = model(sample[0].unsqueeze(0))

    assert model_output.shape == (1, 1)
