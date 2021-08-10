import shutil
import torch

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from neural_tdoa.metrics import Loss
from neural_tdoa.model import TdoaCrnn10
from datasets.dataset import TdoaDataset


def test_neural_tdoa_loss():
    
    cfg = _load_config()

    loss_fn = Loss()
    model = TdoaCrnn10(cfg["model"], cfg["dataset"])

    dataset = TdoaDataset(cfg["dataset"])

    sample = dataset[0]
    target = sample["targets"]

    model_output = model(sample["signals"].unsqueeze(0))
    
    _ = loss_fn(model_output, torch.Tensor([[target]]))


def _load_config():
    GlobalHydra.instance().clear()
    temp_dataset_path = "tests/temp/dataset"
    shutil.rmtree(temp_dataset_path, ignore_errors=True)

    initialize(config_path="../config", job_name="test_app")
    cfg = compose(config_name="config")
    cfg["dataset"]["training_dataset_dir"] = temp_dataset_path
    cfg["dataset"]["n_training_samples"] = 1
    cfg["model"]["feature_type"] = "stft_magnitude"

    return cfg