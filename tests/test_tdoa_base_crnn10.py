import shutil

from datasets.dataset import TdoaDataset
from neural_tdoa.models.tdoa_base_crnn10 import TdoaBaseCrnn10

def test_tdoa_base_crnn10():
    temp_dataset_path = "tests/temp/dataset"
    shutil.rmtree(temp_dataset_path, ignore_errors=True)

    model = TdoaBaseCrnn10()

    dataset = TdoaDataset(n_samples=1, dataset_dir=temp_dataset_path)

    sample = dataset[0]
    target = sample["targets"]

    model_output = model(sample["signals"].unsqueeze(0))

    assert model_output.shape == (1, 1)