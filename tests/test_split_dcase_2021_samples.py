import shutil

from preprocessing.split_dcase_2021_samples import (
    split_dcase_2021_samples
)


def test_split_dcase_2021_samples():

    output_dir_path = "tests/temp/splitted_sample"

    split_dcase_2021_samples(
        "tests/fixtures",
        "tests/fixtures",
        output_dir_path
    )

    shutil.rmtree(output_dir_path)
