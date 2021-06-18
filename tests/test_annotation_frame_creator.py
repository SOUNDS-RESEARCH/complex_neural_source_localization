import shutil
import os

from preprocessing.annotation_frame_creator import (
    _make_frames_for_target
)


def test_make_frames_for_target():
    output_dir_path = "tests/temp/csv_frames"
    shutil.rmtree(output_dir_path, ignore_errors=True)
    _make_frames_for_target(
        "tests/fixtures/fold1_room1_mix001.csv",
        output_dir_path
    )

    assert len(os.listdir(output_dir_path)) == 115

