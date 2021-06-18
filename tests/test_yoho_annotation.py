from yoho.model.yoho_target import (
    NUM_YOHO_TIME_CELLS, YOHO_CELL_SIZE,
    make_yoho_target_from_dcase_2021_annotation
)

from datasets.dcase_2021.dcase_2021_task3_annotation import (
    load_csv_as_dataframe)


def test_make_yoho_target_from_dcase_2021_annotation():
    target_array = make_yoho_target_from_dcase_2021_annotation(
         "tests/fixtures/fold1_room1_mix001.csv")
    assert target_array.shape == (
        NUM_YOHO_TIME_CELLS, YOHO_CELL_SIZE)