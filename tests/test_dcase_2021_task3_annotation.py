from datasets.dcase_2021.dcase_2021_task3_annotation import (
    _normalize_time,
    _normalize_angles,
    load_csv_as_dataframe
)


def test_normalize_time():
    annotation_df = load_csv_as_dataframe(
        "tests/fixtures/fold1_room1_mix001.csv")

    yoho_df_normalized = _normalize_time(annotation_df, 60)

    assert (yoho_df_normalized["start_time"] >= 0).all()
    assert (yoho_df_normalized["start_time"] <= 1.0 + 10e-7).all()
    assert (yoho_df_normalized["end_time"] >= 0).all()
    assert (yoho_df_normalized["end_time"] <= 1.0 + 10e-7).all()


def test_normalize_angles():
    annotation_df = load_csv_as_dataframe(
        "tests/fixtures/fold1_room1_mix001.csv")

    normalized_df = _normalize_angles(annotation_df)

    assert (normalized_df["azimuth"] >= 0).all()
    assert (normalized_df["azimuth"] <= 1).all()
    assert (normalized_df["elevation"] >= 0).all()
    assert (normalized_df["elevation"] <= 1).all()
