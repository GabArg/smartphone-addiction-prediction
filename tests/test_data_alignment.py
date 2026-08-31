import pandas as pd
import pytest

from src.project_paths import DATA_DIR

EXPECTED_FEATURES = {
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day", "app_opens_per_day",
    "weekend_screen_time", "gender", "stress_level", "academic_work_impact",
}


def load_competition_data():
    train_path, test_path = DATA_DIR / "train.csv", DATA_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        pytest.skip("Competition data is not available locally")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def test_train_test_schema_and_ids():
    train, test = load_competition_data()
    assert "addicted_label" in train.columns
    assert "addicted_label" not in test.columns
    assert train["id"].is_unique and test["id"].is_unique
    assert EXPECTED_FEATURES <= set(train.columns) and EXPECTED_FEATURES <= set(test.columns)
    assert set(train.columns) - {"addicted_label"} == set(test.columns)


def test_sample_submission_matches_test_order():
    _, test = load_competition_data()
    path = DATA_DIR / "sample_submission.csv"
    if not path.exists(): pytest.skip("sample_submission.csv is not available locally")
    submission = pd.read_csv(path)
    assert submission["id"].is_unique
    assert submission["id"].equals(test["id"])
