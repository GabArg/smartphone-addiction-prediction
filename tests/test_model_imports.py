"""Compatibility and configuration checks for the migrated EXP-001 model."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import train_logistic_baseline as legacy
from src.models import baseline_logistic as migrated


ROOT = Path(__file__).resolve().parents[1]

# Static snapshot captured from src/train_logistic_baseline.py before migration.
EXPECTED_CONSTANTS = {
    "TARGET": "addicted_label",
    "ID_COLUMN": "id",
    "EXPERIMENT_ID": "EXP-001",
    "RANDOM_STATE": 42,
    "N_SPLITS": 5,
}
EXPECTED_OUTPUT_NAMES = {
    "SUBMISSION_PATH": "submission_exp001_logistic.csv",
    "METRICS_PATH": "exp001_logistic_metrics.txt",
    "OOF_PATH": "exp001_logistic_oof.csv",
    "LOG_PATH": "experiment_log.csv",
}
HISTORICAL_OUTPUTS = (
    ROOT / "outputs" / "metrics" / "experiment_log.csv",
    ROOT / "outputs" / "metrics" / "exp001_logistic_metrics.txt",
    ROOT / "outputs" / "metrics" / "exp001_logistic_oof.csv",
    ROOT / "outputs" / "submissions" / "submission_exp001_logistic.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_legacy_wrapper_reexports_public_api() -> None:
    for name, expected in EXPECTED_CONSTANTS.items():
        assert getattr(migrated, name) == expected
        assert getattr(legacy, name) == expected

    for name in ("build_pipeline", "update_experiment_log", "validate_submission", "main"):
        assert getattr(legacy, name) is getattr(migrated, name)


def test_historical_configuration_is_preserved() -> None:
    pipeline = migrated.build_pipeline(["numeric"], ["categorical"])
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric = preprocessor.transformers[0][1]
    categorical = preprocessor.transformers[1][1]
    model = pipeline.named_steps["model"]

    assert numeric.named_steps["imputer"].strategy == "median"
    assert numeric.named_steps["scaler"].with_mean is True
    assert numeric.named_steps["scaler"].with_std is True
    assert categorical.named_steps["imputer"].strategy == "most_frequent"
    assert categorical.named_steps["onehot"].handle_unknown == "ignore"
    assert model.__class__.__name__ == "LogisticRegression"
    assert model.max_iter == 2000
    assert model.solver == "liblinear"

    for name, filename in EXPECTED_OUTPUT_NAMES.items():
        assert getattr(migrated, name).name == filename


def test_imports_have_no_historical_output_side_effects() -> None:
    existing = [path for path in HISTORICAL_OUTPUTS if path.exists()]
    hashes_before = {path: _sha256(path) for path in existing}

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.models.baseline_logistic; import src.train_logistic_baseline",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert {path: _sha256(path) for path in existing} == hashes_before


def test_small_deterministic_fit_predict_smoke() -> None:
    train_path = migrated.DATA_DIR / "train.csv"
    if not train_path.exists():
        return

    train = pd.read_csv(train_path, nrows=512)
    feature_columns = [
        column for column in train.columns if column not in {migrated.ID_COLUMN, migrated.TARGET}
    ]
    X = train[feature_columns]
    y = train[migrated.TARGET]
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    pipeline = migrated.build_pipeline(numeric_columns, categorical_columns)
    pipeline.fit(X, y)
    probabilities = pipeline.predict_proba(X.iloc[:32])[:, 1]

    assert probabilities.shape == (32,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_final_ensemble_imports_without_model_training() -> None:
    module = importlib.import_module("src.ensembles.final_ensemble")
    wrapper = importlib.import_module("src.finalize_exp039_refined_blend")
    assert wrapper.main is module.main
    assert module.BLEND_METHOD == "rank"
