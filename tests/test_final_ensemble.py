"""Contract, safety and parity checks for the final EXP-039 ensemble."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.ensembles import final_ensemble as migrated
from src import finalize_exp039_refined_blend as legacy


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _oof(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _exp027_oof() -> pd.DataFrame:
    frames = [_oof(path) for path in migrated.EXP027_SEED_PATHS]
    exp022 = _oof(migrated.EXP022_OOF_PATH)
    ids = frames[0]["id"]
    for frame in [*frames[1:], exp022]:
        assert frame["id"].equals(ids)
        assert frame["y_true"].equals(frames[0]["y_true"])
    seed_mean = (
        0.4 * frames[0]["oof_prediction"].to_numpy(np.float64)
        + 0.3 * frames[1]["oof_prediction"].to_numpy(np.float64)
        + 0.3 * frames[2]["oof_prediction"].to_numpy(np.float64)
    )
    values = 0.75 * seed_mean + 0.25 * exp022["oof_prediction"].to_numpy(np.float64)
    return pd.DataFrame({"id": ids, "y_true": frames[0]["y_true"], "prediction": values})


def test_wrapper_reexports_public_api_and_imports_from_any_cwd(tmp_path: Path) -> None:
    for name in migrated.__all__:
        assert getattr(legacy, name) is getattr(migrated, name)
    top_level = importlib.import_module("finalize_exp039_refined_blend")
    assert top_level.main is migrated.main
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "finalize_exp039_refined_blend.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert migrated.HISTORICAL_SUBMISSION_PATH.name in result.stdout


def test_configuration_is_exact() -> None:
    assert migrated.FINAL_WEIGHTS == {"EXP-027": 0.375, "EXP-037": 0.225, "EXP-039": 0.4}
    assert migrated.LEGACY_WEIGHTS == {"EXP-027": 0.4375, "EXP-037": 0.2625, "EXP-039": 0.3}
    assert migrated.BLEND_METHOD == "rank"
    assert migrated.HISTORICAL_OOF_AUC == 0.9674680607837304


def test_rank_normalization_preserves_average_ties() -> None:
    values = np.array([3.0, 1.0, 3.0, 2.0])
    expected = (pd.Series(values).rank(method="average").to_numpy(np.float64) - 1) / 3
    np.testing.assert_array_equal(migrated.normalized_rank(values), expected)


def test_synthetic_rank_blend_is_deterministic_and_preserves_ids() -> None:
    ids = pd.Series(np.arange(1, 101))
    raw = {
        "EXP-027": np.linspace(0.1, 0.9, 100),
        "EXP-037": np.sin(np.arange(100)) / 4 + 0.5,
        "EXP-039": np.linspace(0.9, 0.1, 100),
    }
    ranked = {name: migrated.normalized_rank(values) for name, values in raw.items()}
    first = migrated.build_final_submission(ids, ranked)
    second = migrated.build_final_submission(ids, ranked)
    pd.testing.assert_frame_equal(first, second)
    assert first["id"].equals(ids)
    assert first["addicted_label"].between(0, 1).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_validation_rejects_nonfinite_predictions(bad: float) -> None:
    frame = pd.DataFrame({"id": [1, 2], "prediction": [0.2, bad]})
    with pytest.raises(ValueError, match="finite"):
        migrated.validate_prediction_frame(frame, "prediction")


def test_validation_rejects_duplicates_alignment_and_length() -> None:
    duplicate = pd.DataFrame({"id": [1, 1], "prediction": [0.2, 0.3]})
    with pytest.raises(ValueError, match="duplicate"):
        migrated.validate_prediction_frame(duplicate, "prediction")
    frame = pd.DataFrame({"id": [2, 1], "prediction": [0.2, 0.3]})
    with pytest.raises(ValueError, match="not aligned"):
        migrated.validate_prediction_frame(frame, "prediction", expected_ids=[1, 2])
    with pytest.raises(ValueError, match="length mismatch"):
        migrated.validate_prediction_frame(frame, "prediction", expected_ids=[1, 2, 3])
    with pytest.raises(ValueError, match="length mismatch"):
        migrated.blend_predictions({
            "EXP-027": [0.1, 0.2], "EXP-037": [0.2], "EXP-039": [0.3, 0.4]
        })


def test_exp039_real_oof_score_and_fold_parity() -> None:
    required = [*migrated.EXP027_SEED_PATHS, migrated.EXP022_OOF_PATH,
                migrated.EXP037_OOF_PATH, migrated.EXP039_OOF_PATH]
    if not all(path.exists() for path in required):
        pytest.skip("Historical OOF artifacts are unavailable")
    exp027 = _exp027_oof()
    exp037 = _oof(migrated.EXP037_OOF_PATH)
    exp039 = _oof(migrated.EXP039_OOF_PATH)
    assert exp027["id"].equals(exp037["id"])
    assert exp027["id"].equals(exp039["id"])
    assert exp027["y_true"].equals(exp037["y_true"])
    assert exp027["y_true"].equals(exp039["y_true"])
    blend = migrated.blend_predictions({
        "EXP-027": migrated.normalized_rank(exp027["prediction"]),
        "EXP-037": migrated.normalized_rank(exp037["oof_prediction"]),
        "EXP-039": migrated.normalized_rank(exp039["oof_prediction"]),
    })
    y = exp027["y_true"].to_numpy()
    # The historical score was computed before CSV serialization. Rebuilding from
    # persisted float text changes only a few rank comparisons (~1e-11 overall AUC).
    assert roc_auc_score(y, blend) == pytest.approx(migrated.HISTORICAL_OOF_AUC, abs=1e-9)
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    folds = tuple(roc_auc_score(y[valid], blend[valid]) for _, valid in splitter.split(blend, y))
    np.testing.assert_allclose(folds, migrated.HISTORICAL_FOLD_AUCS, rtol=0, atol=1e-9)


def test_historical_submission_rebuild_is_exact_and_alternate_output(tmp_path: Path) -> None:
    required = [migrated.HISTORICAL_SUBMISSION_PATH, migrated.EXP027_TEST_PATH,
                migrated.EXP037_TEST_PATH, ROOT / "data" / "sample_submission.csv"]
    if not all(path.exists() for path in required):
        pytest.skip("Historical test artifacts are unavailable")
    historical = pd.read_csv(migrated.HISTORICAL_SUBMISSION_PATH)
    rebuilt = migrated.reconstruct_historical_submission()
    pd.testing.assert_series_equal(rebuilt["id"], historical["id"])
    np.testing.assert_allclose(
        rebuilt["addicted_label"], historical["addicted_label"], rtol=0, atol=2e-16
    )
    destination = tmp_path / "submission.csv"
    migrated.main(destination)
    written = pd.read_csv(destination)
    pd.testing.assert_frame_equal(written, rebuilt)


def test_import_has_no_historical_side_effects() -> None:
    paths = [migrated.HISTORICAL_SUBMISSION_PATH, migrated.METRIC_PATH,
             migrated.BLEND_REPORT_PATH, ROOT / "outputs" / "metrics" / "experiment_log.csv"]
    before = {path: _sha256(path) for path in paths if path.exists()}
    result = subprocess.run(
        [sys.executable, "-c", "import src.ensembles.final_ensemble; import src.finalize_exp039_refined_blend"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert {path: _sha256(path) for path in before} == before
