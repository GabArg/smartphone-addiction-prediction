"""Compatibility, configuration and smoke checks for EXP-036/037."""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.features.relational import (
    FINAL_RATIO_RELATIONS,
    add_categorical_relations,
    build_base_representation,
)
from src.models import logistic_relational as migrated
from src import train_exp036_ratio_ablation as exp036
from src import train_exp037_relational_features as legacy


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_OUTPUTS = (
    ROOT / "outputs" / "metrics" / "experiment_log.csv",
    ROOT / "outputs" / "metrics" / "experiment_log_v2.csv",
    ROOT / "outputs" / "metrics" / "exp036_ratio_ablation_metrics.txt",
    ROOT / "outputs" / "metrics" / "exp037_relational_metrics.txt",
    ROOT / "outputs" / "predictions" / "oof_exp036_ratio_logistic.csv",
    ROOT / "outputs" / "predictions" / "test_exp036_ratio_logistic.csv",
    ROOT / "outputs" / "predictions" / "oof_exp037_relational_logistic.csv",
    ROOT / "outputs" / "predictions" / "test_exp037_relational_logistic.csv",
    ROOT / "outputs" / "submissions" / "submission_exp036_ratio_logistic_ensemble.csv",
    ROOT / "outputs" / "submissions" / "submission_exp037_relational_logistic_ensemble.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sample(rows: int = 3000) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if not (migrated.DATA / "train.csv").exists() or not (migrated.DATA / "test.csv").exists():
        pytest.skip("Competition train/test data is not available locally")
    train = pd.read_csv(migrated.DATA / "train.csv", nrows=rows)
    test = pd.read_csv(migrated.DATA / "test.csv", nrows=1000)
    return train, train["addicted_label"], test


def _sparse_fit_predict(x: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, object]:
    train_indices, valid_indices = train_test_split(
        np.arange(len(x)), test_size=0.2, random_state=42, stratify=y
    )
    encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32, sparse_output=True)
    train_matrix = sparse.csr_matrix(
        encoder.fit_transform(x.iloc[train_indices]), dtype=np.float32
    )
    valid_matrix = sparse.csr_matrix(
        encoder.transform(x.iloc[valid_indices]), dtype=np.float32
    )
    model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
    model.fit(train_matrix, y.iloc[train_indices])
    probabilities = model.predict_proba(valid_matrix)[:, 1]
    assert sparse.isspmatrix_csr(train_matrix) and sparse.isspmatrix_csr(valid_matrix)
    assert train_matrix.dtype == valid_matrix.dtype == np.float32
    return probabilities, model


def test_exp036_historical_entrypoint_and_exp037_wrapper_api() -> None:
    top036 = importlib.import_module("train_exp036_ratio_ablation")
    top037 = importlib.import_module("train_exp037_relational_features")
    assert exp036.base_rep is build_base_representation
    for name in (
        "rss", "feature_value", "add_features", "ckey", "fit_cv", "fs",
        "nested_blend", "nested_triple", "main",
    ):
        assert getattr(legacy, name) is getattr(migrated, name)
        assert getattr(top037, name) is getattr(migrated, name)
    for name in ("base_rep", "add_ratios", "fit_cv", "finalize_only", "main"):
        assert hasattr(top036, name)
    assert migrated.BASE_R == dict(FINAL_RATIO_RELATIONS)


def test_exp037_sparse_logistic_cv_final_features_and_outputs_are_preserved() -> None:
    source = Path(migrated.__file__).read_text(encoding="utf-8")
    assert "OneHotEncoder(handle_unknown='ignore',dtype=np.float32,sparse_output=True)" in source
    assert "sparse.csr_matrix(enc.fit_transform(x.iloc[tr]),dtype=np.float32)" in source
    assert "LogisticRegression(solver='liblinear',C=1.,max_iter=1000,random_state=42)" in source
    assert "StratifiedKFold(5,shuffle=True,random_state=42)" in source
    assert "current=[]" in source
    for filename in (
        "oof_exp037_relational_logistic.csv", "test_exp037_relational_logistic.csv",
        "submission_exp037_relational_logistic_ensemble.csv",
        "exp037_relational_metrics.txt", "exp037_difference_candidates.csv",
        "exp037_difference_granularity.csv", "exp037_inverse_ratios.csv",
        "exp037_forward_selection.csv", "exp037_final_ablation.csv",
        "exp037_correlations.csv", "exp037_regional.csv", "exp037_pairwise.csv",
        "exp037_blends.csv", "experiment_log.csv",
    ):
        assert filename in source
    model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
    params = model.get_params()
    assert "LogisticRegression(solver='liblinear',penalty=" not in source
    assert params["penalty"] == inspect.signature(LogisticRegression).parameters["penalty"].default
    assert params["C"] == 1.0
    assert params["class_weight"] is None and params["n_jobs"] is None


def test_exp038_and_finalizer_import_contracts_remain_available() -> None:
    modules = {
        "train_exp036_ratio_ablation": exp036,
        "train_exp037_relational_features": legacy,
    }
    checked = []
    for filename in ("train_exp038_discretization_refinement.py", "finalize_exp037_submission.py"):
        path = ROOT / "src" / filename
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in modules:
                for alias in node.names:
                    assert hasattr(modules[node.module], alias.name)
                    checked.append((filename, alias.name))
    assert checked == [
        ("train_exp038_discretization_refinement.py", "base_rep"),
        ("finalize_exp037_submission.py", "base_rep"),
        ("finalize_exp037_submission.py", "fit_cv"),
    ]


@pytest.mark.parametrize(
    ("filename", "module_name"),
    [
        ("train_exp036_ratio_ablation.py", "exp036_legacy"),
        ("train_exp037_relational_features.py", "exp037_legacy"),
    ],
)
def test_relational_entrypoints_import_from_arbitrary_cwd_without_side_effects(
    tmp_path: Path, filename: str, module_name: str
) -> None:
    existing = [path for path in HISTORICAL_OUTPUTS if path.exists()]
    before = {path: _sha256(path) for path in existing}
    script = ROOT / "src" / filename
    code = (
        "import importlib.util; "
        f"s=importlib.util.spec_from_file_location({module_name!r}, {str(script)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert {path: _sha256(path) for path in existing} == before


def test_exp036_single_relational_variant_sparse_smoke() -> None:
    train, y, _ = _load_sample()
    base = exp036.base_rep(train)
    representation = exp036.add_ratios(
        base, train, [("social_over_screen", 2)]
    )
    probabilities, model = _sparse_fit_predict(representation, y)
    assert model.solver == "liblinear"
    assert probabilities.shape == (600,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_exp037_final_relational_set_sparse_smoke() -> None:
    train, y, _ = _load_sample()
    base = build_base_representation(train)
    representation = migrated.add_features(base, train, [("screen_minus_social", 1)])
    assert representation.columns.tolist() == [
        *base.columns, *FINAL_RATIO_RELATIONS, "screen_minus_social"
    ]
    assert "weekend_over_screen" not in representation
    probabilities, model = _sparse_fit_predict(representation, y)
    assert model.solver == "liblinear"
    assert probabilities.shape == (600,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
