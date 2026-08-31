"""Compatibility, sparse OHE and smoke checks for migrated EXP-035."""

from __future__ import annotations

import ast
import hashlib
import importlib
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

from src.features.exact_values import ORIGINAL, build_rep, safe_ratio, stringify
from src.models import logistic_exact_values as migrated
from src import train_exp035_exact_value_logistic as legacy


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_OUTPUTS = (
    ROOT / "outputs" / "metrics" / "experiment_log.csv",
    ROOT / "outputs" / "metrics" / "experiment_log_v2.csv",
    ROOT / "outputs" / "metrics" / "exp035_exact_logistic_metrics.txt",
    ROOT / "outputs" / "predictions" / "oof_exp035_exact_logistic.csv",
    ROOT / "outputs" / "predictions" / "test_exp035_exact_logistic.csv",
    ROOT / "outputs" / "submissions" / "submission_exp035_exact_logistic_ensemble.csv",
    ROOT / "outputs" / "reports" / "exp035_variants.csv",
    ROOT / "outputs" / "reports" / "exp035_variants_progress.csv",
    ROOT / "outputs" / "reports" / "exp035_correlations.csv",
    ROOT / "outputs" / "reports" / "exp035_regional.csv",
    ROOT / "outputs" / "reports" / "exp035_pairwise.csv",
    ROOT / "outputs" / "reports" / "exp035_blends.csv",
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


def test_exp035_wrapper_reexports_public_api_and_feature_builders() -> None:
    top_level = importlib.import_module("train_exp035_exact_value_logistic")
    for name in (
        "rss", "rank", "load_oof", "stringify", "safe_ratio", "build_rep",
        "fit_fold", "folds_auc", "full_variant", "nested_choice", "nested_blend",
        "finalize_existing", "main",
    ):
        assert getattr(legacy, name) is getattr(migrated, name)
        assert getattr(top_level, name) is getattr(migrated, name)
    assert migrated.ORIGINAL is ORIGINAL
    assert migrated.stringify is stringify
    assert migrated.safe_ratio is safe_ratio
    assert migrated.build_rep is build_rep


def test_exp035_sparse_ohe_logistic_cv_and_output_contract_are_preserved() -> None:
    source = Path(migrated.__file__).read_text(encoding="utf-8")
    assert "OneHotEncoder(handle_unknown='ignore',dtype=np.float32,sparse_output=True)" in source
    assert "enc.fit_transform(x.iloc[tr])" in source
    assert "enc.transform(x.iloc[va])" in source
    assert "enc.transform(xt)" in source
    assert "sparse.csr_matrix(a,dtype=np.float32)" in source
    assert "LogisticRegression(solver='liblinear',penalty='l2',C=C,max_iter=1000,random_state=42)" in source
    assert "StratifiedKFold(5,shuffle=True,random_state=42)" in source
    for filename in (
        "oof_exp035_exact_logistic.csv", "test_exp035_exact_logistic.csv",
        "submission_exp035_exact_logistic_ensemble.csv",
        "exp035_exact_logistic_metrics.txt", "exp035_variants.csv",
        "exp035_variants_progress.csv", "exp035_correlations.csv",
        "exp035_regional.csv", "exp035_pairwise.csv", "exp035_blends.csv",
        "experiment_log.csv",
    ):
        assert filename in source

    model = LogisticRegression(
        solver="liblinear", penalty="l2", C=1.0, max_iter=1000, random_state=42
    )
    params = model.get_params()
    assert params["solver"] == "liblinear"
    assert params["penalty"] == "l2"
    assert params["C"] == 1.0
    assert params["max_iter"] == 1000
    assert params["random_state"] == 42
    assert params["class_weight"] is None
    assert params["tol"] == 1e-4
    assert params["n_jobs"] is None


def test_exp035_consumers_only_request_reexported_symbols() -> None:
    wrapper = legacy
    checked = []
    for path in (ROOT / "src").rglob("*.py"):
        if path.name == "train_exp035_exact_value_logistic.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "train_exp035_exact_value_logistic":
                for alias in node.names:
                    assert hasattr(wrapper, alias.name)
                    checked.append((path, alias.name))
    assert checked


def test_exp035_import_from_arbitrary_cwd_has_no_side_effects(tmp_path: Path) -> None:
    existing = [path for path in HISTORICAL_OUTPUTS if path.exists()]
    before = {path: _sha256(path) for path in existing}
    script = ROOT / "src" / "train_exp035_exact_value_logistic.py"
    code = (
        "import importlib.util; "
        f"s=importlib.util.spec_from_file_location('exp035_legacy', {str(script)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert {path: _sha256(path) for path in existing} == before


def test_exp035_variant_c_sparse_liblinear_smoke() -> None:
    train, y_series, test = _load_sample()
    x = build_rep(train, "C")
    xt = build_rep(test, "C")
    train_indices, valid_indices = train_test_split(
        np.arange(len(train)), test_size=0.2, random_state=42, stratify=y_series
    )
    encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32, sparse_output=True)
    train_matrix = encoder.fit_transform(x.iloc[train_indices])
    valid_matrix = encoder.transform(x.iloc[valid_indices])
    test_matrix = encoder.transform(xt)
    assert sparse.issparse(train_matrix)
    assert sparse.issparse(valid_matrix)
    assert sparse.issparse(test_matrix)
    assert train_matrix.dtype == valid_matrix.dtype == test_matrix.dtype == np.float32

    model = LogisticRegression(
        solver="liblinear", penalty="l2", C=1.0, max_iter=1000, random_state=42
    )
    model.fit(train_matrix, y_series.iloc[train_indices])
    probabilities = model.predict_proba(valid_matrix)[:, 1]
    assert probabilities.shape == (len(valid_indices),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
