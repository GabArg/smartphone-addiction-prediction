import importlib

import pytest

MODULES = [
    "catboost_convergence", "nearest_neighbors", "train_test_shift",
    "original_data_analysis", "seed_stability", "exact_value_encoding",
    "realmlp_audit", "negative_region", "model_gating", "threshold_analysis",
    "score_bands", "xgboost_regularization", "xgboost_parameters",
]


@pytest.mark.parametrize("name", MODULES)
def test_moved_diagnostic_imports(name):
    module = importlib.import_module(f"src.diagnostics.{name}")
    assert callable(module.main)
