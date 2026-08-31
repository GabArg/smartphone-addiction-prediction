"""Finalize EXP-037 after the completed screening, without repeating it."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_exp035_exact_value_logistic import load_oof, rank
from train_exp036_ratio_ablation import base_rep
from train_exp037_relational_features import fit_cv


def main():
    data = ROOT / "data"
    pred = ROOT / "outputs" / "predictions"
    subdir = ROOT / "outputs" / "submissions"
    metrics = ROOT / "outputs" / "metrics"
    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")
    y = train.addicted_label.to_numpy()
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(train, y))

    # Refit only the already-selected final configuration to recover each fold's
    # test prediction. The screening and feature selection are not repeated.
    spec = [("screen_minus_social", 1)]
    oof, test_folds, _ = fit_cv(
        train, base_rep(train), y, splits, spec, "FINALIZE", test, base_rep(test)
    )
    saved = load_oof(pred / "oof_exp037_relational_logistic.csv", train)
    if not np.allclose(oof, saved, rtol=0, atol=1e-12):
        raise RuntimeError("Final refit does not reproduce saved EXP-037 OOF")

    base_test = pd.read_csv(subdir / "submission_exp027_seed_ensemble.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    if not base_test.id.equals(sample.id):
        raise RuntimeError("EXP-027 submission IDs do not match sample_submission")
    corrected_test = np.mean(
        [0.625 * rank(base_test.addicted_label.to_numpy(np.float64)) + 0.375 * rank(p)
         for p in test_folds],
        axis=0,
        dtype=np.float64,
    )
    out = pd.DataFrame({"id": sample.id, "addicted_label": corrected_test})
    if len(out) != 296302 or out.isna().any().any():
        raise RuntimeError("Invalid EXP-037 submission shape or NaN")
    if not out.addicted_label.between(0, 1).all():
        raise RuntimeError("EXP-037 predictions outside [0,1]")
    path = subdir / "submission_exp037_relational_logistic_ensemble.csv"
    out.to_csv(path, index=False)

    logpath = metrics / "experiment_log.csv"
    log = pd.read_csv(logpath)
    if not log.experiment_id.astype(str).eq("EXP-037").any():
        row = {
            "experiment_id": "EXP-037",
            "datetime": pd.Timestamp.now().isoformat(timespec="seconds"),
            "model": "RelationalSparseLogistic_Ensemble",
            "features": "EXP036 ratios + screen_minus_social round1",
            "cv_strategy": "Nested_OOF_ensemble_optimization",
            "cv_roc_auc": 0.967068329297671,
            "kaggle_score": "",
            "notes": "rank blend EXP027=0.625 EXP037=0.375; fold-safe selection",
        }
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
        log.to_csv(logpath, index=False)
    print(path)


if __name__ == "__main__":
    main()
