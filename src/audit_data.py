"""Reproducible data audit for Kaggle Playground Series S6E8."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "data_audit.txt"


class Reporter:
    def __init__(self) -> None:
        self.buffer = StringIO()

    def write(self, value: object = "") -> None:
        text = str(value)
        print(text)
        self.buffer.write(text + "\n")

    def section(self, title: str) -> None:
        self.write("\n" + "=" * 88)
        self.write(title)
        self.write("=" * 88)


def format_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "(sin filas)"
    return frame.to_string(max_rows=max_rows, index=True, float_format=lambda x: f"{x:.4f}")


def missing_table(frame: pd.DataFrame) -> pd.DataFrame:
    count = frame.isna().sum()
    return pd.DataFrame({"missing_count": count, "missing_pct": count / len(frame) * 100})


def identify_target(train: pd.DataFrame, test: pd.DataFrame) -> str:
    candidates = [column for column in train.columns if column not in test.columns]
    if len(candidates) != 1:
        raise ValueError(
            "No se pudo identificar un único target por diferencia de esquemas. "
            f"Columnas sólo en train: {candidates}"
        )
    return candidates[0]


def identify_id_columns(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    common = [column for column in train.columns if column in test.columns]
    named = [c for c in common if c.lower() == "id" or c.lower().endswith("_id")]
    if named:
        return named
    return [c for c in common if train[c].is_unique and test[c].is_unique]


def main() -> None:
    reporter = Reporter()
    paths = {
        "train": DATA_DIR / "train.csv",
        "test": DATA_DIR / "test.csv",
        "sample_submission": DATA_DIR / "sample_submission.csv",
    }
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Faltan archivos requeridos: {missing_files}")

    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])
    sample = pd.read_csv(paths["sample_submission"])
    target = identify_target(train, test)
    features = [column for column in train.columns if column != target]
    id_columns = identify_id_columns(train, test)

    reporter.write("AUDITORÍA DE DATOS - PREDICTING SMARTPHONE ADDICTION (S6E8)")
    reporter.write(f"Raíz del proyecto: {PROJECT_ROOT}")

    reporter.section("1. DIMENSIONES Y COLUMNAS")
    for name, frame in (("train", train), ("test", test), ("sample_submission", sample)):
        reporter.write(f"{name}: {frame.shape[0]:,} filas x {frame.shape[1]} columnas")
        reporter.write(f"Columnas: {frame.columns.tolist()}")

    reporter.section("2. TIPOS DE DATOS")
    for name, frame in (("train", train), ("test", test), ("sample_submission", sample)):
        reporter.write(f"\n{name}:")
        reporter.write(frame.dtypes.astype(str).to_string())

    reporter.section("3. TARGET")
    reporter.write(f"Target identificado automáticamente: {target}")
    target_counts = train[target].value_counts(dropna=False).rename("count").to_frame()
    target_counts["proportion"] = target_counts["count"] / len(train)
    target_counts["percentage"] = target_counts["proportion"] * 100
    reporter.write(format_table(target_counts))

    reporter.section("4. VALORES FALTANTES")
    train_missing = missing_table(train)
    test_missing = missing_table(test)
    reporter.write("Train:")
    reporter.write(format_table(train_missing))
    reporter.write("\nTest:")
    reporter.write(format_table(test_missing))
    missing_comparison = pd.DataFrame(
        {
            "train_missing_pct": train_missing.loc[features, "missing_pct"],
            "test_missing_pct": test_missing.loc[features, "missing_pct"],
        }
    )
    missing_comparison["difference_pp"] = (
        missing_comparison["test_missing_pct"] - missing_comparison["train_missing_pct"]
    )
    reporter.write("\nComparación test - train (puntos porcentuales):")
    reporter.write(format_table(missing_comparison))

    reporter.section("5. DUPLICADOS")
    reporter.write(f"Columnas ID detectadas: {id_columns or 'ninguna'}")
    for id_column in id_columns:
        reporter.write(
            f"{id_column}: train={train[id_column].duplicated().sum():,} IDs duplicados; "
            f"test={test[id_column].duplicated().sum():,} IDs duplicados"
        )
    reporter.write(f"Filas completamente duplicadas en train: {train.duplicated().sum():,}")
    reporter.write(f"Filas completamente duplicadas en test: {test.duplicated().sum():,}")

    categorical_features = [
        c
        for c in features
        if not pd.api.types.is_numeric_dtype(train[c])
        or isinstance(train[c].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(train[c])
    ]
    numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]

    reporter.section("6. COLUMNAS CATEGÓRICAS")
    if not categorical_features:
        reporter.write("No se detectaron columnas categóricas.")
    for column in categorical_features:
        train_values = train[column].nunique(dropna=True)
        test_values = test[column].nunique(dropna=True)
        reporter.write(f"\n{column}: train={train_values:,} únicos; test={test_values:,} únicos")
        counts = train[column].value_counts(dropna=False)
        reporter.write(counts.to_string(max_rows=100))

    reporter.section("7. ESTADÍSTICAS DESCRIPTIVAS NUMÉRICAS")
    reporter.write("Train:")
    reporter.write(format_table(train[numeric_features].describe().T))
    reporter.write("\nTest:")
    reporter.write(format_table(test[numeric_features].describe().T))

    reporter.section("8. CORRELACIÓN NUMÉRICA CON EL TARGET")
    if pd.api.types.is_numeric_dtype(train[target]):
        correlations = train[numeric_features + [target]].corr(numeric_only=True)[target]
        correlations = correlations.drop(target).sort_values(key=lambda s: s.abs(), ascending=False)
        reporter.write(correlations.to_string(float_format=lambda x: f"{x:.6f}"))
    else:
        reporter.write("El target no es numérico; la correlación de Pearson no aplica.")

    reporter.section("9. VALIDACIONES DE ESQUEMA")
    exact_feature_match = features == test.columns.tolist()
    same_feature_set = set(features) == set(test.columns)
    reporter.write(f"Features de train coinciden exactamente y en orden con test: {exact_feature_match}")
    reporter.write(f"Features de train y test contienen el mismo conjunto: {same_feature_set}")
    reporter.write(f"Columnas faltantes en test: {sorted(set(features) - set(test.columns))}")
    reporter.write(f"Columnas extra en test: {sorted(set(test.columns) - set(features))}")
    reporter.write(
        "sample_submission tiene la misma cantidad de filas que test: "
        f"{len(sample) == len(test)} ({len(sample):,} vs {len(test):,})"
    )
    dtype_differences = {
        c: (str(train[c].dtype), str(test[c].dtype))
        for c in features
        if c in test.columns and train[c].dtype != test[c].dtype
    }
    reporter.write(f"Diferencias de dtype train/test: {dtype_differences or 'ninguna'}")

    reporter.section("10. DIFERENCIAS SOSPECHOSAS TRAIN VS TEST")
    anomalies: list[str] = []
    large_missing_shift = missing_comparison[missing_comparison["difference_pp"].abs() > 5]
    for column, row in large_missing_shift.iterrows():
        anomalies.append(f"Missingness en {column}: diferencia de {row['difference_pp']:.2f} pp")

    for column in categorical_features:
        unseen = set(test[column].dropna().unique()) - set(train[column].dropna().unique())
        if unseen:
            preview = list(unseen)[:10]
            anomalies.append(f"{column}: {len(unseen)} categorías sólo en test; muestra={preview}")

    drift_features = [column for column in numeric_features if column not in id_columns]
    numeric_shift_rows = []
    for column in drift_features:
        train_std = train[column].std()
        shift = np.nan if pd.isna(train_std) or train_std == 0 else (
            test[column].mean() - train[column].mean()
        ) / train_std
        numeric_shift_rows.append((column, shift))
        if pd.notna(shift) and abs(shift) > 0.25:
            anomalies.append(f"Media de {column}: desplazamiento estandarizado={shift:.3f}")
    shift_frame = pd.DataFrame(numeric_shift_rows, columns=["column", "standardized_mean_shift"]).set_index("column")
    reporter.write("Desplazamiento de medias (test - train) / std(train):")
    reporter.write(format_table(shift_frame))
    for id_column in id_columns:
        reporter.write(
            f"Rangos de {id_column} (excluido del drift): "
            f"train=[{train[id_column].min()}, {train[id_column].max()}], "
            f"test=[{test[id_column].min()}, {test[id_column].max()}]"
        )

    if dtype_differences:
        anomalies.append(f"Dtypes distintos: {dtype_differences}")
    if not exact_feature_match:
        anomalies.append("El esquema u orden de features no coincide exactamente entre train y test")
    if len(sample) != len(test):
        anomalies.append("La cantidad de filas de sample_submission no coincide con test")

    reporter.write("\nAnomalías marcadas por reglas automáticas:")
    if anomalies:
        for anomaly in anomalies:
            reporter.write(f"- {anomaly}")
    else:
        reporter.write("- No se detectaron anomalías importantes con los umbrales definidos.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(reporter.buffer.getvalue(), encoding="utf-8")
    reporter.write(f"\nReporte guardado en: {REPORT_PATH}")


if __name__ == "__main__":
    main()
