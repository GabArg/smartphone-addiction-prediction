"""Project directories resolved from this file, independent of the working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
MANIFESTS_DIR = OUTPUTS_DIR / "manifests"
DOCS_DIR = PROJECT_ROOT / "docs"
