from pathlib import Path

from src import project_paths


def test_project_root_is_repository_root():
    expected = Path(__file__).resolve().parents[1]
    assert project_paths.PROJECT_ROOT == expected
    assert (project_paths.PROJECT_ROOT / "src" / "project_paths.py").is_file()


def test_named_directories_are_root_relative():
    for directory in (
        project_paths.DATA_DIR, project_paths.OUTPUTS_DIR, project_paths.METRICS_DIR,
        project_paths.REPORTS_DIR, project_paths.PREDICTIONS_DIR,
        project_paths.SUBMISSIONS_DIR, project_paths.MANIFESTS_DIR, project_paths.DOCS_DIR,
    ):
        assert directory.is_absolute()
        assert project_paths.PROJECT_ROOT in directory.parents
