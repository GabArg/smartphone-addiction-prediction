# Inventario del primer commit

Inventario generado con `git ls-files --others --exclude-standard` antes de `git add`.

## Resumen

- Candidatos observados antes de crear este inventario: 280.
- Archivos previstos en el commit, incluidos los dos informes Git de esta fase: 282.
- Tamaño aproximado final: 2,39 MiB.
- Archivos mayores de 1 MiB: 0.
- Archivo candidato más grande: 240,2 KiB.

## Archivos más grandes

| Path relativo | Tamaño aproximado |
|---|---:|
| `outputs/reports/catboost_convergence.csv` | 240,2 KiB |
| `outputs/reports/threshold_grid_results.csv` | 163,2 KiB |
| `assets/model_comparison.png` | 67,6 KiB |
| `outputs/reports/negative_zone_rule_candidates.csv` | 66,7 KiB |
| `assets/oof_progression.png` | 64,7 KiB |
| `outputs/manifests/reorganization_plan.csv` | 62,1 KiB |
| `outputs/manifests/repository_snapshot.csv` | 59,6 KiB |
| `outputs/reports/exp034_interactions.csv` | 47,9 KiB |
| `outputs/reports/exp036_all_fold_progress.csv` | 43,7 KiB |
| `outputs/reports/exp024_knn_configs.csv` | 28,4 KiB |

## Extensiones principales

| Extensión | Cantidad inicial |
|---|---:|
| `.csv` | 132 |
| `.py` | 89 |
| `.txt` | 43 |
| `.md` | 11 |
| `.png` | 2 |
| `.ipynb` | 1 |
| sin extensión | 1 (`LICENSE`) |

Los dos documentos de esta fase incrementan la cantidad final de Markdown.

## Carpetas principales

| Carpeta | Archivos iniciales | Tamaño aproximado |
|---|---:|---:|
| `outputs/` | 173 | 1,34 MiB |
| `src/` | 71 | 789,4 KiB |
| `tests/` | 18 | 82,1 KiB |
| `docs/` | 10 | 59,5 KiB |
| `assets/` | 2 | 132,3 KiB |
| `notebooks/` | 1 | 5,1 KiB |

## Exclusiones confirmadas

- `data/`: excluido.
- `outputs/predictions/`: excluido.
- `outputs/submissions/`: excluido.
- caches y `__pycache__`: excluidos.
- `.env`, `kaggle.json`, claves y credenciales: excluidos.
- binarios de modelos y temporales: excluidos.
- artifacts históricos con rutas locales: excluidos mediante reglas específicas.

La búsqueda de candidatos prohibidos devolvió cero resultados.
