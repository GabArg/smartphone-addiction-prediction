# Project Audit — Predicting Smartphone Addiction

Fecha de auditoría: 2026-08-31  
Alcance: inventario read-only del estado previo al cierre. La creación de este documento es el único cambio realizado. No se movieron, borraron, renombraron ni ejecutaron experimentos.

## 1. Resumen ejecutivo

El proyecto conserva una trayectoria experimental amplia y valiosa: baselines lineales, CatBoost, LightGBM, XGBoost, feature engineering, ensembles, diagnósticos negativos, modelos de diversidad y una auditoría final de RealMLP. La evidencia permite reconstruir EXP-001 a EXP-041, más las revisiones EXP-031B y EXP-041B.

El mejor resultado OOF registrado es el blend refinado de EXP-039: **0.9674680608**. El mejor Public LB registrado en `experiment_log.csv` es **0.96733** para EXP-021. No debe inferirse que EXP-021 fue la submission final ni que los experimentos posteriores sin LB fueron enviados.

El repositorio todavía no está listo para publicación:

- no existe `.git/` ni `.gitignore` en la carpeta auditada;
- ocupa 1,235.32 MiB, principalmente por CSV de datos y predicciones;
- contiene datasets de Kaggle y el dataset original;
- no tiene manifiesto de dependencias;
- el README está desactualizado, declara que aún no hay modelos entrenados y presenta mojibake de codificación;
- `experiment_log.csv` sólo registra 19 experimentos y omite buena parte de la segunda mitad;
- varios reportes contienen rutas locales de Windows;
- hay 42 archivos `.pyc` bajo `src/__pycache__/`;
- no se detectaron secretos, credenciales ni claves por nombre o patrones de texto, pero esto no sustituye una revisión manual antes de publicar.

## 2. Inventario físico

### 2.1 Totales

| Elemento | Cantidad / tamaño |
|---|---:|
| Archivos | 351 |
| Directorios | 11 |
| Tamaño total | 1,295,329,069 bytes / 1,235.32 MiB |
| Python `.py` | 53 / 0.75 MiB |
| Notebooks `.ipynb` | 1 |
| CSV | 205 / 1,233.17 MiB |
| Texto `.txt` | 49 / 0.37 MiB |
| Bytecode `.pyc` | 42 / 1.02 MiB |
| Markdown | 1 previo + este informe |
| Imágenes | 0 |
| Modelos binarios | 0 detectados |

### 2.2 Estructura y peso actual

| Directorio | Archivos | MiB | Contenido |
|---|---:|---:|---|
| `data/` | 4 | 68.51 | train, test, sample submission y dataset original |
| `notebooks/` | 1 | <0.01 | ejecutor Kaggle de EXP-041 |
| `outputs/` | 250 | 1,165.04 | métricas, OOF, test predictions, submissions y reports |
| `src/` | 95 | 1.77 | 53 fuentes y 42 `.pyc` |
| `playground-series-s6e8/` | 0 | 0 | directorio vacío |

Desglose de CSV en `outputs/`:

| Tipo | Cantidad |
|---|---:|
| métricas/log | 4 |
| reports | 115 |
| predictions OOF/test | 55 |
| submissions | 27 |

### 2.3 Archivos grandes

Los mayores archivos son:

| MiB | Archivo |
|---:|---|
| 54.11 | `outputs/predictions/oof_exp026_original_prior_blend.csv` |
| 46.14 | `outputs/predictions/oof_exp023_nested_gating_diagnostic.csv` |
| 42.78 | `data/train.csv` |
| 39.36 | `outputs/predictions/oof_exp024_best_nested_blend.csv` |
| 33.91 | `outputs/predictions/oof_exp030_rank_corrector.csv` |
| 20.51 | `outputs/predictions/oof_exp033_best_te_nested.csv` |
| 20.48 | `outputs/predictions/oof_exp023_stack_logreg.csv` |
| 19.30 | `outputs/predictions/oof_exp025_adversarial_train.csv` |
| ~19.0 cada uno | numerosos OOF completos de 691,369 filas |
| 17.81 | `data/test.csv` |
| ~7.7 cada uno | test predictions y submissions |

GitHub normal rechaza archivos individuales mayores de 100 MiB, pero el problema práctico aquí es el tamaño agregado: versionar 1.2 GiB de CSV haría el repositorio lento y poco profesional. Los artefactos pesados deberían archivarse fuera de Git o en una release/dataset con checksums y manifiesto.

## 3. Catálogo de experimentos

Estados usados:

- **SUCCESS**: resultado conservado, submission generada o avance claro.
- **MARGINAL**: ejecutado correctamente con ganancia pequeña/empate o sin evidencia suficiente para promoción.
- **DIAGNOSTIC**: diseñado principalmente para aprender/descartar una hipótesis, sin submission.
- **REJECTED**: ejecución válida que empeoró o no superó el gate.
- **FAILED**: no completó la ejecución prevista.

Los estados son una clasificación editorial basada en evidencia; no reemplazan los mensajes originales. `—` significa dato no registrado.

| EXP | Modelo / objetivo / features | CV/OOF recuperado | Public LB | Estado | Evidencia principal |
|---|---|---:|---:|---|---|
| 001 | Logistic Regression baseline; originales, imputación/escalado/OHE | 0.911452 | 0.91355 | SUCCESS | script, metrics, OOF, submission, log |
| 002 | CatBoost baseline; originales, missing nativo | 0.958956 | 0.95996 | SUCCESS | script, metrics, OOF, submission, log |
| 003 | CatBoost 4,000 iteraciones | 0.963593 | 0.96497 | SUCCESS | script, metrics, OOF, submission, log |
| 004 | LightGBM baseline | 0.963537 | 0.96515 | SUCCESS | script, metrics, OOF/test, submission, log |
| 005 | Blend CatBoost/LightGBM | 0.964077 | 0.96536 | SUCCESS | script, metrics, submission, log |
| 006 | LightGBM + 8 features conductuales | 0.963665 | 0.96538 | SUCCESS | script, metrics, OOF/test, submission, log |
| 007 | Ensemble EXP-003/004/006 | 0.964257 | 0.96555 | SUCCESS | script, metrics, submission, log |
| 008 | XGBoost baseline, categorías ordinales | 0.964358 | 0.96587 | SUCCESS | script, metrics, OOF/test, submission, log |
| 009 | Ensemble con XGBoost | 0.964742 | 0.96601 | SUCCESS | script, metrics, submission, log |
| 010 | XGBoost + 8 features EXP-006 | 0.964376 | — | MARGINAL | script, metrics, OOF/test, submission presente, log |
| 011 | Ensemble incorporando EXP-010 | 0.964830 | 0.96611 | SUCCESS | script, metrics, submission, log |
| 012 | XGBoost + thresholds/regiones | 0.965233 | 0.96696 | SUCCESS | script, metrics, OOF/test, submission, log |
| 013 | Ensemble centrado en EXP-012 | 0.965413 | 0.96688 | MARGINAL | script, metrics, submission, log; LB menor que 012 |
| 014 | EXP-012 + contexto ambiguo | 0.965003 | — | REJECTED | script, metrics, OOF/test, submission presente, log; empeora |
| 015 | XGBoost EXP-012 con depth 5 | 0.965605 | 0.96710 | SUCCESS | script, metrics, OOF/test, submission, log |
| 016 | EXP-015 con 9,000 árboles | 0.965702 | 0.96730 | SUCCESS | script, metrics, OOF/test, submission, report, log |
| 017 | Ensemble centrado en EXP-016 | 0.965781 | 0.96725 | MARGINAL | script, metrics, submission, log |
| 018 | Reglas nested de zona negativa | 0.965694 | — | REJECTED | script, metrics, OOF/test, submission presente, report, log |
| 019 | LightGBM con thresholds EXP-012/016 | 0.964513 | — | REJECTED | script, metrics, OOF/test, submission presente, log |
| 020 | CatBoost con thresholds | 0.964878 | — | REJECTED | script, metrics, OOF/test, submission presente, log |
| 021 | Blend EXP-016/020 | 0.965860 | 0.96733 | SUCCESS | script, metrics, submission, log |
| 022 | CatBoost thresholds, 9,000 iteraciones | 0.964911 | — | MARGINAL | script, metrics, OOF/test, submission presente, log |
| 023 | Gating y logistic stacking nested | 0.965851 mejor stack; 0.965702 gating | — | DIAGNOSTIC | 2 scripts, 2 metrics, reports, 2 OOF; sin submission |
| 024 | KNN y rounded signatures | 0.935963 standalone; 0.965691 blend | — | REJECTED | script, diagnostic, reports, OOF; rama descartada |
| 025 | Adversarial train/test validation | 0.565542 adversarial | — | DIAGNOSTIC | script, diagnostic, reports, OOF; no justificó weighting |
| 026 | Forensics contra dataset original/prior | prior blends empeoran | — | DIAGNOSTIC | script, diagnostic, reports, OOF; sin señal residual |
| 027 | Seed diversity XGBoost + CatBoost | 0.965919 | — | SUCCESS | script, metrics, OOF seeds, reports, submission, log |
| 028 | Extensión a cinco seeds | 0.965931 | — | MARGINAL | script, metrics, OOF/test seeds, reports; submission=false |
| 029 | Diagnóstico de errores pairwise/ranking | señal secundaria detectada | — | DIAGNOSTIC | script, metrics y reports; no predicción final |
| 030 | LightGBM rank corrector cross-fitted | 0.966052 aprox.; delta vs 027 negativa | — | REJECTED | script, metrics, OOF y reports; submission=false |
| 031 | ExtraTrees inicial | sin Fold completo | — | FAILED | script y metrics: bloqueado antes de Fold 1 |
| 031B | ExtraTrees memory-safe | 0.932194 | — | REJECTED | script, metrics, OOF/test y reports; diversidad no útil |
| 032 | MLP categórica embebida, viability | Fold 1: 0.938359 | — | REJECTED | script, metrics y reports; full CV no ejecutado |
| 033 | Target encoding exacto/discretizado | 0.965923 blend nested | — | DIAGNOSTIC | script, metrics, OOF y reports; no justificó EXP-034 |
| 034 | Factorization Machine | 0.932567 | — | REJECTED | script, metrics, OOF/test y reports; sin submission |
| 035 | Sparse logistic de valores exactos + blend | 0.963315 individual; 0.966810 ensemble | — | SUCCESS | script/finalizer, metrics, OOF/test, reports, submission, log |
| 036 | Sparse logistic + ratios | 0.963510 individual; 0.967037 ensemble | — | SUCCESS | script, metrics, OOF/test, reports, submission, log |
| 037 | Features relacionales sparse logistic | 0.963575 individual; 0.967068 ensemble | — | SUCCESS | script/finalizer, metrics, OOF/test, reports, submission, log |
| 038 | Refinamiento por discretización | 0.963575; delta 0 | — | REJECTED | script, metrics, OOF/test y reports; sin submission |
| 039 | High-bin LightGBM + blend refinado | 0.966684 individual; **0.967468** ensemble | — | SUCCESS | script/finalizer, metrics, OOF/test, reports, submission, log |
| 040 | LightGBM dual numeric/categorical | 3-fold 0.966195 mejor candidato; delta -0.000317 | — | REJECTED | script, metrics y reports; gate cerró full CV |
| 041 | RealMLP-TD GPU, Fold 1 A/B | A 0.942791; B 0.944172 | — | REJECTED | script + notebook; resultados informados fuera del repo, sin artifacts locales |
| 041B | Auditoría read-only de RealMLP | no entrena | — | DIAGNOSTIC | script; concluye cerrar RealMLP |

### 3.1 Discontinuidades y faltantes

- `experiment_log.csv` contiene 19 filas: 001–022 con huecos sólo posteriores a 022, más 027, 035, 036, 037 y 039.
- No están registrados formalmente 023–026, 028–034, 038, 040, 041 ni 041B.
- EXP-041 no tiene métricas, OOF ni reportes descargados en el proyecto. Sus dos AUC sólo constan en el contexto operativo de cierre y deben marcarse como “reported, artifact missing”.
- EXP-041B no tiene output persistido; su script imprime el diagnóstico.
- EXP-001/002 conservan OOF bajo `outputs/metrics/`, mientras el resto usa `outputs/predictions/`.
- EXP-003 tiene una copia OOF en metrics y otra normalizada en predictions.
- Varias submissions existen aunque el experimento fue rechazado o no tenga LB; existencia no equivale a submission enviada.
- Los LB sólo se consideran conocidos cuando figuran en `experiment_log.csv`. No se inventaron LB faltantes.
- No hay un registro de Private LB, ranking final, nombre de equipo, fecha de submissions ni selección final documentada.

## 4. Mapa exhaustivo hacia una estructura profesional

Estructura recomendada, todavía **no ejecutada**:

```text
.
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml                  # o requirements.lock + requirements.txt
├── configs/
├── docs/
│   ├── PROJECT_AUDIT.md
│   ├── COMPETITION_REPORT.md
│   ├── EXPERIMENT_LOG.md
│   ├── REPRODUCIBILITY.md
│   └── PORTFOLIO_NOTES.md
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── experiments/
│   ├── diagnostics/
│   └── ensembles/
├── notebooks/
├── outputs/
│   ├── metrics/
│   ├── reports/
│   ├── figures/
│   └── manifests/
├── assets/
└── tests/
```

### 4.1 Mapeo por archivo y patrón

Este mapeo cubre todos los archivos actuales. Los patrones son mutuamente claros; no se propone ejecutar nada aún.

| Archivo(s) actual(es) | Destino propuesto | Tratamiento |
|---|---|---|
| `README.md` | `README.md` | reescribir conservando historia; corregir UTF-8 |
| `docs/PROJECT_AUDIT.md` | igual | conservar |
| `data/train.csv`, `data/test.csv`, `data/sample_submission.csv` | fuera de Git; ruta local `data/raw/` documentada | ignorar en Git |
| `data/original/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv` | fuera de Git; `data/external/` local | ignorar; documentar fuente/licencia |
| `playground-series-s6e8/` | no requiere destino | directorio vacío; revisar manualmente antes de retirar |
| `notebooks/exp041_realmlp_kaggle.ipynb` | `notebooks/experiments/exp041_realmlp_kaggle.ipynb` | conservar como ejecutor GPU |
| `src/audit_data.py` | `src/data/audit_data.py` | código de datos |
| `src/train_logistic_baseline.py`, `train_catboost_baseline.py`, `train_catboost_exp003.py`, `train_lightgbm_exp004.py`, `train_lightgbm_exp006_features.py`, `train_xgboost_exp008.py`, `train_xgboost_exp010_features.py`, `train_xgboost_exp012_threshold_features.py`, `train_xgboost_exp014_ambiguous_features.py`, `train_xgboost_exp015_depth5.py`, `train_xgboost_exp016_depth5_9000.py`, `train_xgboost_exp018_nested_negative_rules.py`, `train_lightgbm_exp019_thresholds.py`, `train_catboost_exp020_thresholds.py`, `train_catboost_exp022_thresholds_9000.py`, `train_exp028_seed_extension.py`, `train_exp030_rank_corrector.py`, `train_exp031_extratrees.py`, `train_exp031b_extratrees_memory_safe.py`, `train_exp032_neural_tabular.py`, `train_exp034_factorization_machine.py`, `train_exp035_exact_value_logistic.py`, `train_exp036_ratio_ablation.py`, `train_exp037_relational_features.py`, `train_exp038_discretization_refinement.py`, `train_exp039_high_bin_lgbm.py`, `train_exp040_dual_repr_lgbm.py`, `train_exp041_realmlp.py` | `src/experiments/` | preservar nombres EXP y resultados negativos |
| `src/create_blend_exp005.py`, `ensemble_exp007.py`, `ensemble_exp009.py`, `ensemble_exp011.py`, `ensemble_exp013.py`, `ensemble_exp017.py`, `ensemble_exp021.py`, `stack_exp023_logreg.py`, `finalize_exp037_submission.py`, `finalize_exp039_refined_blend.py` | `src/ensembles/` | blends, stacking y finalización |
| `src/diagnose_catboost_convergence.py`, `diagnose_exp024_nearest_neighbors.py`, `diagnose_exp025_adversarial_validation.py`, `diagnose_exp026_original_forensics.py`, `diagnose_exp027_seed_diversity.py`, `diagnose_exp029_pairwise_ranking.py`, `diagnose_exp033_exact_value_te.py`, `diagnose_exp041b_realmlp.py`, `diagnose_negative_zone.py`, `diagnose_nested_gating_exp023.py`, `diagnose_thresholds.py`, `diagnose_weak_bands.py`, `diagnose_xgboost_depth5_regularization.py`, `diagnose_xgboost_hyperparams.py` | `src/diagnostics/` | preservar diagnósticos positivos y negativos |
| `src/__pycache__/*.pyc` (42) | ningún destino versionado | conservar por ahora; luego ignorar y retirar sólo con aprobación |
| `outputs/metrics/experiment_log.csv` | `outputs/metrics/experiment_log.csv` | ampliar/corregir sin alterar registros originales; idealmente crear v2 |
| `outputs/metrics/exp*.txt` y diagnósticos `.txt` | `outputs/reports/experiments/` | reportes textuales livianos, aptos para Git tras sanitizar rutas |
| `outputs/metrics/exp001_logistic_oof.csv`, `exp002_catboost_oof.csv`, `exp003_catboost_oof.csv` | archivo externo de artifacts; manifiesto en `outputs/manifests/` | OOF pesados, no Git normal |
| `outputs/reports/*.csv` (115) | `outputs/reports/tables/` | versionar sólo tablas pequeñas; excluir las grandes si aparecen |
| `outputs/predictions/oof_*.csv` (34) | almacenamiento externo `artifacts/predictions/oof/` | no Git; checksum + metadata |
| `outputs/predictions/test_*.csv` (21) | almacenamiento externo `artifacts/predictions/test/` | no Git; checksum + metadata |
| `outputs/submissions/*.csv` (27) | almacenamiento externo `artifacts/submissions/`; conservar manifest resumido | no Git normal; puede conservarse una submission de ejemplo pequeña |

Los futuros módulos compartidos deberían extraerse desde los scripts, sin modificar resultados:

- `src/features/`: thresholds/regiones, ratios, exact-value encodings y categorías estables.
- `src/models/`: builders comunes de XGBoost, LightGBM, CatBoost, sparse logistic y RealMLP.
- `configs/`: parámetros exactos por experimento, seeds, paths relativos y gates.
- `tests/`: alineación de IDs, ausencia de leakage, determinismo de features y validación de submissions.

## 5. Git safety y privacidad

### 5.1 Estado Git

- No existe `.git/` en la carpeta auditada; no se pudo obtener branch, commits ni estado.
- No existe `.gitignore`.
- No existe configuración de Git LFS.
- Antes de `git init`, debe crearse y revisarse `.gitignore`; hacerlo después arriesga añadir datasets y artifacts por accidente.

### 5.2 Debe ignorarse

Propuesta inicial, no aplicada:

```gitignore
# Competition and external data
data/
playground-series-s6e8/

# Heavy/generated artifacts
outputs/predictions/
outputs/submissions/
outputs/models/
outputs/**/*.zip
*.pkl
*.pickle
*.joblib
*.pt
*.pth
*.ckpt
*.onnx

# Python/runtime
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/

# Notebook/runtime
.ipynb_checkpoints/

# Credentials and local configuration
.env
.env.*
!.env.example
kaggle.json
*.pem
*.key

# OS/editor/temp
.DS_Store
Thumbs.db
*.tmp
*.bak
*.log
```

No conviene ignorar indiscriminadamente `outputs/metrics/` ni `outputs/reports/`: contienen la evidencia histórica principal. Se recomienda seleccionar métricas/reportes livianos y sanitizados para Git.

### 5.3 Secretos e información privada

- No se encontraron archivos `.env`, `kaggle.json`, credenciales, PEM/keys ni patrones comunes de API tokens.
- No se detectaron imágenes ni metadata visual privada.
- Sí aparecen rutas absolutas con el usuario local `Admin` en métricas 027, 035, 036 y 037, entre otras. Deben sanitizarse en copias publicables, preservando los originales fuera de Git.
- Los CSV incluyen IDs de competencia y predicciones; no parecen datos personales, pero están sujetos a las reglas de distribución de Kaggle.
- El dataset original de 7,500 filas requiere comprobar fuente, licencia y permiso de redistribución antes de publicar.
- Deben revisarse manualmente notebook outputs/metadata y todo historial Git futuro antes de publicación.

## 6. Reproducibilidad

### 6.1 Dependencias inferidas

Dependencias directas observadas:

- Python
- NumPy
- pandas
- SciPy
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- PyTorch
- psutil
- pytabkit 1.7.3 para EXP-041/041B

Versiones del entorno auditado, que no necesariamente coinciden con las usadas en todos los EXP:

| Paquete | Versión local |
|---|---:|
| Python | 3.12.10 |
| NumPy | 2.5.2 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| SciPy | 1.18.0 |
| XGBoost | 3.4.1 |
| LightGBM | 4.7.0 |
| CatBoost | 1.2.10 |
| PyTorch | 2.13.0 CPU |
| psutil | 7.2.2 |
| pytabkit | no instalado localmente; notebook fija 1.7.3 |

Riesgo: no hay `requirements.txt`, `pyproject.toml`, lockfile ni captura contemporánea de versiones por EXP. Algunas versiones locales son posteriores a las corridas y no deben declararse como versiones históricas confirmadas.

### 6.2 Seeds y CV

- La convención dominante es `StratifiedKFold(5, shuffle=True, random_state=42)`.
- Seed base dominante: 42.
- EXP-027 incorpora 42, 2026 y 777.
- EXP-028 agrega 31415 y 1234.
- EXP-041 usa seed 42 y deriva seeds por Fold; EXP-041B no entrena.
- Hay nested selection en 018, 023, 024, 030, 033 y blends posteriores; debe documentarse para evitar confundir OOF simple con OOF nested.

### 6.3 Esquema de datos requerido

- Identificador: `id`.
- Target: `addicted_label` sólo en train.
- 12 features originales: `age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, `weekend_screen_time`, `gender`, `stress_level`, `academic_work_impact`.
- Train auditado: 691,369 filas; test: 296,302 según reportes.
- Los scripts esperan normalmente `data/train.csv`, `data/test.csv` y `data/sample_submission.csv` desde la raíz.
- EXP-026 requiere además `data/original/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`.
- Muchos ensembles requieren OOF/test intermedios exactos; no son reproducibles desde un único comando limpio todavía.

### 6.4 Entrypoints y capacidad de reproducción

- Auditoría de datos: `python src/audit_data.py`.
- Cada `train_*.py`, `ensemble_*.py`, `diagnose_*.py` y `finalize_*.py` actúa como entrypoint independiente.
- EXP-001–031B y 033–040 son conceptualmente reproducibles localmente con datos y artifacts requeridos; el costo va de segundos a horas.
- EXP-032 corrió sólo un Fold CPU y se cerró por viabilidad; GPU sería recomendable para retomarlo, aunque no corresponde hacerlo.
- EXP-041 requiere Kaggle/GPU y `pytabkit==1.7.3`; el notebook es el ejecutor documentado.
- EXP-041B es read-only y puede ejecutarse localmente; la introspección completa de API requiere instalar pytabkit 1.7.3.
- No existe un orquestador global, CLI común ni Makefile. Los imports entre scripts dependen de ejecutar con `src` en `sys.path`.
- La reproducibilidad exacta no puede garantizarse hasta fijar versiones, documentar hardware y verificar hashes de datos/artifacts.

## 7. Riesgos antes de publicar

Prioridad crítica:

1. Publicar accidentalmente los datasets o 1.1 GiB de artifacts.
2. Redistribuir archivos de Kaggle o el dataset original sin verificar reglas/licencia.
3. Inicializar Git antes de tener `.gitignore` y un staging auditado.
4. Presentar como reproducibles versiones históricas que no fueron registradas.

Prioridad alta:

1. README falso/desactualizado y con codificación dañada.
2. Bitácora incompleta después de EXP-022.
3. Rutas absolutas locales en reports.
4. Confusión entre OOF individual, blend nested y Public LB.
5. Submissions existentes sin evidencia de haber sido enviadas.

Prioridad media:

1. Código plano en `src/` con imports cruzados frágiles.
2. Bytecode y directorio vacío.
3. Falta de licencia, tests, CI y guía de reproducción.
4. Ausencia de figuras para portfolio/README.

## 8. Plan de limpieza seguro

No ejecutar hasta aprobación explícita:

1. Crear backup o snapshot inmutable del estado actual y un manifiesto SHA-256 de los 351 archivos.
2. Verificar reglas de Kaggle y licencia/procedencia del dataset original.
3. Crear `.gitignore` antes de inicializar Git.
4. Separar artifacts pesados a almacenamiento externo sin borrar los originales.
5. Crear manifiestos de artifacts con EXP, tipo, filas, columnas, tamaño, hash y ubicación.
6. Completar un `experiment_log_v2.csv` preservando `experiment_log.csv` intacto.
7. Sanitizar rutas absolutas únicamente en copias destinadas a publicación.
8. Reorganizar código mediante movimientos trazables, sin cambiar lógica.
9. Añadir configuración reproducible y pruebas mínimas.
10. Reescribir README e informe; luego revisar `git status` y staging archivo por archivo.
11. Sólo después inicializar/usar Git, hacer un commit local revisable y pedir aprobación separada antes de cualquier push.

## 9. Plan del README

El README profesional debería contener:

1. Resumen de la competencia, métrica y restricciones.
2. Resultado final confirmado: CV, Public/Private LB y ranking sólo si existe evidencia.
3. Evolución visual desde Logistic 0.911 hasta blend EXP-039 0.967468 OOF.
4. Metodología de CV y prevención de leakage.
5. Familias de modelos y feature engineering.
6. Tabla breve de experimentos decisivos; enlace a bitácora completa.
7. Hallazgos negativos útiles: KNN, ExtraTrees, neural CPU, FM, discretización, dual representation y RealMLP.
8. Arquitectura final del ensemble.
9. Estructura del repositorio y comandos de reproducción.
10. Instrucciones para obtener los datos, sin redistribuirlos.
11. Dependencias/hardware.
12. Limitaciones y próximos pasos.
13. Licencia del código y atribuciones.

## 10. Plan del informe completo

`docs/COMPETITION_REPORT.md` debería organizarse como una narrativa técnica:

1. Contexto, dataset y métrica.
2. Auditoría inicial y missingness.
3. Baselines 001–004.
4. Primera etapa de ensembles 005–011.
5. Descubrimiento de thresholds y XGBoost 012–022.
6. Diagnósticos de distribución, vecinos, stacking y dataset original 023–026.
7. Diversidad de seeds y corrección de ranking 027–030.
8. Búsqueda de diversidad de modelo 031–034.
9. Exact-value sparse models y relaciones 035–038.
10. High-bin LightGBM y ensemble final 039.
11. Cierres negativos 040–041B.
12. Qué funcionó, qué no funcionó y por qué.
13. Validación vs leaderboard, riesgos de selección y limitaciones.
14. Resultado final confirmado y lecciones transferibles.

Anexos recomendados:

- catálogo exhaustivo de EXP;
- parámetros y seeds;
- manifiesto de artifacts;
- curva OOF/LB cronológica;
- matriz de dependencias entre experimentos;
- checklist de reproducibilidad.

## 11. Próxima decisión recomendada

El siguiente paso no debería ser mover archivos. Primero conviene aprobar una fase de **preservación y seguridad**:

1. snapshot + hashes;
2. verificación de licencias/reglas de datos;
3. `.gitignore`;
4. manifiesto de artifacts pesados;
5. bitácora v2 con las omisiones identificadas.

Después de revisar esos cinco entregables, se puede autorizar la reorganización física y la redacción del README/informe sin riesgo de perder la historia experimental.
