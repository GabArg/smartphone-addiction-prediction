# Plan de organización y publicación

Este documento define cómo quiero presentar el proyecto sin borrar su historia experimental. Es un plan: todavía no implica mover ni renombrar archivos.

## Estructura propuesta

```text
.
├── README.md
├── LICENSE
├── pyproject.toml
├── configs/
├── docs/
│   ├── bitacora_competencia.md
│   ├── metodologia.md
│   ├── experimentos.md
│   ├── decisiones_y_descartes.md
│   ├── resultados.md
│   └── internal/
│       ├── auditoria_repositorio.md
│       ├── informe_preservacion.md
│       ├── seguridad_publicacion.md
│       └── plan_repositorio.md
├── src/
│   ├── data/
│   ├── models/
│   ├── features/
│   ├── ensembles/
│   ├── diagnostics/
│   └── experiments/
│       └── archive/
├── notebooks/
├── outputs/
│   ├── metrics/
│   ├── reports/
│   │   └── experiments/
│   ├── figures/
│   └── manifests/
├── tests/
└── assets/
```

La carpeta `src/experiments/archive/` no oculta los intentos fallidos: los mantiene consultables sin hacer que el recorrido principal del código parezca una lista de 41 pruebas consecutivas. Cada archivo archivado conservará el identificador EXP en su nombre.

## Código que quedaría en primer plano

| Archivo actual | Nombre propuesto | Papel |
|---|---|---|
| `src/train_logistic_baseline.py` | `src/models/baseline_logistic.py` | baseline lineal — **MIGRATED** (wrapper compatible conservado) |
| `src/train_catboost_exp003.py` | `src/models/catboost_model.py` | referencia CatBoost — **MIGRATED** (wrapper compatible conservado) |
| `src/train_lightgbm_exp004.py` | `src/models/lightgbm_baseline.py` | baseline LightGBM y utilidades categóricas — **MIGRATED** (wrapper compatible conservado) |
| `src/train_xgboost_exp008.py` | `src/models/xgboost_baseline.py` | baseline XGBoost y codificación estable — **MIGRATED** (wrapper compatible conservado) |
| `src/train_xgboost_exp012_threshold_features.py` | `src/features/thresholds.py` | thresholds y regiones reutilizados — **MIGRATED/CENTRALIZED** (entrypoint histórico conservado) |
| `src/train_xgboost_exp016_depth5_9000.py` | `src/models/xgboost_thresholds.py` | XGBoost principal de thresholds — **MIGRATED** (wrapper compatible conservado) |
| `src/train_exp035_exact_value_logistic.py` | `src/models/logistic_exact_values.py` | Logistic sparse de valores exactos — **MIGRATED** (wrapper compatible conservado) |
| lógica exact-value de EXP-035 | `src/features/exact_values.py` | representaciones A/B/C — **CENTRALIZED** |
| `src/train_exp036_ratio_ablation.py` | `src/train_exp036_ratio_ablation.py` | selección relacional — **HISTORICAL / COMPATIBLE** |
| lógica relacional EXP-036/037 | `src/features/relational.py` | seis relaciones ganadoras — **CENTRALIZED** |
| `src/train_exp037_relational_features.py` | `src/models/logistic_relational.py` | extensión con relaciones — **MIGRATED** (wrapper compatible conservado) |
| frecuencia fold-safe de EXP-039 | `src/features/frequency.py` | frequency encoding — **CENTRALIZED** |
| `src/train_exp039_high_bin_lgbm.py` | `src/models/lightgbm_high_resolution.py` | LightGBM de alta resolución — **MIGRATED** (wrapper compatible conservado) |
| `src/finalize_exp039_refined_blend.py` | `src/ensembles/final_ensemble.py` | ensemble principal — **MIGRATED** (wrapper compatible conservado) |
| `src/ensemble_exp009.py` | `src/ensembles/blend_search.py` | utilidades de búsqueda de blends |
| `src/audit_data.py` | `src/data/data_audit.py` | revisión reproducible de datos |

Antes de ejecutar estos movimientos hay que separar las funciones reutilizables de los entrypoints. En particular, el archivo de EXP-012 mezcla definición de features y entrenamiento; no debería moverse mecánicamente a `features/` sin refactorizar imports.

**CORE MIGRATION = COMPLETE.** Los modelos principales, las features compartidas y el ensemble final ya tienen módulos oficiales; los entrypoints históricos permanecen como wrappers compatibles.

## Documentación pública

- `README.md`: entrada breve al proyecto.
- `docs/bitacora_competencia.md`: relato cronológico y técnico.
- `docs/metodologia.md`: validación, prevención de leakage y reproducibilidad.
- `docs/experimentos.md`: catálogo resumido enlazado con `outputs/metrics/experiments.csv`.
- `docs/decisiones_y_descartes.md`: hipótesis que no funcionaron y qué aprendí.
- `docs/resultados.md`: OOF, Public LB y Private LB cuando esté disponible.

## Documentación interna

- `PROJECT_AUDIT.md` → `docs/internal/auditoria_repositorio.md`.
- `PRESERVATION_REPORT.md` → `docs/internal/informe_preservacion.md`.
- `PUBLICATION_SAFETY.md` → `docs/internal/seguridad_publicacion.md`.
- Este plan → `docs/internal/plan_repositorio.md`.

Los tres documentos existentes tienen valor probatorio. Propongo moverlos y darles nombres naturales, no reescribirlos ni reemplazarlos. Sus nombres originales seguirán registrados en los manifiestos.

## Índice del futuro README

1. **Predicting Smartphone Addiction** — una introducción de dos o tres párrafos sobre el problema que resolví.
2. **Datos y métrica** — tamaño, variables, faltantes, ROC AUC y enlace oficial para descargar los datos.
3. **Cómo trabajé** — validación, baselines, ciclos cortos de hipótesis y cuidado con leakage.
4. **Experimentos que cambiaron el proyecto** — Logistic, CatBoost, thresholds en XGBoost, valores exactos y LightGBM high-bin.
5. **Evolución del score** — una tabla o figura con OOF y LB confirmados, sin mezclar ambos.
6. **Modelo final** — componentes de EXP-039 y pesos del blend refinado.
7. **Qué aprendí** — señal de thresholds, valor de las representaciones discretas y límites de modelos alternativos.
8. **Cómo reproducir** — entorno, descarga de datos, comandos y diferencias entre CPU y GPU.
9. **Estructura del repositorio** — mapa corto del código, documentación y artifacts excluidos.
10. **Resultado final** — Public LB ya documentado y Private LB/ranking sólo cuando sean oficiales.
11. **Licencia y datos** — licencia del código, atribuciones y aclaración de que los datos no se incluyen.

El tono será directo y personal. Cuando una decisión fue mía, usaré frases como “probé”, “descarté” o “conservé”. Las métricas y conclusiones técnicas permanecerán impersonales cuando eso facilite la lectura.

## Riesgos antes de ejecutar el plan

- Los scripts se importan entre sí por nombre plano; mover uno puede romper varios entrypoints.
- Hay rutas construidas con `Path(__file__).resolve().parents[1]`; al añadir niveles de carpetas dejarán de apuntar a la raíz.
- Los finalizadores dependen de nombres exactos de OOF y test predictions.
- Algunos reportes y métricas registran paths absolutos.
- El notebook busca `train_exp041_realmlp.py` por nombre; un rename lo rompería.
- La documentación y los manifiestos enlazan nombres actuales.
- Un archivo monolítico puede necesitar dividirse entre `features/`, `models/` y `experiments/`; el CSV propone su papel editorial, no garantiza un movimiento mecánico seguro.

## Orden seguro de ejecución futura

1. Crear pruebas de imports, features y alineación de artifacts.
2. Añadir un módulo común para resolver la raíz y paths.
3. Extraer utilidades compartidas sin cambiar resultados.
4. Mover primero diagnósticos aislados y validar.
5. Migrar modelos principales uno por uno con wrappers temporales.
6. Mantener wrappers con los nombres antiguos durante una transición.
7. Actualizar notebook, documentación y comandos.
8. Comparar hashes de artifacts y ejecutar sólo pruebas ligeras, no entrenamientos completos.
9. Revisar el plan ejecutado antes de inicializar Git.
