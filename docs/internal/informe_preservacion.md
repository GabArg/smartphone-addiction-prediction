# Preservation Report

Fecha: 2026-08-31

## Estado preservado

- Archivos existentes antes de la fase: **352**.
- Tamaño pre-fase: **1295354617 bytes (1235.35 MiB)**.
- Todos los archivos preexistentes fueron hasheados con SHA-256.
- Archivos originales faltantes después de la fase: **0**.
- Archivos originales cuyo hash cambió: **0**.
- Resultado de integridad: **PASS**.

## Entregables

- Snapshot: `outputs/manifests/repository_snapshot.csv` (352 filas).
- Hashes críticos: `outputs/manifests/critical_artifacts_sha256.txt` (134 artifacts).
- Heavy artifacts: `outputs/manifests/heavy_artifacts.csv` (88 archivos; 1231.73 MiB).
- Categorías heavy: `{"DATA": 3, "PREDICTION": 58, "SUBMISSION": 27}`.
- Rutas absolutas: `outputs/manifests/absolute_paths_report.csv` (11 coincidencias).
- Dependencias: `outputs/manifests/dependency_inventory.csv` (11 paquetes externos detectados).
- Log v2: `outputs/metrics/experiment_log_v2.csv` (43 experimentos/variantes).
- Seguridad: `docs/internal/seguridad_publicacion.md`.
- Protección Git: `.gitignore` creado antes de inicializar Git.

## Artifacts críticos y pesados

Los hashes críticos incluyen scripts EXP, README, auditoría, log original, submissions y predicciones persistidas. Los archivos mayores de 5 MiB permanecen en su ubicación original; no se copiaron ni borraron. Datos, predicciones, submissions y caches están marcados `publish_git=false`.

## Experiment log v2

El log original permanece intacto. V2 contiene EXP-001–EXP-041 más EXP-031B y EXP-041B. Los Public LB de 035/036/037/039 fueron aportados explícitamente en la solicitud de preservación y están marcados `REPORTED`; EXP-041 también está marcado `REPORTED` y sus AUC se describen sólo como Fold 1.

## Inconsistencias encontradas

- El log original tiene 19 entradas y omite numerosos experimentos recuperables.
- No existen artifacts locales de resultados para EXP-041.
- Hay submissions presentes sin prueba local de envío a Kaggle.
- Algunas métricas contienen rutas absolutas locales.
- No hay captura histórica de versiones por experimento ni requirements definitivo.
- README permanece desactualizado; no fue editado.

## Qué no se modificó

- No se movió, renombró ni borró ningún archivo.
- No se modificó ningún script o notebook original.
- No se modificó `README.md`.
- No se modificó `outputs/metrics/experiment_log.csv`.
- No se ejecutó entrenamiento ni se regeneraron predicciones.
- No se creó `.git/`, no se hizo `git init`, add, commit ni push.

## Próximos pasos

Revisar estos manifiestos y luego autorizar una fase separada de reorganización. Antes de mover archivos: confirmar licencias, decidir almacenamiento de artifacts pesados, crear un manifiesto de ubicación externa y aprobar la bitácora v2.
