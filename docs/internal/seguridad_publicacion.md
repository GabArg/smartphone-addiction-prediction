# Publication Safety

Fecha: 2026-08-31

## Resultado

No se detectaron credenciales, API keys, claves privadas ni archivos de secretos mediante nombres y patrones comunes. La publicación todavía requiere controles de datos, licencias, rutas locales y artifacts pesados.

## Datos

- `data/train.csv`, `data/test.csv` y `data/sample_submission.csv` son Competition Data y no deben incluirse en el repositorio público.
- `data/original/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv` es un dataset externo. No publicar una copia hasta confirmar fuente, autor, licencia explícita y condiciones de redistribución.
- El README futuro debe decir exactamente: **“Competition data is not included in this repository. Download it directly from the official Kaggle competition page.”**
- No se copian aquí las reglas completas de Kaggle; antes de publicar hay que revisar la página oficial y las reglas aplicables de la competencia.

## Predictions y submissions

- `outputs/predictions/` contiene predicciones fila a fila y debe permanecer fuera de Git.
- `outputs/submissions/` contiene submissions completas y debe permanecer fuera de Git.
- Publicar sólo métricas agregadas, tablas pequeñas y un manifiesto con tamaños/hashes.

## Rutas locales y privacidad

- Se detectaron 11 coincidencias de rutas absolutas en archivos potencialmente publicables. Ver `outputs/manifests/absolute_paths_report.csv`.
- No se modificaron esos archivos. En una fase posterior deben crearse copias sanitizadas o reemplazarse las rutas por rutas relativas.
- No se observó información personal sensible en el escaneo automático. El nombre local `Admin` aparece como parte de rutas del equipo.
- Realizar revisión humana de notebooks, metadata y reportes antes de staging.

## Credenciales

- No incluir `kaggle.json`, `.env`, tokens, certificados, archivos `.pem` o `.key`.
- `.gitignore` cubre estos patrones, pero no sustituye un secret scan previo a cada publicación.

## Licencias

- Añadir una licencia para el código sólo después de confirmar que todo el código publicable es propio o compatible.
- Registrar atribuciones para Kaggle, el dataset externo y las bibliotecas.
- Verificar las licencias de dependencias directas (NumPy, pandas, scikit-learn, SciPy, LightGBM, XGBoost, CatBoost) y de dependencias experimentales (PyTorch, pytabkit, psutil) antes de hacer afirmaciones de compatibilidad.
- No redistribuir binarios, datasets o pesos bajo la suposición de que la licencia del código también los cubre.

## Gate recomendado antes de repo público

1. Confirmar reglas/licencia de los dos orígenes de datos.
2. Sanitizar rutas absolutas.
3. Revisar el contenido staged archivo por archivo.
4. Ejecutar un secret scan adicional sobre el conjunto exacto a publicar.
5. Publicar sólo código, documentación y métricas/reportes pequeños seleccionados.
