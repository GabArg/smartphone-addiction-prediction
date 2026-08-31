# Versiones del entorno de cierre

Inventario obtenido del entorno local el 31 de agosto de 2026. No convierte estas versiones en restricciones obligatorias; documenta el entorno en el que pasaron los tests finales.

| Paquete | Versión instalada | Uso |
|---|---:|---|
| Python | 3.12 | runtime local |
| numpy | 2.5.2 | core |
| pandas | 3.0.5 | core |
| scipy | 1.18.0 | core sparse/estadística |
| scikit-learn | 1.9.0 | core |
| lightgbm | 4.7.0 | core |
| xgboost | 3.4.1 | core |
| catboost | 1.2.10 | core |
| joblib | 1.5.3 | utilidades históricas |
| psutil | 7.2.2 | medición de memoria experimental |
| matplotlib | 3.11.1 | gráficos de documentación |
| pytest | 9.1.1 | tests |
| torch | 2.13.0 | experimental |
| pytabkit | no instalado localmente | RealMLP experimental; Kaggle usó 1.7.3 |

`requirements.txt` se dejó sin pins arbitrarios para instalación general. Este documento conserva las versiones observadas para diagnóstico y reproducción más estricta.

No se creó `pyproject.toml`: el repositorio es un proyecto de competencia con scripts y módulos internos, no un paquete destinado a distribución. En esta etapa `requirements.txt` aporta lo necesario sin introducir una herramienta de packaging adicional.
