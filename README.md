# Predicción de adicción al smartphone

Proyecto desarrollado para **Kaggle Playground Series S6E8**.

## Resumen

El problema consiste en predecir `addicted_label`, una variable binaria, a partir de hábitos de uso del smartphone y variables personales. La métrica de la competencia es **ROC AUC**.

El conjunto de entrenamiento tiene 691.369 filas y el de test 296.302, con 12 variables predictoras. Trabajé con validación cruzada estratificada, guardé predicciones out-of-fold (OOF) para comparar modelos en las mismas particiones y mantuve una bitácora de los experimentos que funcionaron y de los que descarté.

El resultado final fue:

- **OOF ROC AUC:** `0.967468`
- **Public LB:** `0.96876` (resultado reportado desde Kaggle)

## Qué hice

Comencé con una regresión logística sencilla y comparé CatBoost, LightGBM y XGBoost. XGBoost mejoró al incorporar umbrales explícitos sobre tiempo de pantalla y uso de redes sociales. Más adelante probé una representación distinta: convertir valores numéricos exactos en categorías y entrenar una Logistic sparse. Esa rama añadió diversidad y terminó siendo una parte relevante del ensemble.

Después incorporé relaciones entre variables, especialmente cocientes de horas y la diferencia entre pantalla y redes sociales. La última mejora individual importante vino de LightGBM: aumentar `max_bin`, añadir frequency encoding fold-safe y conservar las relaciones como variables numéricas continuas.

No todas las ramas avanzaron. ExtraTrees, redes neuronales, Factorization Machine, target encoding, correctores de ranking y la representación dual de EXP-040 quedaron documentados porque también explican cómo llegué a la solución final.

## Hallazgos técnicos

- Las features de thresholds mejoraron XGBoost de `0.964358` a `0.965233` OOF.
- Tratar valores numéricos exactos como categorías sparse funcionó mejor de lo esperado y aportó una señal complementaria.
- Cinco ratios y `screen_minus_social` ayudaron a la Logistic relacional; `weekend_over_screen` fue descartada en la selección.
- En LightGBM, subir `max_bin` de 255 a 2047 produjo una mejora clara. El cambio de 2047 a 4095 fue prácticamente neutro en el screening, pero se conservó 4095 en la configuración final.
- `weekend_freq` y `screen_freq`, calculadas sólo con el fold de entrenamiento, aportaron señal. Añadir todas las frecuencias no mejoró el resultado.
- Las copias categóricas y la representación dual de EXP-040 no superaron el baseline comparable.
- La diversidad entre XGBoost, Logistic sparse y LightGBM permitió mejorar mediante rank blending.

## Resultados principales

| Experimento | Modelo | OOF ROC AUC | Public LB | Comentario |
|---|---|---:|---:|---|
| EXP-001 | Logistic baseline | 0.911452 | 0.91355 | Punto de partida reproducible |
| EXP-003 | CatBoost | 0.963593 | 0.96497 | Primer boosting competitivo |
| EXP-008 | XGBoost | 0.964358 | 0.96587 | Mejor modelo individual de esa etapa |
| EXP-012 | XGBoost + thresholds | 0.965233 | 0.96696 | Salto importante por features |
| EXP-016 | XGBoost refinado | 0.965702 | 0.96730 | Referencia fuerte de XGBoost |
| EXP-027 | Ensemble de seeds XGBoost + CatBoost | 0.965919 | — | Reducción de varianza |
| EXP-035 | Logistic exact-values + blend | 0.966810 | 0.96815* | Nueva representación sparse |
| EXP-036 | Logistic con ratios + blend | 0.967037 | 0.96838* | Mejora consistente |
| EXP-037 | Logistic relacional + blend | 0.967068 | 0.96842* | Añade `screen_minus_social` |
| EXP-039 | Ensemble final | **0.967468** | **0.96876*** | Mejor resultado del proyecto |

\* Public LB registrado como resultado reportado; no está respaldado por un artifact descargado de Kaggle dentro del repositorio.

## Ensemble final

El resultado final es un rank blend:

- EXP-027, ensemble XGBoost: **37,5 %**
- EXP-037, Logistic relacional: **22,5 %**
- EXP-039, LightGBM high-resolution: **40 %**

Cada componente se transforma a ranking con empates promedio, se normaliza al intervalo `[0, 1]` y luego se aplica el promedio ponderado. La implementación validada está en [`src/ensembles/final_ensemble.py`](src/ensembles/final_ensemble.py).

## Estructura del repositorio

```text
src/
  models/         # modelos core migrados
  features/       # transformaciones compartidas
  ensembles/      # ensemble final
  diagnostics/    # análisis que no entrenan el pipeline principal
  experiments/    # espacio para preservar experimentos históricos
tests/             # imports, contratos, smoke tests y paridad
docs/              # bitácora, metodología, decisiones y resultados
outputs/metrics/   # métricas livianas
outputs/reports/   # tablas y reportes reproducibles
outputs/manifests/ # inventarios, hashes y trazabilidad
assets/            # gráficos para la documentación
```

Los scripts con nombres `EXP` se conservan como entrypoints históricos o wrappers. Esto mantiene la trazabilidad sin duplicar la lógica core.

## Reproducibilidad

1. Crear un entorno con una versión reciente de Python compatible con las dependencias.
2. Instalar el pipeline principal:

   ```bash
   pip install -r requirements.txt
   ```

3. Descargar los datos desde la página oficial de la competencia y ubicar `train.csv`, `test.csv` y `sample_submission.csv` en `data/`.
4. Ejecutar las comprobaciones:

   ```bash
   python -m compileall -q src
   pytest tests -q
   ```

Los datos de competencia, OOF completos, test predictions y submissions no se incluyen. Los experimentos neuronales y RealMLP tienen dependencias separadas en `requirements-experiments.txt`. El repositorio preserva los entrypoints históricos, pero no afirma que los 43 experimentos puedan reproducirse con un único comando.

## Documentación

- [Bitácora de la competencia](docs/bitacora_competencia.md)
- [Metodología](docs/metodologia.md)
- [Decisiones y descartes](docs/decisiones_y_descartes.md)
- [Resultados](docs/resultados.md)

## Datos y licencia

Competition data is not included in this repository. Download it directly from the official Kaggle competition page.

El código y la documentación necesitan una licencia explícita antes de publicar el repositorio. Esa decisión se mantiene separada de la licencia de los datos.
