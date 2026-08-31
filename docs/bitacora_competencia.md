# Bitácora de la competencia

Esta bitácora reconstruye el recorrido completo a partir de scripts, métricas y reportes preservados. No es una lista de victorias: incluye pruebas marginales, diagnósticos y ramas cerradas. Los valores vacíos no se completaron por intuición.

## 1. Baselines y primeras referencias — EXP-001 a EXP-007

**EXP-001** estableció el punto de partida con LogisticRegression, imputación, scaling y one-hot encoding. Alcanzó `0.911452` OOF y `0.91355` Public LB. Fue útil como control reproducible, pero dejó claro que la relación entre variables no era suficientemente lineal.

**EXP-002** llevó el problema a CatBoost y produjo un salto a `0.958956` OOF. **EXP-003** aumentó su capacidad hasta 4.000 iteraciones y llegó a `0.963593`. La hipótesis de que las categóricas y las interacciones no lineales requerían boosting quedó respaldada.

**EXP-004** probó LightGBM sobre las variables originales (`0.963537`). **EXP-005** combinó referencias linealmente y mejoró a `0.96407743`. **EXP-006** añadió features a LightGBM con una mejora pequeña (`0.963665`), y **EXP-007** consolidó tres modelos en un blend de probabilidad (`0.96425749`). Estos experimentos confirmaron que la diversidad aportaba, aunque todavía de forma modesta.

## 2. XGBoost, rankings y thresholds — EXP-008 a EXP-018

**EXP-008** introdujo XGBoost con mapping ordinal estable para las categóricas. Obtuvo `0.964358` OOF y se convirtió en la referencia individual. **EXP-009** mostró que combinar por ranks (`0.96474203`) era más robusto que depender sólo de la escala de probabilidades.

**EXP-010** modificó XGBoost, pero quedó casi empatado con EXP-008 (`0.96437584`). **EXP-011** aprovechó esa diversidad en otro rank ensemble (`0.96483039`).

El cambio importante llegó con **EXP-012**: indicadores y distancias alrededor de umbrales de tiempo de pantalla y redes sociales, además de regiones positiva, negativa y ambigua. El OOF subió a `0.965233`. **EXP-013** mejoró OOF mediante ensemble, pero su Public LB quedó por debajo de EXP-012; se archivó para no perseguir una discrepancia del leaderboard.

**EXP-014** añadió un bloque contextual que empeoró CV (`0.96500336`) y se rechazó. **EXP-015** ajustó profundidad y generalización (`0.96560512`). **EXP-016**, con depth 5, techo de 9.000 árboles y early stopping, alcanzó `0.96570217`; quedó como la referencia XGBoost de la rama.

**EXP-017** obtuvo una mejora pequeña mediante ensemble (`0.96578124`), insuficiente para reemplazar con claridad la referencia. **EXP-018** añadió reglas negativas anidadas y empeoró (`0.96569444`), por lo que esas reglas no se incorporaron.

## 3. Modelos complementarios, stacking y datos externos — EXP-019 a EXP-026

**EXP-019** evaluó LightGBM como alternativa (`0.96451349`): no ganó individualmente, pero sirvió para medir diversidad. **EXP-020** hizo lo mismo con CatBoost y thresholds (`0.96487849`). **EXP-021** combinó las ramas mediante ranks y alcanzó `0.9658597975` OOF y `0.96733` Public LB.

**EXP-022** extendió CatBoost hasta 9.000 iteraciones. El cambio individual fue marginal (`0.96491092`), aunque el modelo siguió siendo útil como componente. **EXP-023** probó gating y stacking logístico con evaluación nested; `0.9658514697` no superó EXP-021 y la rama se cerró.

**EXP-024** estudió vecinos y firmas discretas. Su mejor blend (`0.9656907606`) empeoró la referencia. **EXP-025** fue una validación adversarial train/test: AUC `0.5655417257`, una diferencia moderada asociada sobre todo a missingness. No justificó reponderar folds ni perseguir el test.

**EXP-026** investigó el dataset original externo, priors y KNN. Los blends con esos priors empeoraron. Se preservó el análisis, pero no se incorporó el dataset externo ni se asumió una licencia para publicarlo.

## 4. Estabilidad, errores y alternativas — EXP-027 a EXP-034

**EXP-027** entrenó XGBoost con seeds 42, 2026 y 777 y combinó el bag con CatBoost. Alcanzó `0.9659191881` y redujo la dependencia de una sola inicialización. **EXP-028** añadió dos seeds; la mejora fue sólo `+0.0000115`, así que el costo adicional no se justificó.

**EXP-029** analizó errores por pares y encontró estructura secundaria. **EXP-030** intentó convertirla en un corrector LightGBM de rankings, pero terminó en `0.965835`, por debajo de EXP-027/028.

**EXP-031** probó ExtraTrees y falló antes de completar el primer fold por consumo de memoria. La revisión memory-safe, **EXP-031B**, sí terminó, pero obtuvo `0.9321944743`; la diversidad no compensaba su baja calidad.

**EXP-032** evaluó un MLP con embeddings categóricos. Su screen de `0.9383593818` no superó el criterio de calidad/tiempo y evitó un CV completo. **EXP-033** auditó exact target encoding: `0.9659226941` no mostró señal residual directa suficiente y se cerró para evitar complejidad y riesgo de leakage. **EXP-034**, una Factorization Machine, obtuvo `0.9325665537`; tampoco mejoró al mezclarse.

## 5. Valores exactos y relaciones — EXP-035 a EXP-038

**EXP-035** partió de una idea poco habitual: convertir los valores numéricos exactos en categorías y entrenar una LogisticRegression sparse con `liblinear`. Se conservaron tres variantes: A con valores exactos, B con interacciones y C con ratios categóricos. El modelo complementó a EXP-027 y el blend llegó a `0.9668097849` OOF. Su Public LB `0.96815` está registrado como reportado.

**EXP-036** hizo ablations de ratios, leave-one-out, pares, triples, granularidad y forward selection. El conjunto ganador incluyó `social_over_screen`, `gaming_over_screen`, `work_over_screen`, `work_over_social` y `gaming_over_social`; `weekend_over_screen` no quedó en el set final. El blend alcanzó `0.9670368278` y mejoró los cinco folds frente a EXP-035.

**EXP-037** investigó diferencias, ratios invertidos y otras discretizaciones. Sólo `screen_minus_social`, redondeada a un decimal, pasó la selección estricta. El resultado final fue `0.9670683293`. **EXP-038** refinó bins y discretizaciones, pero el mejor resultado individual (`0.9635752939`) no añadió mejora frente al modelo relacional; la rama se cerró.

## 6. LightGBM de alta resolución — EXP-039 y EXP-040

**EXP-039** revisó `max_bin` en 255, 511, 1023, 2047 y 4095. El screen de tres folds pasó de `0.96356591` con 255 a `0.96591427` con 2047; 4095 produjo una diferencia prácticamente nula. Se conservó 4095 para el modelo final.

Después se evaluaron frecuencias, códigos de valor exacto y relaciones numéricas. `weekend_freq` y `screen_freq`, ajustadas sólo sobre el outer-train, mejoraron el screen; el conjunto combinado de frecuencias y seis relaciones llegó a `0.96627831` en esa etapa. El ajuste final usó `num_leaves=15`, `min_child_samples=100`, `learning_rate=0.03`, 10.000 árboles como techo y early stopping.

El ensemble refinado de EXP-039 obtuvo `0.9674680607837304` OOF. **EXP-040** duplicó representaciones numéricas como categorías y probó parámetros categóricos. Su mejor screen (`0.966195`) quedó `0.000317` por debajo del baseline comparable, por lo que no pasó a full CV.

## 7. Redes finales y cierre — EXP-041 / EXP-041B

**EXP-041** ejecutó RealMLP-TD en Kaggle GPU. Sólo se evaluó Fold 1: la variante A obtuvo `0.94279075` y la B `0.94417173`. Esos números no son OOF de cinco folds. Dado el costo aproximado y la distancia frente a las referencias, no se continuó.

**EXP-041B** auditó la API y la configuración de pytabkit sin entrenar. No encontró una omisión crítica capaz de explicar una mejora grande, por lo que RealMLP quedó cerrado.

## Modelo final y estado del cierre

El modelo final combina ranks normalizados de EXP-027 (`0.375`), la Logistic relacional de EXP-037 (`0.225`) y el LightGBM high-resolution de EXP-039 (`0.400`). El Public LB `0.96876` se conserva como valor reportado. No hay Private LB confirmado en los artifacts al momento de cerrar esta documentación.

Los scripts históricos siguen presentes. La lógica reutilizable fue centralizada y validada con tests de imports, configuración, smoke y paridad; las predicciones pesadas permanecen fuera del repositorio público.
