# Metodología

## Métrica y particiones

La competencia usa ROC AUC. Esta métrica ordena positivos frente a negativos y no depende de elegir un threshold de clasificación. En este proyecto fue especialmente útil porque permitió comparar tanto probabilidades como blends basados en rankings.

La referencia principal fue `StratifiedKFold` con cinco folds, `shuffle=True` y seed 42. Mantener las mismas particiones hizo comparables las predicciones OOF de modelos diferentes. Algunos screens costosos usaron tres folds o sólo Fold 1; en la bitácora están marcados como screens y nunca se presentan como OOF completo.

## Predicciones OOF

Cada fila de entrenamiento recibió una predicción de un modelo que no había visto esa fila durante el fit. Al concatenar los cinco folds obtuve una predicción OOF completa, sobre la que calculé ROC AUC y analicé correlaciones, regiones y errores.

Estas OOF fueron también el contrato entre ramas. Permitieron evaluar ensembles sin reentrenar todos los componentes y evitaron comparar modelos sobre particiones distintas.

## Prevención de leakage

Las transformaciones aprendidas se ajustaron dentro de cada fold. Esto incluye OneHotEncoder, mappings de categorías y frequency encoding. En EXP-039, por ejemplo, las frecuencias de `weekend_screen_time` y `daily_screen_time_hours` se calcularon sólo con el outer-train; valores no vistos se mapearon según el contrato histórico.

Las features algebraicas —ratios, diferencias y thresholds— no usan el target y pueden aplicarse de forma determinista. El exact target encoding de EXP-033 se auditó con evaluación apropiada y no se adoptó porque no aportó señal residual suficiente.

## Evaluación nested

Cuando una decisión dependía de las mismas OOF que después se querían combinar, usé selección nested: los pesos o variantes se escogían sin mirar el fold evaluado. Esto fue importante en las ramas de Logistic exact-values y relaciones. Evitó reportar como generalización una mejora obtenida al optimizar directamente sobre todas las etiquetas OOF.

## Rank blending

Los modelos producen probabilidades con escalas y calibraciones distintas. Para el ensemble final transformé cada vector a ranks con empates promedio, normalicé entre 0 y 1 y apliqué pesos fijos. La composición final fue 37,5 % EXP-027, 22,5 % EXP-037 y 40 % EXP-039.

El rank blend aprovecha el orden de cada modelo y reduce la influencia de diferencias de calibración. Los pesos se seleccionaron con OOF; no se volvieron a optimizar contra el leaderboard.

## CV frente a Public LB

CV fue el criterio de decisión. Public LB se usó como comprobación externa, no como función objetivo. Un ejemplo fue EXP-013: mejoró OOF pero no Public LB; se archivó sin iniciar una búsqueda de pesos guiada por el leaderboard.

El Public LB cubre sólo una parte del test y puede favorecer decisiones por azar. Por eso los valores reportados se muestran separados de OOF y con su nivel de evidencia. No hay Private LB inventado ni inferido.

## Adversarial validation

EXP-025 entrenó un clasificador para distinguir train de test. El AUC `0.5655417257` indicó shift moderado, impulsado sobre todo por patrones de missingness. No era suficiente para cambiar la validación ni ponderar folds. El diagnóstico quedó como alerta, no como una licencia para adaptar el modelo al test.

## Preservación de folds y artifacts

Las seeds, el orden de columnas, los mappings y el uso de `best_iteration` forman parte del experimento. Durante la migración se conservaron wrappers legacy y se verificaron features, configuraciones y hashes. Las predicciones completas no se publican, pero sus rutas y contratos permanecen registrados en manifests internos.
