# Decisiones y descartes

## Modelos base

La Logistic de EXP-001 quedó como control, no como candidata final. CatBoost, LightGBM y XGBoost mostraron rápidamente que el problema requería no linealidad. XGBoost fue la referencia inicial por su mejor combinación de OOF y estabilidad; CatBoost y LightGBM siguieron siendo útiles por diversidad.

## Thresholds

Los cortes alrededor de 6/8 horas de pantalla y 4 horas de redes sociales produjeron una mejora clara en EXP-012. Conservé indicadores, regiones y distancias porque la ganancia se repitió. Los bloques contextuales de EXP-014 y las reglas negativas anidadas de EXP-018 no mejoraron CV y se descartaron.

## Dataset original y señales externas

La rama de EXP-026 buscó priors y vecinos en el dataset original externo. No mejoró los blends y planteaba además una cuestión de licencia. Cerré esa vía: el dataset externo no forma parte del repositorio y no es necesario para el pipeline final.

## Ensembles, stacking y correcciones

Los blends simples y por ranks mejoraron varias referencias. En cambio, el gating/stacking de EXP-023 y el corrector de rankings de EXP-030 añadieron complejidad sin superar la referencia. Mantuve ensembles transparentes con pesos fijos y cerré los meta-modelos que no justificaron su riesgo.

La extensión de tres a cinco seeds en EXP-028 aportó sólo `0.0000115`; conservé EXP-027 por eficiencia. El ensemble final usa tres ramas suficientemente distintas y no añade componentes con peso residual.

## Representación exact-value

EXP-035 mostró que los valores exactos podían funcionar como tokens categóricos en una matriz sparse. No eliminé las variantes A y B: documentan cómo las interacciones y los ratios de C produjeron el salto. El encoder se ajusta por fold, usa `handle_unknown="ignore"` y no densifica la matriz.

## Relaciones

EXP-036 seleccionó cinco ratios. `weekend_over_screen` y otras candidatas quedaron fuera al no aportar de forma consistente. EXP-037 añadió `screen_minus_social`; diferencias adicionales, inversos y log transforms no pasaron la selección estricta. EXP-038 confirmó que refinar discretizaciones no agregaba valor.

## LightGBM high-bin

El aumento de `max_bin` fue una de las mejoras más claras: 255 quedó muy por debajo de 2047. Entre 2047 y 4095 la diferencia fue prácticamente cero, así que 4095 no debe interpretarse como un nuevo salto, sino como la configuración preservada del modelo final.

Entre las frecuencias, `weekend_freq` y `screen_freq` funcionaron juntas. Añadir `work_freq`, `social_freq` y `gaming_freq` no mejoró el conjunto. Los value codes tampoco entraron en el feature set ganador.

EXP-040 probó copias categóricas y representación dual. Su mejor screen quedó por debajo del baseline comparable; la rama se cerró antes de full CV.

## Redes neuronales y otras alternativas

El MLP de EXP-032 (`0.9383593818`) no alcanzó el criterio mínimo. RealMLP en EXP-041 obtuvo `0.94279075` y `0.94417173` en Fold 1; la auditoría EXP-041B no encontró una configuración crítica ausente. No gasté otra corrida GPU.

ExtraTrees falló primero por memoria y luego rindió `0.9321944743` en la versión segura. Factorization Machine obtuvo `0.9325665537`. KNN/signatures, exact target encoding y correcciones locales tampoco justificaron continuar. Todas estas ramas permanecen registradas porque ayudaron a delimitar dónde estaba —y dónde no estaba— la señal útil.
