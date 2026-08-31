# Auditoría pre-publicación

Fecha: 31 de agosto de 2026.

## Alcance

- `files_checked`: 278 archivos potencialmente trackeables según la simulación de `.gitignore`.
- Archivos de texto inspeccionados por rutas y patrones sensibles: 276.
- Se revisaron código, notebooks, documentación, métricas, reportes, manifests, configuración y assets.
- `data/`, predictions, submissions, caches, entornos, credenciales y binarios de modelos quedaron fuera del conjunto publicable.

## Findings

### Critical

Ninguno. No se encontraron credenciales de Kaggle, API keys, tokens, passwords, emails privados ni archivos de claves en el conjunto publicable.

### Warnings

1. Ocho métricas históricas, un reporte de auditoría de datos y el manifiesto de rutas absolutas contienen paths locales. Como son artifacts históricos inmutables, se añadieron exclusiones específicas a `.gitignore` en lugar de reescribirlos.
2. Los Public LB de EXP-035, EXP-036, EXP-037 y EXP-039 están marcados como `REPORTED`; la documentación pública conserva esa distinción.
3. No existe todavía un archivo `LICENSE`. Esto no impide inicializar Git, pero debe resolverse antes de publicar el repositorio si se desea conceder permisos explícitos de reutilización.
4. El dataset original externo no debe publicarse hasta confirmar una licencia explícita. No forma parte del conjunto trackeable.

## Tamaño publicable simulado

- Archivos: 278 antes de añadir este informe; 279 al incluirlo.
- Tamaño aproximado: 2,39 MiB.
- Archivos mayores de 1 MiB: 0.
- Archivos mayores de 5 MiB: 0.
- Archivos mayores de 10 MiB: 0.

Los archivos más grandes son reportes CSV livianos; los dos gráficos PNG están por debajo de 70 KiB cada uno.

## Safe to publish

`safe_to_publish`: **CONDITIONAL**.

El contenido técnico y privado está listo. Antes de un push público conviene elegir una licencia para el código. La competencia y el dataset externo conservan sus propias reglas; los datos no se redistribuyen.

## Estado pre-Git

Es seguro inicializar Git después de revisar visualmente el README y decidir la licencia. Esta auditoría no inicializó repositorios, no creó commits y no contactó servicios externos.
