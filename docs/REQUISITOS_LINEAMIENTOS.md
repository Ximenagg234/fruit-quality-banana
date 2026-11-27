# Checklist de los lineamientos del profesor (`0_Proyecto Final 26-2.pdf`)

**Entrega: jueves 26 de noviembre de 2026, en la mañana.** Grupo de 3 estudiantes.

**Alcance (confirmado por el profesor):** el sistema debe cubrir variedad de frutas (posiblemente también verduras), no una sola especie. Especies del dataset base: manzana, banano, guayaba, lima, naranja y granada.

## Entregables
- [x] Informe final, **máximo 7 páginas**, formato IEEE (plantilla del profesor o Overleaf IEEE)
- [x] Repositorio en GitHub con estructura por fases de CRISP-DM (este repo)
- [x] Enlace del repo + integrantes reportados en el documento compartido por el profesor
- [x] Diagrama de flujo **personalizado** de CRISP-DM
- [x] Interfaz gráfica (Streamlit) **funcional**: cargar una imagen **o capturarla con la cámara** (§1.4 y §4.6) y mostrar la predicción de calidad y de tamaño
- [x] Video de **máximo 10 minutos** (contexto, técnicas, resultados, logros)

## Caso de estudio
- [x] Entrada: imagen estática de una fruta individual, fondo simple y uniforme
- [x] Salida 1: clase de calidad (≥ 3 categorías definidas por el grupo)
- [x] Salida 2: tamaño (pequeño/mediano/grande) o diámetro en píxeles normalizados

## Datos
- [x] Dataset base: Fruit Quality Classification (Kaggle, `ryandpark`), citado
- [x] Carpeta `mixed_quality`: segmentar cada fruta individual
- [x] **30 a 50 imágenes propias** con distintos estados de madurez, tamaños y defectos
- [x] Anotación manual de calidad y tamaño, coherente con `docs/definicion_clases.md`
- [x] Mecanismo acordado para compartir anotaciones con toda la clase

## Modelos
- [x] **≥ 2 modelos de ML tradicional** distintos (LogReg, KNN, SVM, Bagging, árboles, RF, XGBoost)
- [x] **≥ 1 modelo de deep learning** (CNN pequeña; se recomienda desde cero)
- [x] Hiperparámetros ajustados por validación cruzada o búsqueda en rejilla
- [x] Comparación contra una línea base

## CRISP-DM con evidencia en cada fase
- [x] 1. Comprensión del negocio (impacto económico/social)
- [x] 2. Comprensión de los datos (distribución de clases, desbalance, calidad de imagen, variabilidad)
- [x] 3. Preparación (procesamiento, manejo de desbalance)
- [x] 4. Modelado (entrenamiento, hiperparámetros, selección)
- [x] 5. Evaluación (métricas en test, análisis de errores, comparación con línea base)
- [x] 6. Despliegue (GUI)

## Informe: estructura obligatoria
- [x] Título · [x] Resumen · [x] Introducción · [x] Fundamentos teóricos (sin generalidades)
- [x] Metodología (no copiar el diagrama CRISP-DM tal cual) · [x] Resultados
- [x] Análisis de resultados (generalización, sobreajuste, qué falla, comparación con la literatura)
- [x] Conclusiones y trabajo futuro · [x] Referencias IEEE (solo las leídas y usadas)

## Competencias (`1_competencias_consigna_Actualizada.pdf`)
- [x] Tabla/sección de dilemas éticos ↔ código de ética (PI1)
- [x] Matriz de impactos social, económico, ambiental, global + mitigaciones (PI2)
- [x] Formulación, estrategia, fundamentos matemáticos, análisis y conclusiones (PI3)

## Reglas de integridad
- [x] Todo dato o código de terceros referenciado explícitamente (si no, cuenta como fraude)
- [x] Gráficos de calidad vectorial
- [x] Código bien documentado
