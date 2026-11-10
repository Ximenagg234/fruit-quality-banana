# Checklist de los lineamientos del profesor (`0_Proyecto Final 26-2.pdf`)

**Entrega: 24 de noviembre de 2026.** Grupo de 3 estudiantes.

## Entregables
- [ ] Informe final, **máximo 7 páginas**, formato IEEE (plantilla del profesor o Overleaf IEEE)
- [ ] Repositorio en GitHub con estructura por fases de CRISP-DM (este repo)
- [ ] Enlace del repo + integrantes reportados en el documento compartido por el profesor
- [ ] Diagrama de flujo **personalizado** de CRISP-DM
- [ ] Interfaz gráfica (Streamlit): cargar imagen o usar cámara, mostrar predicción
- [ ] Video de **máximo 10 minutos** (contexto, técnicas, resultados, logros)

## Caso de estudio
- [ ] Entrada: imagen estática de una fruta individual, fondo simple y uniforme
- [ ] Salida 1: clase de calidad (≥ 3 categorías definidas por el grupo)
- [ ] Salida 2: tamaño (pequeño/mediano/grande) o diámetro en píxeles normalizados

## Datos
- [ ] Dataset base: Fruit Quality Classification (Kaggle, `ryandpark`), citado
- [ ] Carpeta `mixed_quality`: segmentar cada fruta individual
- [ ] **30 a 50 imágenes propias** con distintos estados de madurez, tamaños y defectos
- [ ] Anotación manual de calidad y tamaño, coherente con `docs/definicion_clases.md`
- [ ] Mecanismo acordado para compartir anotaciones con toda la clase

## Modelos
- [ ] **≥ 2 modelos de ML tradicional** distintos (LogReg, KNN, SVM, Bagging, árboles, RF, XGBoost)
- [ ] **≥ 1 modelo de deep learning** (CNN pequeña; se recomienda desde cero)
- [ ] Hiperparámetros ajustados por validación cruzada o búsqueda en rejilla
- [ ] Comparación contra una línea base

## CRISP-DM con evidencia en cada fase
- [ ] 1. Comprensión del negocio (impacto económico/social)
- [ ] 2. Comprensión de los datos (distribución de clases, desbalance, calidad de imagen, variabilidad)
- [ ] 3. Preparación (procesamiento, manejo de desbalance)
- [ ] 4. Modelado (entrenamiento, hiperparámetros, selección)
- [ ] 5. Evaluación (métricas en test, análisis de errores, comparación con línea base)
- [ ] 6. Despliegue (GUI)

## Informe: estructura obligatoria
- [ ] Título · [ ] Resumen · [ ] Introducción · [ ] Fundamentos teóricos (sin generalidades)
- [ ] Metodología (no copiar el diagrama CRISP-DM tal cual) · [ ] Resultados
- [ ] Análisis de resultados (generalización, sobreajuste, qué falla, comparación con la literatura)
- [ ] Conclusiones y trabajo futuro · [ ] Referencias IEEE (solo las leídas y usadas)

## Competencias (`1_competencias_consigna_Actualizada.pdf`)
- [ ] Tabla/sección de dilemas éticos ↔ código de ética (PI1)
- [ ] Matriz de impactos social, económico, ambiental, global + mitigaciones (PI2)
- [ ] Formulación, estrategia, fundamentos matemáticos, análisis y conclusiones (PI3)

## Reglas de integridad
- [ ] Todo dato o código de terceros referenciado explícitamente (si no, cuenta como fraude)
- [ ] Gráficos de calidad vectorial
- [ ] Código bien documentado
