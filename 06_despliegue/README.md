# Fase 6 — Despliegue (app Streamlit)

**Aplicación:** `app.py`. Se ejecuta desde la raíz del repositorio:

```bash
pip install -r requirements.txt
streamlit run 06_despliegue/app.py
```

**Qué hace:** recibe una imagen —cargada desde un archivo o **capturada con la cámara** (la cámara solo se activa si se elige esa opción)— y muestra:
- la **calidad** predicha, su confianza y las probabilidades por clase, con aviso cuando la confianza es menor que 70 %;
- el **tamaño** como longitud normalizada (eje mayor dividido por el lado menor de la imagen) con la fruta detectada resaltada, y un aviso cuando la fruta toca el borde de la imagen (posible mano, brazo o fruta cortada).

**Privacidad (E7):** la imagen se procesa en memoria; ni la app ni `src/inference.py` la guardan ni la envían a terceros (hay una prueba que lo verifica).

**Modelo incluido:** `modelo/calidad_cnn.keras`, la CNN T6 de tres clases (Buena, Regular, Mala; 29.5 MB, 2 453 059 parámetros) entrenada con Kaggle y los recortes de *Mixed* etiquetados por el equipo (`../04_modelado/entrenar_tres_clases.py`). Se eligió por su F1 macro de validación (0.725). En la prueba con fotos propias acierta *Mala* y *Buena*, pero **no detectó ninguna *Regular*** (0 de 14): la app lo avisa. Sus predicciones coinciden en las 69 fotos propias con las de la evaluación. Si falta ese archivo, la app usa `modelo/calidad.joblib`, el Random Forest provisional de dos clases. Costo de cómputo: 116 ms por foto en CPU (`../05_evaluacion/`).

**Límites:** una sola fruta sobre fondo liso y claro; el tamaño es relativo (depende de la distancia a la cámara) y no es fiable si hay una mano en la foto; apoyo a la decisión, no certificación de inocuidad.

**Pruebas:** `../tests/test_app.py` (arranque y entradas), `../tests/test_inference.py` (lógica).
Estado: 🟡 funciona con carga de imagen y con la CNN de tres clases (pruebas automáticas y coincidencia con la evaluación); la captura con cámara no se pudo verificar en el entorno de desarrollo; faltan las clases de tamaño por especie.
