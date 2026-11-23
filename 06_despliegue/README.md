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

**Modelo incluido:** `modelo/calidad.joblib`, un Random Forest entrenado **solo con el conjunto de entrenamiento** de Kaggle para distinguir **Buena** y **Mala** (`../04_modelado/modelos_clasicos.ipynb`). Es **provisional**: se reemplazará al entrenar con la clase *Regular*. Sus métricas en Kaggle no demuestran que haya aprendido la calidad de la fruta (el fondo solo ya predice la clase; ver el EDA y el cuaderno de modelos).

**Límites:** una sola fruta sobre fondo liso y claro; el tamaño es relativo (depende de la distancia a la cámara) y no es fiable si hay una mano en la foto; apoyo a la decisión, no certificación de inocuidad.

**Pruebas:** `../tests/test_app.py` (arranque y entradas), `../tests/test_inference.py` (lógica).
Estado: 🟡 funciona con carga de imagen (verificado en un navegador); la captura con cámara no se pudo verificar en el entorno de desarrollo; faltan la clase *Regular* y las clases de tamaño por especie.
