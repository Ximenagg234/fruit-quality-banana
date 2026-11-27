# Plan de despliegue

Distingue lo que **ya está hecho** de lo que se **propone** para después. No se publicó la app en internet: el enunciado pide una interfaz gráfica simple que permita cargar una imagen o capturarla con la cámara y mostrar la predicción.

## 1. Alcance y usuarios
- **Para qué:** apoyo a la clasificación de calidad (Buena, Regular, Mala) y medida relativa del tamaño de manzana, banano, guayaba, lima, naranja y granada.
- **Quién:** una persona que selecciona o revisa fruta. La decisión final es de la persona; la app **no certifica inocuidad**.
- **Condiciones:** una fruta por foto, fondo liso y claro, buena luz, preferiblemente sin mano.

## 2. Versión actual (hecha)
| Elemento | Estado |
|---|---|
| Interfaz | Streamlit (`app.py`): subir imagen o usar la cámara; calidad, confianza, probabilidades, aviso de baja confianza, fruta resaltada y longitud normalizada |
| Modelo | CNN propia de tres clases, `modelo/calidad_cnn.keras` (29.5 MB); modelo de respaldo de dos clases, `modelo/calidad.joblib` |
| Dónde corre | computador local, sin tarjeta gráfica |
| Tiempo por foto | 116 ms de mediana (percentil 95: 128 ms) en un Intel i7-1255U (`../05_evaluacion/costo_inferencia.csv`) |
| Privacidad | la imagen se procesa en memoria; no se guarda ni se envía (probado en `../tests/test_inference.py`) |
| Pruebas | `../tests/test_app.py` y `test_inference.py`; prueba manual con cámara en `../docs/prueba_camara.md` |

## 3. Instalación y ejecución
Requisitos: Python 3.11, las dependencias de `../requirements.txt`, un navegador y, para capturar, una cámara.

```powershell
git clone https://github.com/Ximenagg234/fruit-quality-banana.git
cd fruit-quality-banana
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run 06_despliegue/app.py
```

La app queda en `http://localhost:8501`. La cámara del navegador solo funciona en `localhost` o en una página segura (`https`): desde otro equipo por `http` se puede subir la foto, pero no usar el botón de cámara.

## 4. Uso y manejo de errores
1. Elegir "Subir imagen" o "Usar la cámara".
2. Revisar la clase y la confianza.
3. **Confianza menor que 70 %:** repetir la foto con mejor fondo y luz, y revisar la fruta a mano.
4. **Resultado *Regular* o dudoso:** revisar a mano; es la clase menos fiable.
5. **Aviso de que la fruta toca el borde:** repetir sin mano y con la fruta completa; el tamaño no es fiable.

## 5. Límites conocidos (medidos)
- En 69 fotos propias la CNN obtuvo F1 macro de 0.439, **no detectó ninguna *Regular*** (0 de 14) y su exactitud (0.681 ± 0.056) no superó a la línea base (0.710).
- En la prueba con cámara, un banano con defectos que se esperaba *Regular* salió *Buena* con 54 % de confianza y aviso de revisión.
- La longitud normalizada **no mide centímetros**: su correlación con la medida real fue débil (Spearman 0.26).
- Por esto, la versión actual es un **prototipo académico**, no apta para decidir sin revisión humana.

## 6. Plan propuesto (no implementado)
| Etapa | Qué se haría | Criterio para pasar a la siguiente |
|---|---|---|
| A. Datos locales | Fotografiar frutas del lugar de uso, en superficie y sin mano, con más *Regular* y *Mala*; etiquetar por dos personas | una prueba nueva, separada por fruta |
| B. Reentrenamiento | Entrenar incluyendo parte de esas fotos; comparar con la CNN actual | superar a la línea base y al modelo actual en la prueba nueva, con su error estándar |
| C. Piloto | Uso con revisión humana de todos los casos; anotar desacuerdos entre la app y la persona | acuerdo y errores por clase aceptados por quien usa la app |
| D. Publicación (opcional) | Servir la app con `https` (contenedor o servicio de alojamiento) o llevarla a una Raspberry Pi con cámara | revisar privacidad (Ley 1581 de 2012 si hay datos personales), costos y tiempos |
| E. Seguimiento | Registrar errores sin guardar imágenes sin consentimiento; revisar el modelo cuando cambien frutas, cámara o iluminación; poder volver a la versión anterior | revisión periódica |
