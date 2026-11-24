"""Interfaz Streamlit: calidad y tamaño de una fruta a partir de una imagen.

Permite **cargar una imagen o capturarla con la cámara** y muestra la calidad
predicha con su confianza y el tamaño (longitud normalizada). La imagen se procesa
en memoria: **no se guarda ni se envía a terceros** (medida E7, ``docs/etica_tabla.md``).

Ejecutar desde la raíz del repositorio::

    streamlit run 06_despliegue/app.py

Streamlit no se vio en clase (el enunciado lo propone como opción de interfaz); se
declara y se cita en el informe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import inference as inf  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "modelo"
#: Se usa la CNN de tres clases si está; si no, el modelo provisional de dos clases.
MODEL_PATHS = [MODEL_DIR / "calidad_cnn.keras", MODEL_DIR / "calidad.joblib"]
ORIGENES = ["Subir imagen", "Usar la cámara"]


@st.cache_resource(show_spinner="Cargando el modelo…")
def cargar_modelo():
    """Carga el modelo una sola vez por sesión del servidor."""
    for ruta in MODEL_PATHS:
        if ruta.is_file():
            return inf.load_model(ruta)
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_DIR}")


def mostrar_calidad(model, im: Image.Image) -> None:
    """Calidad predicha, confianza y probabilidades por clase."""
    r = inf.predict_quality(model, im)
    st.subheader("Calidad")
    c1, c2 = st.columns(2)
    c1.metric("Predicción", r["label"])
    c2.metric("Confianza", f"{r['confidence']:.0%}")
    if r["low_confidence"]:
        st.warning(f"La confianza es menor que {inf.LOW_CONFIDENCE:.0%}: tómala como una sugerencia y "
                   "revisa la fruta a mano. Prueba con un fondo liso y claro, buena luz y una sola fruta.")
    st.bar_chart(r["probabilities"], horizontal=True)
    if "Regular" not in r["probabilities"]:
        st.caption("Este modelo distingue solo **Buena** y **Mala**.")
    elif r["label"] == "Regular" or r["probabilities"]["Regular"] >= 0.25:
        st.caption("**Regular** es la clase menos fiable: en las fotos propias de prueba el modelo no la detectó. "
                   "Tómala como una alerta para revisar la fruta, no como un resultado.")


def mostrar_tamano(im: Image.Image) -> None:
    """Tamaño relativo (longitud normalizada) con la máscara de la fruta."""
    st.subheader("Tamaño")
    medida = inf.measure_size(im)
    if medida is None:
        st.warning("No se detectó la fruta. Usa un fondo liso y claro (blanco o gris), sin personas ni manos.")
        return
    c1, c2 = st.columns([2, 1])
    c1.image(inf.overlay_mask(im, medida["mask"]), caption="Fruta detectada (en magenta)",
             use_container_width=True)
    c2.metric("Longitud normalizada", f"{medida['ell']:.2f}")
    c2.caption("Eje mayor de la fruta dividido por el lado menor de la imagen. Es una medida **relativa**: "
               "depende de la distancia a la cámara.")
    if medida["border_frac"] > inf.HAND_BORDER_FRAC:
        st.info("La fruta detectada toca el borde de la imagen: probablemente hay una mano o un brazo, o la "
                "fruta está cortada, y la medida no es fiable. Fotografía la fruta sola sobre una superficie.")


def main() -> None:
    """Página de Streamlit: imagen o cámara, calidad y tamaño."""
    st.set_page_config(page_title="Calidad y tamaño de frutas", page_icon="🍎", layout="centered")
    st.title("Calidad y tamaño de frutas")
    st.write("Carga una foto o usa la cámara. Una sola fruta, sobre un fondo liso y claro.")
    with st.expander("Alcance y límites"):
        st.markdown(
            "- **Es un apoyo a la decisión, no una decisión final** ni una certificación de inocuidad.\n"
            "- Entrenado con el dataset *Fruit Quality Classification* (Kaggle) y recortes de su carpeta *Mixed* "
            "etiquetados por el equipo; pensado para manzana, banano, guayaba, lima, naranja y granada. "
            "Con fotos propias (mano, otros fondos) su acierto es bajo, sobre todo en *Regular*.\n"
            "- **La imagen no se guarda ni se envía a ningún servicio**: se procesa en memoria.\n"
            "- El tamaño es relativo (depende de la distancia a la cámara)."
        )
    try:
        model = cargar_modelo()
    except FileNotFoundError:
        st.error("No se encontró el modelo entrenado (`06_despliegue/modelo/calidad_cnn.keras`).")
        st.stop()

    # La cámara solo se activa si la persona la elige: un widget de cámara dentro de una pestaña
    # oculta igual pediría permiso de cámara al cargar la página.
    origen = st.radio("¿De dónde viene la imagen?", ORIGENES, horizontal=True)
    if origen == ORIGENES[0]:
        entrada = st.file_uploader("Elige una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])
    else:
        entrada = st.camera_input("Toma la foto de la fruta")
    if entrada is None:
        st.info("Esperando una imagen.")
        return
    try:
        im = ImageOps.exif_transpose(Image.open(entrada)).convert("RGB")
    except Exception:                                       # archivo dañado o no es una imagen
        st.error("No se pudo leer la imagen. Prueba con otro archivo JPG o PNG.")
        return
    st.image(im, caption="Imagen recibida", use_container_width=True)
    mostrar_calidad(model, im)
    mostrar_tamano(im)


main()
