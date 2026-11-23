"""Etiquetador manual de frutas (recortes de ``Mixed``) para construir la clase *Regular*.

Muestra un recorte a la vez con su especie y pide una etiqueta: **Buena**, **Regular**, **Mala** o
**Descartar** (no es una fruta o el recorte no sirve). Cada persona etiqueta **de forma
independiente**: la herramienta no muestra etiquetas de otras personas y guarda su propio archivo
(``etiquetas_mixed_<INICIALES>.csv``) dentro de la carpeta de recortes. Se puede cerrar y seguir
después: retoma en el primer recorte pendiente.

Uso (desde la raíz del repositorio, con la carpeta de recortes descomprimida)::

    streamlit run 03_preparacion_datos/etiquetador_mixed.py

Las reglas de cada clase están en ``docs/definicion_clases.md``: **madurez no es defecto** (una fruta
verde sin defectos es Buena); Regular = defectos en ~10 %–40 % de la superficie o un golpe localizado;
Mala = más de ~40 %, moho o aspecto blando. Los umbrales son una propuesta: ante la duda, elige la
clase de la peor zona visible y anótalo.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import labeling as lb  # noqa: E402

CARPETA_DEFECTO = os.environ.get("ETIQUETADOR_CARPETA", str(REPO / "data" / "processed" / "mixed_crops"))
DEFECTOS = {
    "apple": "zonas oscuras hundidas o blandas, podredumbre, golpes extensos",
    "banana": "manchas marrones extensas, zonas negras, cáscara aplastada o blanda, moho",
    "guava": "manchas marrones u oscuras, zonas blandas o hundidas",
    "lemon": "manchas marrones o negras, moho blanco o grisáceo, aspecto seco o arrugado",
    "lime": "manchas marrones o negras, moho blanco o grisáceo, aspecto seco o arrugado",
    "orange": "podredumbre marrón, zonas blandas, moho, depresiones oscuras",
    "pomegranate": "manchas oscuras, grietas con podredumbre, zonas hundidas",
    "foto": "manchas oscuras o marrones extensas, zonas hundidas o blandas, moho, arrugas por deshidratación",
}
#: Ancho (px) de la imagen mostrada: 300 sirve para recortes pequeños; para fotos completas conviene 450.
ANCHO = int(os.environ.get("ETIQUETADOR_ANCHO", "300"))
#: Prefijo del archivo de etiquetas (``etiquetas_mixed`` por defecto; ``etiquetas_fotos`` para las fotos propias).
PREFIJO = os.environ.get("ETIQUETADOR_PREFIJO", "etiquetas_mixed")


def imagen_con_recuadro(carpeta: Path, fila) -> Image.Image | None:
    """Foto de origen con el recuadro del recorte (da contexto al etiquetar), si está disponible."""
    ruta = carpeta / "fuentes" / f"{fila['species']}__{fila['source_image']}"
    if not ruta.is_file():
        return None
    im = Image.open(ruta).convert("RGB")
    ImageDraw.Draw(im).rectangle((fila["x0"], fila["y0"], fila["x1"], fila["y1"]), outline=(255, 0, 255), width=3)
    return im


def main() -> None:
    st.set_page_config(page_title="Etiquetador de frutas", page_icon="🍋", layout="centered")
    st.title("Etiquetador de frutas (Mixed)")
    st.caption("Etiqueta **sin comparar** con otra persona: el acuerdo entre anotadores se mide después.")

    iniciales_txt = st.sidebar.text_input("Tus iniciales", value=os.environ.get("ETIQUETADOR_INICIALES", ""))
    carpeta = Path(st.sidebar.text_input("Carpeta de recortes", value=CARPETA_DEFECTO))
    try:
        ruta = lb.ruta_etiquetas(carpeta, iniciales_txt, PREFIJO)
    except ValueError:
        st.info("Escribe tus iniciales en la barra lateral para empezar.")
        return
    try:
        muestra = lb.cargar_muestra(carpeta)
    except FileNotFoundError:
        st.error(f"No encuentro `muestra_etiquetado.csv` en `{carpeta}`. Descomprime la carpeta de recortes y revisa la ruta.")
        return

    etiquetas = lb.cargar_etiquetas(ruta)
    hechos, total = lb.progreso(muestra, etiquetas)
    clave = f"{carpeta}|{ruta.name}"
    if st.session_state.get("clave") != clave:                      # cambió la persona o la carpeta: retoma donde quedó
        st.session_state["clave"] = clave
        pend = lb.primer_pendiente(muestra, etiquetas)
        st.session_state["pos"] = pend if pend is not None else total - 1
    pos = int(min(max(st.session_state["pos"], 0), total - 1))
    fila = muestra.iloc[pos]

    st.progress(hechos / total, text=f"{hechos} de {total} etiquetados · archivo: {ruta.name}")
    if hechos == total:
        st.success("¡Terminaste! Envía tu archivo de etiquetas (descárgalo abajo) y no lo compartas con la otra persona antes.")
        st.download_button("Descargar mis etiquetas", ruta.read_bytes(), file_name=ruta.name, mime="text/csv")

    esp = fila["species"]
    col1, col2 = st.columns([1, 1])
    crop = carpeta / "recortes" / esp / f"{fila['crop_id']}.jpg"
    if crop.is_file():
        col1.image(str(crop), caption=f"Imagen {int(fila['orden'])} de {total} · {esp}", width=ANCHO)
    else:
        col1.error(f"Falta la imagen {crop.name}")
    ctx = imagen_con_recuadro(carpeta, fila)
    if ctx is not None:
        col2.image(ctx, caption="Foto de origen (recuadro magenta)", use_container_width=True)
    with st.expander(f"Qué cuenta como defecto en {esp}"):
        st.write(DEFECTOS.get(esp, "ver docs/definicion_clases.md"))
        st.write("Madurez no es defecto. Buena: < ~10 % con defecto · Regular: ~10–40 % o un golpe localizado · "
                 "Mala: > ~40 %, moho o aspecto blando.")

    actual = etiquetas.get(fila["crop_id"])
    if actual:
        st.info(f"Tu etiqueta actual para este recorte: **{actual['calidad']}**")
    nota = st.text_input("Notas (opcional)", value=actual["notas"] if actual else "", key=f"nota_{fila['crop_id']}")

    botones = st.columns(4)
    for col, clase in zip(botones, lb.CLASES + [lb.DESCARTAR]):
        if col.button(clase, key=f"b_{clase}_{fila['crop_id']}", use_container_width=True):
            lb.guardar_etiqueta(ruta, fila["crop_id"], clase, nota)
            st.session_state["pos"] = min(pos + 1, total - 1)
            st.rerun()
    nav = st.columns(2)
    if nav[0].button("← Anterior", disabled=pos == 0):
        st.session_state["pos"] = pos - 1
        st.rerun()
    if nav[1].button("Ir al primer pendiente"):
        pend = lb.primer_pendiente(muestra, lb.cargar_etiquetas(ruta))
        st.session_state["pos"] = pend if pend is not None else total - 1
        st.rerun()
    st.caption("**Descartar**: no es una fruta, está cortada o el recorte no se entiende.")


main()
