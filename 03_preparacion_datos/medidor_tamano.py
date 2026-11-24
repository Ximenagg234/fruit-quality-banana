"""Registro de la medida real (cm) de cada fruta de las fotos propias.

Muestra una foto a la vez y pide el **lado más largo de la fruta medido con regla, en cm** (columna
``max_dim_cm`` de ``data/own_photos/annotations.csv``). Sirve para validar la longitud normalizada ℓ.

Uso (desde la raíz del repositorio, con la carpeta de fotos descomprimida)::

    $env:MEDIDOR_CARPETA = "C:\\etiquetado\\fotos_para_etiquetar"
    $env:MEDIDOR_INICIALES = "XG"
    streamlit run 03_preparacion_datos/medidor_tamano.py

Guarda ``medidas_fotos_<INICIALES>.csv`` en esa carpeta y retoma donde se quedó. Si una foto muestra varias
frutas, se mide la **central o la más visible** y se anota en *Notas*.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import labeling as lb  # noqa: E402
import medidas as md  # noqa: E402

CARPETA_DEFECTO = os.environ.get("MEDIDOR_CARPETA", os.environ.get("ETIQUETADOR_CARPETA", ""))
ANCHO = int(os.environ.get("MEDIDOR_ANCHO", "500"))


def main() -> None:
    """Interfaz de Streamlit para registrar la medida en cm de cada foto, con guardado automático."""
    st.set_page_config(page_title="Medidas de las fotos propias", page_icon="📏", layout="centered")
    st.title("Medida real de cada fruta (cm)")
    st.caption("Lado **más largo** de la fruta, medido con regla. Si la foto tiene varias frutas, mide la central "
               "y anótalo en *Notas*.")

    iniciales = st.sidebar.text_input("Tus iniciales", value=os.environ.get("MEDIDOR_INICIALES", ""))
    carpeta = Path(st.sidebar.text_input("Carpeta de fotos", value=CARPETA_DEFECTO))
    try:
        ruta = md.ruta_medidas(carpeta, iniciales)
    except ValueError:
        st.info("Escribe tus iniciales en la barra lateral para empezar.")
        return
    try:
        muestra = lb.cargar_muestra(carpeta)
    except FileNotFoundError:
        st.error(f"No encuentro `muestra_etiquetado.csv` en `{carpeta}`. Revisa la ruta.")
        return

    ids = list(muestra["crop_id"])
    total = len(ids)
    medidas = md.cargar_medidas(ruta)
    clave = f"{carpeta}|{ruta.name}"
    if st.session_state.get("clave_medidor") != clave:
        st.session_state["clave_medidor"] = clave
        pend = md.primer_pendiente(ids, medidas)
        st.session_state["pos_medidor"] = pend if pend is not None else total - 1
    pos = int(min(max(st.session_state["pos_medidor"], 0), total - 1))
    fid = ids[pos]

    hechos = sum(1 for i in ids if i in medidas)
    st.progress(hechos / total, text=f"{hechos} de {total} registradas · archivo: {ruta.name}")
    if hechos == total:
        st.success("¡Todas registradas! El archivo ya está guardado.")

    imagen = carpeta / "recortes" / "foto" / f"{fid}.jpg"
    if imagen.is_file():
        st.image(str(imagen), caption=f"Foto {fid} ({pos + 1} de {total})", width=ANCHO)
    else:
        st.error(f"Falta la imagen {imagen.name}")

    actual = medidas.get(fid)
    if actual:
        txt = actual["max_dim_cm"] + " cm" if actual["max_dim_cm"] else "sin medida"
        st.info(f"Registrado para esta foto: **{txt}**")
    anterior = medidas.get(ids[pos - 1]) if pos > 0 else None

    def avanzar() -> None:
        """Pasa a la foto siguiente y vuelve a dibujar la página."""
        st.session_state["pos_medidor"] = min(pos + 1, total - 1)
        st.rerun()

    with st.form(f"form_{fid}", clear_on_submit=False):
        valor = st.text_input("Lado más largo (cm)", value=actual["max_dim_cm"] if actual else "", key=f"v_{fid}",
                              placeholder="por ejemplo 12.5")
        nota = st.text_input("Notas (opcional)", value=actual["notas"] if actual else "", key=f"n_{fid}")
        c1, c2, c3 = st.columns(3)
        guardar = c1.form_submit_button("Guardar y siguiente", type="primary", use_container_width=True)
        igual = c2.form_submit_button("Misma fruta que la anterior", use_container_width=True,
                                      disabled=not (anterior and anterior["max_dim_cm"]))
        no_medida = c3.form_submit_button("No pude medir", use_container_width=True)

    if guardar:
        try:
            md.guardar_medida(ruta, fid, md.parsear_cm(valor), nota)
        except ValueError as e:
            st.error(str(e))
        else:
            avanzar()
    elif igual and anterior:
        md.guardar_medida(ruta, fid, float(anterior["max_dim_cm"]), nota or "misma fruta que la foto anterior")
        avanzar()
    elif no_medida:
        md.guardar_medida(ruta, fid, None, nota or "no medida")
        avanzar()

    nav = st.columns(2)
    if nav[0].button("← Anterior", disabled=pos == 0):
        st.session_state["pos_medidor"] = pos - 1
        st.rerun()
    if nav[1].button("Ir a la primera pendiente"):
        pend = md.primer_pendiente(ids, md.cargar_medidas(ruta))
        st.session_state["pos_medidor"] = pend if pend is not None else total - 1
        st.rerun()


main()
