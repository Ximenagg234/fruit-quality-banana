"""Genera docs/diagrama_crispdm.svg y .png: diagrama CRISP-DM personalizado del proyecto (editable).

Uso (desde la raíz del repositorio):  python docs/generar_diagrama_crispdm.py docs
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

salida = Path(sys.argv[1])
plt.rcParams["svg.fonttype"] = "none"          # texto como texto (editable), no como curvas
plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(15, 9.2))
ax.set_xlim(0, 150); ax.set_ylim(0, 92); ax.axis("off")

COL = {"fase": "#E8EEF7", "borde": "#2B4C7E", "dato": "#FFF6E0", "bdato": "#C77D0A", "nota": "#FDECEA", "bnota": "#B23A32",
       "ok": "#E6F4EA", "bok": "#2E7D32"}


def caja(x, y, w, h, titulo, lineas, fc=COL["fase"], ec=COL["borde"], size=7.6):
    """Dibuja un recuadro de una fase con su título y su lista."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 2.2, titulo, ha="center", va="top", fontsize=9.6, fontweight="bold", color=ec)
    ax.text(x + 1.6, y + h - 6.6, "\n".join(lineas), ha="left", va="top", fontsize=size, color="#222", linespacing=1.35)


def flecha(a, b, color="#2B4C7E", estilo="-|>", ls="-", rad=0.0, lw=1.8):
    """Dibuja una flecha entre dos puntos."""
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=estilo, mutation_scale=14, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


ax.text(75, 90.2, "Clasificación de calidad y tamaño de frutas · CRISP-DM aplicado al proyecto", ha="center", fontsize=13.5,
        fontweight="bold", color="#1b2d4f")
ax.text(75, 87.2, "Cada fase lista lo que se hizo, el archivo que lo demuestra y lo que se decidió con datos medidos", ha="center",
        fontsize=8.6, color="#555")

# fila superior: fases 1-3
caja(3, 56, 44, 24, "1 · Comprensión del negocio",
     ["• Desperdicio de alimento; clasificación manual",
      "   lenta y subjetiva",
      "• Salidas: calidad (Buena, Regular, Mala) y tamaño",
      "• Entrada: una fruta sobre fondo liso",
      "• Ética e impactos: docs/etica_tabla.md,",
      "   docs/impactos_matriz.md"])
caja(53, 56, 44, 24, "2 · Comprensión de los datos",
     ["• Kaggle (CC0): 19 526 imágenes, 6 especies",
      "• Regular no existe; Mixed sin etiquetas",
      "• Hallazgo: el borde de la imagen ya predice",
      "   la clase (F1 macro 0.76 a 0.96)",
      "• 02_comprension_datos/eda.ipynb"])
caja(103, 56, 44, 24, "3 · Preparación de los datos",
     ["• Grupos por objeto: tiempo, hash y contador",
      "• Partición por grupo, sin fuga de datos",
      "• Mixed: frutas individuales segmentadas",
      "• Regular: fotos propias + Mixed etiquetado",
      "• Tamaño: longitud normalizada (PCA)"])

# fila inferior: fases 4-6
caja(3, 17, 44, 28, "4 · Modelado",
     ["• Línea base: clase mayoritaria",
      "• ML tradicional: regresión logística, SVM,",
      "   Random Forest (GridSearchCV agrupado)",
      "• CNN desde cero 32-64-128: Flatten o GAP,",
      "   Dropout, aumento de datos, L2",
      "• Clasificador de especie · 04_modelado/"])
caja(53, 17, 44, 28, "5 · Evaluación",
     ["• F1 macro, kappa ponderado, error estándar",
      "• Globales y por especie",
      "• Línea base \"solo borde\" junto a cada modelo",
      "• Fotos propias: cambio de dominio",
      "• Análisis de errores con imágenes",
      "• Tiempo y energía de cómputo"])
caja(103, 17, 44, 28, "6 · Despliegue",
     ["• App Streamlit: subir imagen o usar la cámara",
      "• Calidad, confianza y aviso de baja confianza",
      "• Tamaño con la fruta resaltada",
      "• En memoria: no guarda las imágenes",
      "• Apoyo a la decisión · 06_despliegue/app.py"])

# flujo principal: 1 -> 2 -> 3, codo de 3 a 4 (carril y=50.5) y 4 -> 5 -> 6
flecha((47.6, 68), (52.4, 68)); flecha((97.6, 68), (102.4, 68))
ax.add_patch(FancyArrowPatch((125, 55.4), (125, 50.5), arrowstyle="-", color=COL["borde"], lw=1.8))
ax.add_patch(FancyArrowPatch((125, 50.5), (25, 50.5), arrowstyle="-", color=COL["borde"], lw=1.8))
flecha((25, 50.5), (25, 45.8))
flecha((47.6, 31), (52.4, 31)); flecha((97.6, 31), (102.4, 31))

# bucle de iteración propio del proyecto: evaluación -> modelado
ax.add_patch(FancyArrowPatch((75, 16.4), (75, 11.0), arrowstyle="-", color=COL["bnota"], lw=1.6, ls="--"))
ax.add_patch(FancyArrowPatch((75, 11.0), (25, 11.0), arrowstyle="-", color=COL["bnota"], lw=1.6, ls="--"))
flecha((25, 11.0), (25, 16.4), color=COL["bnota"], ls="--")
ax.text(50, 8.2, "Si no supera la línea base \"solo borde\" o falla en fotos propias: volver a modelado y a los datos",
        ha="center", fontsize=7.8, color=COL["bnota"], style="italic")

# datos propios y evidencia
ax.add_patch(FancyBboxPatch((3, 0.6), 56, 4.6, boxstyle="round,pad=0.3,rounding_size=0.8", fc=COL["dato"], ec=COL["bdato"], lw=1.4))
ax.text(5, 2.9, "Datos propios: fotos de 5 especies; Mixed con dos anotadores", fontsize=7.8, va="center", color="#6b4500")
ax.add_patch(FancyBboxPatch((63, 0.6), 84, 4.6, boxstyle="round,pad=0.3,rounding_size=0.8", fc=COL["ok"], ec=COL["bok"], lw=1.4))
ax.text(65, 2.9, "Control transversal: trazabilidad de la rúbrica, pruebas automáticas y revisión cruzada",
        fontsize=7.8, va="center", color="#1b5e20")

fig.savefig(salida / "diagrama_crispdm.svg", bbox_inches="tight")
fig.savefig(salida / "diagrama_crispdm.png", dpi=120, bbox_inches="tight")
plt.rcParams["pdf.fonttype"] = 42
fig.savefig(salida / "diagrama_crispdm.pdf", bbox_inches="tight")
print("diagrama guardado en", salida)
