# Reparto del trabajo y forma de colaborar

Entrega: **jueves 26 de noviembre de 2026, en la mañana.** Este documento dice quién hace qué hasta esa fecha, cómo se trabaja en el repositorio y cómo se etiquetan las fotos. El estado real de cada criterio está en [`RUBRICA_TRAZABILIDAD.md`](RUBRICA_TRAZABILIDAD.md); si algo de aquí contradice esa matriz, manda la matriz.

## 1. Equipo

| Integrante | GitHub | Rol |
|---|---|---|
| Ximena Gómez Gómez | @Ximenagg234 | Integra el repositorio (revisa y mezcla los PR); datos, modelos y app |
| Natalia Delgado | @ndelgadoar | Informe y referencias; etiquetado |
| Juan Pablo Bello | _por confirmar_ | Diagrama, video y revisión; etiquetado |

Los tres saben hacer de todo: el reparto es por **carga de trabajo**, no por habilidad, y cualquiera puede cambiar de frente si lo avisa. Cada entregable tiene **un responsable** (lo escribe) y **un revisor** (lo califica con la rúbrica sin haberlo escrito).

## 2. Frentes de trabajo hasta el jueves

| # | Entregable | Criterio de la rúbrica | Responsable | Revisor | Para cuándo |
|---|---|---|---|---|---|
| A | Fotos propias, `object_id` y medidas con regla | datos | Ximena | Natalia | ✅ hecho (69 fotos en `data/own_photos/annotations.csv`) |
| B | Etiquetado independiente de *Mixed* (dos rondas) y de las fotos propias | 1, 6 | Natalia, Juan Pablo y Ximena | Ximena | ✅ hecho (760 recortes; kappa 0.607 y 0.606; fotos propias por consenso) |
| C | Clase *Regular*, CNN, tabla de experimentos y evaluación final (prueba **una sola vez**) | 4, 5, 6 | Ximena | Juan Pablo | ✅ hecho (`04_modelado/`, `ANALISIS_ERRORES.md`) |
| D | App Streamlit con la CNN de tres clases; **probar la cámara en un computador real** | 8 | Ximena (app); **Juan Pablo** (cámara) | Natalia | app ✅; **cámara: miércoles** |
| E | **Informe IEEE** de máximo 7 páginas y `report/refs.bib` (cada referencia verificada) | 3, 4, 5, 6, 7, 8 | **Natalia** | Juan Pablo | borrador: miércoles en la mañana; final: miércoles en la noche |
| F | **Diagrama de flujo CRISP-DM** personalizado (SVG): revisar el que está regenerado y afinarlo | 8 | **Juan Pablo** | Natalia | miércoles |
| G | **Video** de máximo 10 minutos (guion y grabación) | 8 | **Juan Pablo** (con los tres frente a cámara si quieren) | Natalia | guion: miércoles; video: miércoles en la noche |
| H | Ética e impactos: revisar las tablas ya redactadas (v2/v3) y llevarlas al informe | 1, 2 | **Natalia y Juan Pablo** | Ximena | miércoles |
| I | **Auditoría final** criterio por criterio (revisión cruzada) | todos | los tres | — | miércoles en la noche – jueves en la mañana |

Pendientes que **decide Ximena** y no se tocan sin hablar con ella: la fila E8 de `etica_tabla.md` (declaración del uso de herramientas) y la política del curso al respecto.

## 3. Cómo trabajar en el repositorio

Una vez, en PowerShell:

```powershell
git clone https://github.com/Ximenagg234/fruit-quality-banana.git
cd fruit-quality-banana
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
git config user.name "Tu nombre completo"
git config user.email "tu_correo_o_noreply_de_github"
```

Para cada tarea:

```powershell
git checkout main
git pull
git checkout -b informe/borrador          # una rama por tarea: informe/…, video/…, etica/…, datos/…
# ...trabajar...
python -m pytest tests -q                 # deben pasar todas antes de subir
git add <archivos concretos>              # no uses "git add ." a ciegas
git commit -m "Describe qué cambió, en español"
git push -u origin informe/borrador
# abrir un Pull Request hacia main en GitHub y avisar a Ximena para revisarlo y mezclarlo
```

Reglas:
- **Nada va directo a `main`.** Todo cambio entra por PR y lo mezcla Ximena.
- Commits **pequeños, en español y a tu nombre**; así queda visible la participación de cada quien.
- **No se suben** `data/raw/` (3 GB de Kaggle), `data/processed/`, modelos pesados ni las fotos originales. Sí van las fotos propias **reducidas y sin metadatos** (las prepara `src/prepare_photos.py`) y el modelo de la app.
- Los notebooks entregables se suben **ejecutados, con salidas** y sin rutas personales; las figuras del informe, en PDF o SVG.
- Todo **número** del informe sale de código ejecutado sobre los datos reales (con su notebook enlazado). Si no existe todavía, se escribe `[MEDIR]`; no se inventa.
- Todo dato, imagen, código o texto de terceros se **cita**. Una referencia solo entra a `refs.bib` si se comprobó que existe (autores, año, DOI o URL).
- Las técnicas que no se vieron en clase (por ejemplo `StratifiedGroupKFold`, hash perceptual, Streamlit) se **declaran** como tales.

## 4. Dónde está cada cosa

| Qué | Dónde |
|---|---|
| Estado por criterio de la rúbrica y requisitos del enunciado | `docs/RUBRICA_TRAZABILIDAD.md`, `docs/REQUISITOS_LINEAMIENTOS.md` |
| Definición de clases (Buena, Regular, Mala) y protocolo de fotos | `docs/definicion_clases.md` |
| Formulación matemática | `docs/formulacion_matematica.md` |
| Ética e impactos | `docs/etica_tabla.md`, `docs/impactos_matriz.md` |
| EDA, modelos y app | `02_comprension_datos/`, `04_modelado/`, `06_despliegue/` |
| Código compartido y pruebas | `src/`, `tests/` |
| Datos y esquema de anotaciones | `data/README.md` |

Las fotos originales y los archivos pesados se comparten por la **carpeta de Drive del grupo**, no por GitHub.

## 5. Etiquetado (hecho)

**Ya está hecho:** los 760 recortes de *Mixed* se etiquetaron en dos rondas por dos personas cada una, por separado, y los desacuerdos se resolvieron en grupo (`data/mixed_labels/`). Las fotos propias se decidieron por consenso del grupo. Lo que sigue describe el procedimiento, por si hay que repetirlo con más datos.

Se necesitan **dos anotadores** que etiqueten **sin verse entre sí** para poder medir el acuerdo (kappa de Cohen ponderado); los desacuerdos se resuelven entre todos.

1. Leer antes `docs/definicion_clases.md` (regla general y defectos por especie): **madurez no es defecto**, y los umbrales de ~10 % y ~40 % de superficie con defecto son una propuesta.
2. Copiar `docs/plantilla_etiquetado.csv`, llenar una fila por foto y guardarla como `etiquetas_<iniciales>.csv`.
   - `calidad`: `Buena`, `Regular` o `Mala` (obligatoria).
   - `tamano_visual`: `Pequeño`, `Mediano` o `Grande`, **respecto de la misma especie** (opcional; el tamaño final se mide en la imagen y se compara con esta estimación).
   - `notas`: golpe, moho, mano en la foto, duda, etc.
3. **No comentar las etiquetas** con el otro anotador hasta entregar. Entregar el archivo por Drive, no por PR, para no ver el del otro.
4. Cuando haya duda entre dos clases, elegir la que describa mejor la **peor zona visible** y dejarlo anotado en `notas`.
5. Si en una foto aparece una persona o algo que no debería estar (rostro, documento), avisar a Ximena antes de seguir.

## 6. Privacidad de las fotos

No se fotografían rostros (confirmado: ninguna de las fotos propias incluye rostros). Las manos que aparecen son de integrantes del equipo, que aceptan salir en su propio proyecto. Las fotos se suben reducidas y **sin metadatos** (EXIF/GPS); lo garantiza `src/prepare_photos.py`.
