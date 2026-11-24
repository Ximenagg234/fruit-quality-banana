# Definición operacional de clases y protocolo de fotografía

> Estado: **propuesta, umbrales por validar con las fotos propias y los datos medidos en el EDA.** Todo cambio a esta tabla se debe reflejar en `docs/formulacion_matematica.md` §3.2.
>
> Alcance: frutas de varias especies (el dataset base trae manzana, banano, guayaba, lima, naranja y granada). Posibles verduras quedan fuera del núcleo.

## 1. Clases de calidad (regla general)

La calidad se decide por la **fracción de superficie visible con defecto**, $\rho \in [0, 1]$, y por la presencia de defectos graves. Un *defecto* es cualquier zona con alteración visible distinta de la maduración normal de la especie (ver §2).

| Clase | Criterio | Fuente de imágenes |
|---|---|---|
| **Buena** | $\rho <$ ~10 %; sin moho, sin golpes marcados, sin zonas blandas ni podridas | Kaggle *Good* (etiqueta débil) + fotos propias + frutas de *Mixed* etiquetadas |
| **Regular** | ~10 % $\le \rho \le$ ~40 %, o un golpe/defecto localizado; sin moho ni zonas blandas extensas | Fotos propias + frutas de *Mixed* segmentadas y etiquetadas a mano + (si se aprueba) nivel intermedio de otro dataset |
| **Mala** | $\rho >$ ~40 %, o moho, o aspecto blando/podrido | Kaggle *Bad* (etiqueta débil) + fotos propias + frutas de *Mixed* etiquetadas |

**Los umbrales de 10 % y 40 % son una propuesta**, no un resultado. Se validan midiendo $\rho$ sobre las imágenes etiquetadas por el grupo y revisando los casos frontera.

### Reglas que valen para todas las especies
- **Madurez no es defecto.** Una fruta verde sin defectos (lima, naranja, guayaba verdes aparecen en *Good*) es **Buena**; el color por sí solo no baja la calidad. Para el banano verde, la propuesta es Buena (se documenta como decisión).
- **Regular no existe en el dataset de Kaggle** (solo *Good*, *Bad* y *Mixed*; *Mixed* son varias frutas por foto, sin etiqueta por fruta, no una calidad intermedia). Se construye con las fuentes de la tabla.
- **Cada fuente debe aportar las tres clases.** Si *Regular* viniera solo de una fuente (p. ej. las fotos propias), el modelo podría aprender la fuente (fondo, cámara) en lugar de la calidad. Por eso las fotos propias y *Mixed* también llevan Buena y Mala.
- **Las etiquetas de Kaggle son débiles:** *Good* → Buena y *Bad* → Mala se toman como vienen; el criterio de sus autores no está documentado y puede haber ruido. Se declara como limitación.

## 2. Qué cuenta como defecto en cada especie

*Observado a ojo en muestras de las carpetas Bad; es una guía para etiquetar, no una medición.*

| Especie | Defectos a buscar |
|---|---|
| Manzana (apple) | Zonas oscuras hundidas o blandas, podredumbre, golpes extensos |
| Banano (banana) | Manchas marrones extensas, zonas negras, cáscara aplastada o blanda, moho |
| Guayaba (guava) | Manchas marrones u oscuras, zonas blandas o hundidas |
| Lima (lime) | Manchas marrones o negras, moho blanco/grisáceo, aspecto seco o arrugado |
| Naranja (orange) | Podredumbre marrón, zonas blandas, moho, depresiones oscuras |
| Granada (pomegranate) | Manchas oscuras, grietas con podredumbre, zonas hundidas |

## 3. Clases de tamaño

Pequeño / Mediano / Grande por terciles de la longitud normalizada $\ell$, **calculados por especie y solo con entrenamiento** (ver formulación §3.1). La misma fórmula sirve para cualquier forma (en un disco da el diámetro; en una elipse, el eje mayor).

Limitaciones que se declaran:
- Las imágenes de Kaggle **no tienen escala**: el zoom y la distancia cambian el tamaño aparente, así que ahí el tamaño es solo relativo.
- En las fotos propias, la regla o el objeto de referencia permiten medir el tamaño en cm y **validar** $\ell$.
- La mano que sostiene la fruta, o varias frutas en una foto, pueden alterar la medida.

## 4. Protocolo para las fotos propias

Mínimo exigido por los lineamientos: 30 a 50 imágenes. Las fotos propias son las del 23 de noviembre: 69 fotos de 34 frutas de cinco especies del alcance (apple, banana, guava, lime, orange), con `object_id` por fruta y distintos grados de deterioro. Siguen faltando frutas *Mala* y *Regular* de lima, granada y sobre superficie sin mano (ver `docs/RUBRICA_TRAZABILIDAD.md`).

1. **Un objeto por foto**, con un papel con su código antes de cada objeto (especie y número, p. ej. `A03`, `G02`, `L04`, `N01`, `P05`).
2. Fondo liso claro (pared o cartulina), luz natural difusa, sin flash, ~30–40 cm, mismo encuadre.
3. **Regla pegada al objeto**, en el mismo plano; medir con regla el lado más largo y anotarlo en `max_dim_cm`.
4. **Mitad de las fotos en mano y mitad sobre la superficie** en cada calidad, para que la mano no quede asociada a una clase (en Kaggle *Bad* las manos parecen más frecuentes; hipótesis por medir en el EDA).
5. Nombre de archivo `<codigo>_d<dia>_<angulo>.jpg` (p. ej. `A03_d0_top.jpg`).
6. Un registro por foto en `data/own_photos/annotations.csv` (esquema en `data/README.md`), con `object_id` para separar train/test por objeto.
7. **Sin rostros** en las fotos; si aparecen manos de otras personas, pedirles su visto bueno (ver `docs/etica_tabla.md`).
8. Reducir las fotos a ~1024 px y **quitar los metadatos** antes de subirlas al repositorio; los originales quedan fuera.
9. Los recortes de `Mixed` se etiquetan de forma **independiente por dos integrantes** y se mide el acuerdo (kappa de Cohen ponderado) antes de resolver los desacuerdos. La calidad de las fotos propias se decidió por **consenso del grupo**, así que para ellas no hay kappa (limitación que se declara).
