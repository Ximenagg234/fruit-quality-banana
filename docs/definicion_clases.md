# Definición operacional de clases y protocolo de fotografía

> Estado: **propuesta acordada en conversación, umbrales por validar en el EDA.** Todo cambio a esta tabla se debe reflejar en `docs/formulacion_matematica.md` §3.2.

## Clases de calidad

| Clase | Criterio | Fuente de imágenes |
|---|---|---|
| **Buena** | Cáscara amarilla o amarilla-verdosa; < ~10 % de la superficie con manchas marrones; sin golpes ni moho | Dataset (Good) + fotos propias |
| **Regular** | ~10 %–40 % de manchas marrones, o un golpe localizado; sin moho ni zonas blandas visibles | Fotos propias + bananas segmentadas de `mixed_quality` |
| **Mala** | > ~40 % marrón/negro, o moho, o aspecto blando/aplastado | Dataset (Bad) + fotos propias |

- Decisión pendiente: **banana verde** (sin madurar). Propuesta: Buena, documentado en el informe.
- Los umbrales de 10 % y 40 % son una propuesta; se validarán calculando $\rho$ (fracción de mancha) sobre las imágenes etiquetadas.
- La clase **Regular** no existe en el dataset de Kaggle (tiene Good, Bad y Mixed; `Mixed` son varias frutas por foto, no una calidad intermedia): se construye con fotos propias y segmentación.

## Clases de tamaño
Pequeño / Mediano / Grande por terciles de la longitud normalizada $\ell$ (calculados solo con entrenamiento). Ver formulación §3.1.

## Protocolo para las 30–50 fotos propias
1. Comprar 8–10 bananas con distinta maduración (algunas verdes, amarillas y manchadas).
2. Fotografiarlas en varios días (p. ej. días 0, 2, 4, 6) para obtener las tres clases a partir de las mismas bananas.
3. Fondo blanco o liso, buena luz, **misma distancia** (o con objeto de referencia de escala), 2–3 ángulos por banana.
4. Un registro por foto en `data/own_photos/annotations.csv` (ver `data/README.md`), con `banana_id` para poder separar train/test por banana.
5. **Sin personas ni rostros** en las fotos (ver `docs/etica_tabla.md`).
6. Etiquetar cada foto de forma independiente por dos integrantes y medir el acuerdo (kappa de Cohen) antes de resolver los desacuerdos.
