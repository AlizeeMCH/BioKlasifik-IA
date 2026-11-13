# Clasificador de residuos con cámara web

Programa de escritorio para Windows que usa OpenCV y un modelo YOLOv8 entrenado
para detectar residuos y etiquetarlos como **Basura Orgánica** o **Basura
Inorgánica**. No requiere entrenar un modelo.

## Aplicación web en Vercel

El proyecto también incluye una versión web que usa la cámara del navegador y
ejecuta el modelo localmente con WebGPU. Las imágenes no se envían a un servidor.
La primera vez se descargan aproximadamente 50 MB de modelo; usa Chrome o Edge
actualizados y permite el acceso a la cámara. En Vercel, el nombre del proyecto
debe escribirse en minúsculas: `bioklasifik-ia`.

Para probar la web localmente:

```powershell
npm install
npm run dev
```

Para validar la compilación:

```powershell
npm run build
```

El modelo web se convirtió desde
[kendrickfff/waste-classification-yolov8-ken](https://huggingface.co/kendrickfff/waste-classification-yolov8-ken),
publicado bajo CC BY 4.0. Se conserva la atribución al autor en la aplicación.

## Requisito importante: Python

Usa Python 3.14 de 64 bits. El entorno `.venv` de este proyecto usa Python
3.14.10; se verificaron las importaciones de PyTorch, Torchvision, Ultralytics
y OpenCV en ese entorno.

## Pandas offline

Se instalaron Pandas 3.0.5, NumPy 2.5.3 y sus dependencias desde
`PANDAS_OFFLINE` en `.venv`. Las ruedas de esa carpeta requieren Python 3.14
de 64 bits. Para repetir la instalación sin conexión:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-index --find-links=.\PANDAS_OFFLINE pandas==3.0.5
```

## Instalación y ejecución

Abre PowerShell en esta carpeta y ejecuta:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python waste_classifier.py
```

La primera ejecución descarga de Hugging Face los pesos
`yolov8n-waste-12cls-best.pt` (aprox. 51 MB), que después permanecen en caché.
Si no hay conexión, coloca ese mismo archivo junto a `waste_classifier.py`
para usarlo como respaldo. El modelo expone 12 etiquetas: battery, biological,
brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes,
trash y white-glass. Fuente y licencia del modelo: [kendrickfff/waste-classification-yolov8-ken](https://huggingface.co/kendrickfff/waste-classification-yolov8-ken), CC BY 4.0.

La cámara solicita 640x480, activa autofocus cuando está disponible y aplica
CLAHE a la luminancia más un desenfoque gaussiano ligero antes de inferir.
YOLO procesa cada segundo fotograma y reutiliza las detecciones en el siguiente;
las cajas y confianzas se promedian para reducir parpadeos.
Para seleccionar otra cámara:

```powershell
python waste_classifier.py --camera 1
```

La confianza mínima predeterminada es 0.35. Para reducir falsos positivos,
prueba 0.40; para objetos pequeños puedes probar inferencia a 960, a costa de
velocidad. También se puede elegir resolución e historial:

```powershell
python waste_classifier.py --confidence 0.40 --imgsz 640 --width 640 --height 480 --history 5
```

`--brightness` y `--exposure` aceptan valores específicos del controlador de la
cámara; no tienen una escala universal, así que déjalos sin configurar salvo
que conozcas los valores adecuados. Si autofocus causa problemas, usa
`--no-autofocus`. Presiona `q` o `ESC` para salir.

## Alcance del modelo

`biological` y sus variantes se muestran como **Orgánico**; plastic, paper,
cardboard, metal, glass, battery, clothes, shoes y trash se muestran como
**Basura Inorgánica**. Las variantes brown/green/white-glass conservan el tipo
de vidrio en la etiqueta. El color verde indica orgánico y azul inorgánico; la
confianza es la puntuación del detector, no una garantía de que el objeto sea
basura. Se exige 0.65 de confianza para `clothes` (0.35 para las demás clases)
para reducir falsos positivos provocados por mangas o ropa de quien sostiene
el objeto.

Para usar otros pesos locales: `python waste_classifier.py --model
.\modelo_residuos.pt`. El detector fue entrenado con un conjunto específico y
puede confundir objetos fuera de sus clases o contexto; conviene presentar el
residuo centrado, ocupando buena parte de la imagen y con la mano fuera de la
región del objeto siempre que sea posible.

Para mejores resultados, usa luz frontal uniforme, evita sombras/reflejos y un
fondo sencillo. Presenta un objeto a la vez, enfocado y suficientemente cerca
para que ocupe una parte visible del cuadro; la distancia óptima depende de la
cámara y del tamaño del residuo.

## Cámara no detectada

- En **Configuración > Privacidad y seguridad > Cámara**, activa el acceso a la
  cámara y el acceso para aplicaciones de escritorio.
- Cierra otras aplicaciones que estén usando la cámara.
- Prueba otros índices: `python waste_classifier.py --camera 1` (o `2`).
- El programa solicita la cámara con el controlador DirectShow de Windows.
- Si PowerShell bloquea la activación del entorno virtual, abre una nueva
  terminal o activa el entorno con:
  `& .\.venv\Scripts\python.exe .\waste_classifier.py`

## Prueba de clasificación

Con el entorno virtual activado, ejecuta:

```powershell
python -m unittest -v
```
