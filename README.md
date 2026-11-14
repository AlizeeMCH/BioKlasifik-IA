# Clasificador de residuos con cámara web
Programa de escritorio para Windows y SmartPhones que usa OpenCV y un modelo YOLOv8 entrenado
para detectar residuos y etiquetarlos como **Basura Orgánica** o **Basura
Inorgánica**. No requiere entrenar un modelo.

## Aplicación web en Vercel

El proyecto también incluye una versión web que usa la cámara del navegador y
combina COCO-SSD para localizar el objeto y MobileNetV2 (alpha 1.0, ImageNet)
para clasificar su recorte. TensorFlow.js usa WebGL cuando está disponible y
CPU como alternativa. El análisis se limita al objeto más cercano al centro y
se estabiliza con las últimas seis predicciones. En Vercel, el sitio sigue
usando únicamente esta inferencia de navegador; el servidor Python es opcional
y no cambia el despliegue estático. En Vercel, el nombre del proyecto debe
escribirse en minúsculas: `bioklasifik-ia`.

Una detección orgánica semántica con evidencia combinada de al menos 15% tiene
prioridad; si MobileNet no reconoce el objeto, un respaldo conservador analiza
los colores naturales del recorte e ignora reflejos claros. El resultado web
solo muestra **Basura Orgánica** o **Basura Inorgánica**.

Para probar la web localmente:

```powershell
npm install
npm run dev
```

Para validar la compilación:

```powershell
npm run build
```

La versión web usa COCO-SSD y MobileNetV2 de
[TensorFlow.js](https://github.com/tensorflow/tfjs-models), cuyos modelos se
entrenaron con COCO e ImageNet respectivamente. Ambos son clasificadores
genéricos, no un modelo de reciclaje; la inferencia puede equivocarse y la
regla de color solo se aplica cuando no hay evidencia semántica clara.

## Modo local con Python y Flask

En `localhost` la página detecta el servidor Flask y puede usar el motor Python
en vez de TensorFlow.js. El backend une el modelo especializado de residuos con
YOLO-COCO y el análisis del 60% central de cada caja; captura y detección
funcionan en hilos separados. El resultado se estabiliza con histéresis de seis
fotogramas (cuatro votos para cambiar, o confianza superior a 60%) y solo
muestra las dos categorías generales.

Requiere Python 3.14 de 64 bits y Node.js/npm para compilar la interfaz local.
El entorno `.venv` del proyecto usa Python 3.14.10. Instala las dependencias y
la interfaz una vez:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
npm install
```

Elige el modo desde la terminal de VS Code:

```powershell
# Ventana clásica de OpenCV
python waste_classifier.py

# Sitio local servido por Flask y procesado por YOLO en Python
python waste_classifier.py --web
# Alternativa equivalente
python app.py
```

La interfaz local se abre en `http://127.0.0.1:5000`. `app.py` compila
`dist/` antes de servir los archivos y expone `/video_feed` (MJPEG) y
`/api/status` (última categoría y porcentaje), además de rutas internas para
iniciar/detener la cámara y actualizar la confianza. Para cambiar el puerto o
los modelos:

```powershell
python waste_classifier.py --web --port 5050 --camera 1
python waste_classifier.py --web --model .\modelo_residuos.pt --object-model .\yolo11n.pt
```

La primera ejecución puede descargar los pesos del detector de residuos desde
Hugging Face y `yolo11n.pt` para COCO; ambos se reutilizan después. Si el primer
modelo no puede descargarse, coloca `yolov8n-waste-12cls-best.pt` junto a
`waste_classifier.py`. El modelo de residuos expone 12 clases; el frontend y el
modo Python las reducen a las dos categorías solicitadas. Fuente y licencia:
[kendrickfff/waste-classification-yolov8-ken](https://huggingface.co/kendrickfff/waste-classification-yolov8-ken),
CC BY 4.0.

La cámara solicita 640x480, activa autofocus cuando está disponible y aplica
CLAHE a la luminancia más un desenfoque gaussiano ligero antes de inferir.
YOLO analiza los fotogramas en segundo plano mientras OpenCV mantiene fluida la
captura y la visualización. La ventana y el stream web comparten exactamente
el mismo motor y el mismo suavizador de detecciones.
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

`biological` y las clases de alimentos/frutas/verduras se muestran como
**Basura Orgánica**; plastic, paper, cardboard, metal, glass, battery, clothes,
shoes, trash y objetos COCO inorgánicos se muestran como **Basura Inorgánica**.
El color verde indica orgánico y azul inorgánico; la confianza es la puntuación
del detector, no una garantía de que el objeto sea basura. Se exige 0.65 de
confianza para `clothes` para reducir falsos positivos provocados por ropa o
mangas de quien sostiene el objeto.

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
