"""Detección de residuos con cámara web y un YOLO especializado en residuos."""

import argparse
import sys
import unicodedata
from collections import deque
from pathlib import Path


ID_REPOSITORIO = "kendrickfff/waste-classification-yolov8-ken"
ARCHIVO_MODELO = "yolov8n-waste-12cls-best.pt"
INTERVALO_INFERENCIA = 2
UMBRAL_PIEL_BIOLOGICO = 0.45
AREA_CAJA_MAXIMA = 0.75
ANCHO_VENTANA = 800
ALTO_VENTANA = 600

UMBRALES_POR_CLASE = {
    "clothes": 0.65,
    "green glass": 0.55,
    "brown glass": 0.55,
    "white glass": 0.55,
}

CLASES_RESIDUOS = {
    "biological": ("Basura Orgánica", "Orgánico"),
    "organic": ("Basura Orgánica", "Orgánico"),
    "food": ("Basura Orgánica", "Orgánico"),
    "fruit": ("Basura Orgánica", "Orgánico"),
    "vegetable": ("Basura Orgánica", "Orgánico"),
    "peel": ("Basura Orgánica", "Cáscara"),
    "plastic": ("Basura Inorgánica", "Plástico"),
    "paper": ("Basura Inorgánica", "Papel"),
    "cardboard": ("Basura Inorgánica", "Cartón"),
    "metal": ("Basura Inorgánica", "Metal"),
    "glass": ("Basura Inorgánica", "Vidrio"),
    "brown glass": ("Basura Inorgánica", "Vidrio marrón"),
    "green glass": ("Basura Inorgánica", "Vidrio verde"),
    "white glass": ("Basura Inorgánica", "Vidrio transparente"),
    "battery": ("Basura Inorgánica", "Batería"),
    "clothes": ("Basura Inorgánica", "Residuo Inorgánico"),
    "shoes": ("Basura Inorgánica", "Calzado"),
    "trash": ("Basura Inorgánica", "Residuo mixto"),
}

COLORES = {
    "Basura Orgánica": (0, 200, 0),  # Verde en formato RGB
    "Basura Inorgánica": (0, 120, 255),  # Azul en formato RGB
}


def normalizar_nombre_clase(nombre_clase):
    normalizado = unicodedata.normalize("NFKD", nombre_clase.strip().lower())
    normalizado = "".join(
        caracter for caracter in normalizado if not unicodedata.combining(caracter)
    )
    return " ".join(normalizado.replace("_", " ").replace("-", " ").split())


def clasificar_clase(nombre_clase):
    """Devuelve categoría y etiqueta en español para las clases de residuos."""
    normalizado = normalizar_nombre_clase(nombre_clase)
    if normalizado in CLASES_RESIDUOS:
        return CLASES_RESIDUOS[normalizado]
    if any(
        palabra in normalizado
        for palabra in ("organic", "biological", "food", "fruit", "peel")
    ):
        return "Basura Orgánica", "Orgánico"
    return None


def supera_umbral(nombre_clase, confianza_deteccion, umbral_base):
    """Aplica umbrales específicos a ropa y vidrio."""
    umbral_clase = UMBRALES_POR_CLASE.get(
        normalizar_nombre_clase(nombre_clase), umbral_base
    )
    return confianza_deteccion >= max(umbral_base, umbral_clase)


def porcentaje_piel_en_caja(frame_bgr, caja, cv2):
    """Calcula la fracción de píxeles con crominancia típica de piel en una caja."""
    alto, ancho = frame_bgr.shape[:2]
    x1 = max(0, min(ancho, int(caja[0])))
    y1 = max(0, min(alto, int(caja[1])))
    x2 = max(0, min(ancho, int(caja[2])))
    y2 = max(0, min(alto, int(caja[3])))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    region = frame_bgr[y1:y2, x1:x2]
    region_ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
    mascara_piel = cv2.inRange(region_ycrcb, (0, 133, 77), (255, 173, 127))
    return cv2.countNonZero(mascara_piel) / ((x2 - x1) * (y2 - y1))


def caja_ocupa_demasiada_area(caja, forma_frame):
    """Descarta cajas que cubren más del porcentaje permitido del fotograma."""
    alto, ancho = forma_frame[:2]
    area_frame = ancho * alto
    if area_frame <= 0:
        return False
    ancho_caja = max(0.0, caja[2] - caja[0])
    alto_caja = max(0.0, caja[3] - caja[1])
    return (ancho_caja * alto_caja) / area_frame > AREA_CAJA_MAXIMA


def redimensionar_frame(frame, cv2):
    """Ajusta el fotograma al tamaño de la ventana sin bandas vacías."""
    interpolacion = (
        cv2.INTER_AREA
        if frame.shape[1] > ANCHO_VENTANA or frame.shape[0] > ALTO_VENTANA
        else cv2.INTER_LINEAR
    )
    return cv2.resize(
        frame,
        (ANCHO_VENTANA, ALTO_VENTANA),
        interpolation=interpolacion,
    )


def obtener_ruta_modelo():
    """Descarga el detector especializado o usa el archivo local si no hay red."""
    ruta_local = Path(__file__).resolve().with_name(ARCHIVO_MODELO)
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
        from requests.exceptions import RequestException
    except ImportError as error:
        if ruta_local.is_file():
            print(f"Aviso: Hugging Face Hub no está disponible; usaré {ruta_local}.")
            return str(ruta_local)
        raise RuntimeError(
            "Falta huggingface_hub. Instálalo con: "
            "python -m pip install -r requirements.txt"
        ) from error

    try:
        return hf_hub_download(
            repo_id=ID_REPOSITORIO,
            filename=ARCHIVO_MODELO,
            repo_type="model",
        )
    except (HfHubHTTPError, LocalEntryNotFoundError, RequestException) as error:
        if ruta_local.is_file():
            print(
                "Aviso: no se pudo descargar desde Hugging Face; "
                f"usaré el modelo local {ruta_local}."
            )
            return str(ruta_local)
        raise RuntimeError(
            "No se pudo descargar el modelo especializado y no existe el respaldo "
            f"local. Copia {ARCHIVO_MODELO} a la carpeta del proyecto y vuelve "
            "a ejecutar."
        ) from error


def interseccion_sobre_union(caja_a, caja_b):
    """Calcula IoU entre dos cajas (x1, y1, x2, y2)."""
    x1 = max(caja_a[0], caja_b[0])
    y1 = max(caja_a[1], caja_b[1])
    x2 = min(caja_a[2], caja_b[2])
    y2 = min(caja_a[3], caja_b[3])
    interseccion = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, caja_a[2] - caja_a[0]) * max(0.0, caja_a[3] - caja_a[1])
    area_b = max(0.0, caja_b[2] - caja_b[0]) * max(0.0, caja_b[3] - caja_b[1])
    union = area_a + area_b - interseccion
    return interseccion / union if union else 0.0


class SuavizadorTemporal:
    """Promedia cajas y confianza de cada objeto a través de varios fotogramas."""

    def __init__(self, historial=5, ausencias_maximas=2, iou_minimo=0.25):
        self.historial = historial
        self.ausencias_maximas = ausencias_maximas
        self.iou_minimo = iou_minimo
        self.rastros = []

    @staticmethod
    def _caja_promedio(rastro):
        cantidad = len(rastro["cajas"])
        return tuple(
            sum(caja[indice] for caja in rastro["cajas"]) / cantidad
            for indice in range(4)
        )

    def update(self, detecciones):
        pares = []
        for indice_rastro, rastro in enumerate(self.rastros):
            for indice_deteccion, deteccion in enumerate(detecciones):
                if rastro["clase"] != deteccion["clase"]:
                    continue
                iou = interseccion_sobre_union(
                    self._caja_promedio(rastro), deteccion["caja"]
                )
                if iou >= self.iou_minimo:
                    pares.append((iou, indice_rastro, indice_deteccion))

        rastros_usados = set()
        detecciones_usadas = set()
        for _, indice_rastro, indice_deteccion in sorted(pares, reverse=True):
            if (
                indice_rastro in rastros_usados
                or indice_deteccion in detecciones_usadas
            ):
                continue
            rastro = self.rastros[indice_rastro]
            deteccion = detecciones[indice_deteccion]
            rastro["cajas"].append(tuple(deteccion["caja"]))
            rastro["confianzas"].append(deteccion["confianza"])
            rastro["ausencias"] = 0
            rastros_usados.add(indice_rastro)
            detecciones_usadas.add(indice_deteccion)

        for indice_rastro, rastro in enumerate(self.rastros):
            if indice_rastro not in rastros_usados:
                rastro["ausencias"] += 1

        for indice_deteccion, deteccion in enumerate(detecciones):
            if indice_deteccion in detecciones_usadas:
                continue
            self.rastros.append(
                {
                    "clase": deteccion["clase"],
                    "categoria": deteccion["categoria"],
                    "objeto": deteccion["objeto"],
                    "cajas": deque(
                        [tuple(deteccion["caja"])], maxlen=self.historial
                    ),
                    "confianzas": deque(
                        [deteccion["confianza"]], maxlen=self.historial
                    ),
                    "ausencias": 0,
                }
            )

        self.rastros = [
            rastro
            for rastro in self.rastros
            if rastro["ausencias"] <= self.ausencias_maximas
        ]
        salida = []
        for rastro in self.rastros:
            cantidad = len(rastro["confianzas"])
            salida.append(
                {
                    "categoria": rastro["categoria"],
                    "objeto": rastro["objeto"],
                    "caja": self._caja_promedio(rastro),
                    "confianza": sum(rastro["confianzas"]) / cantidad,
                }
            )
        return salida


def preprocesar_cuadro(cuadro, cv2, clahe):
    """Mejora contraste local y reduce ligeramente el ruido antes de inferir."""
    ycrcb = cv2.cvtColor(cuadro, cv2.COLOR_BGR2YCrCb)
    luminancia, canal_cr, canal_cb = cv2.split(ycrcb)
    luminancia = clahe.apply(luminancia)
    mejorado = cv2.cvtColor(
        cv2.merge((luminancia, canal_cr, canal_cb)), cv2.COLOR_YCrCb2BGR
    )
    return cv2.GaussianBlur(mejorado, (3, 3), 0)


def configurar_camara(camara, cv2, ancho, alto, autofocus, brillo, exposicion):
    """Solicita resolución y controles; algunos controladores pueden ignorarlos."""
    ajustes = [
        (cv2.CAP_PROP_FRAME_WIDTH, ancho, "ancho"),
        (cv2.CAP_PROP_FRAME_HEIGHT, alto, "alto"),
        (cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0, "autofocus"),
    ]
    if brillo is not None:
        ajustes.append((cv2.CAP_PROP_BRIGHTNESS, brillo, "brillo"))
    if exposicion is not None:
        ajustes.append((cv2.CAP_PROP_EXPOSURE, exposicion, "exposición"))

    for propiedad, valor, nombre in ajustes:
        if not camara.set(propiedad, valor):
            print(f"Aviso: la cámara no admite ajustar {nombre}.")

    ancho_real = int(camara.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_real = int(camara.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolución efectiva de cámara: {ancho_real}x{alto_real}")


def analizar_camara(
    indice,
    modelo,
    confianza,
    tamano,
    ancho,
    alto,
    autofocus,
    brillo,
    exposicion,
    historial,
):
    """Abre la cámara, dibuja detecciones compatibles y libera recursos al salir."""
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError(
            "Faltan dependencias. Instálalas con: "
            "python -m pip install -r requirements.txt"
        ) from error

    detector = YOLO(modelo)
    nombres_modelo = (
        list(detector.names.values())
        if isinstance(detector.names, dict)
        else list(detector.names)
    )
    print("Clases del modelo:", ", ".join(str(nombre) for nombre in nombres_modelo))
    fuente = ImageFont.truetype("arial.ttf", 16)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    suavizador = SuavizadorTemporal(historial=historial)
    detecciones_almacenadas = []
    numero_cuadro = 0
    camara = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    if not camara.isOpened():
        camara.release()
        raise RuntimeError(
            f"No se pudo abrir la cámara con índice {indice}. "
            "Prueba --camera 1 o revisa los permisos de Windows."
        )
    try:
        configurar_camara(
            camara, cv2, ancho, alto, autofocus, brillo, exposicion
        )
        nombre_ventana = "Clasificador de residuos - q/ESC para salir"
        cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(nombre_ventana, ANCHO_VENTANA, ALTO_VENTANA)
        print("Cámara activa. Presiona q o ESC para salir.")
        while True:
            leido, cuadro = camara.read()
            if not leido:
                raise RuntimeError("No se pudo leer un cuadro de la cámara.")

            if numero_cuadro % INTERVALO_INFERENCIA == 0:
                resultados = detector.predict(
                    source=preprocesar_cuadro(cuadro, cv2, clahe),
                    conf=confianza,
                    imgsz=tamano,
                    agnostic_nms=True,
                    iou=0.45,
                    device="cpu",
                    verbose=False,
                )
                resultado = resultados[0]
                detecciones_nuevas = []
                if resultado.boxes is not None:
                    for caja in resultado.boxes:
                        id_clase = int(caja.cls.item())
                        nombre_clase = resultado.names[id_clase]
                        clasificacion = clasificar_clase(nombre_clase)
                        if clasificacion is None:
                            continue

                        categoria, objeto = clasificacion
                        confianza_deteccion = float(caja.conf.item())
                        caja_xyxy = tuple(
                            float(coordenada)
                            for coordenada in caja.xyxy[0].tolist()
                        )
                        if caja_ocupa_demasiada_area(caja_xyxy, cuadro.shape):
                            continue
                        if (
                            normalizar_nombre_clase(nombre_clase) == "biological"
                            and porcentaje_piel_en_caja(
                                cuadro, caja_xyxy, cv2
                            )
                            > UMBRAL_PIEL_BIOLOGICO
                        ):
                            continue
                        if not supera_umbral(
                            nombre_clase, confianza_deteccion, confianza
                        ):
                            continue
                        detecciones_nuevas.append(
                            {
                                "clase": " ".join(
                                    nombre_clase.lower().replace("_", " ").split()
                                ),
                                "categoria": categoria,
                                "objeto": objeto,
                                "caja": caja_xyxy,
                                "confianza": confianza_deteccion,
                            }
                        )

                detecciones_almacenadas = suavizador.update(detecciones_nuevas)
            numero_cuadro += 1
            detecciones = detecciones_almacenadas

            imagen = Image.fromarray(cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB))
            dibujo = ImageDraw.Draw(imagen)

            for deteccion in detecciones:
                x1, y1, x2, y2 = (
                    int(round(coordenada)) for coordenada in deteccion["caja"]
                )
                color = COLORES[deteccion["categoria"]]
                etiqueta = (
                    f"{deteccion['objeto']} | {deteccion['categoria']} | "
                    f"{deteccion['confianza']:.0%}"
                )
                limites = dibujo.textbbox((0, 0), etiqueta, font=fuente)
                ancho_texto = limites[2] - limites[0]
                alto_texto = limites[3] - limites[1]
                texto_x = min(max(x1, 0), max(0, imagen.width - ancho_texto - 10))
                texto_y = y1 - alto_texto - 8
                if texto_y < 0:
                    texto_y = min(y1 + 3, imagen.height - alto_texto - 8)

                dibujo.rectangle((x1, y1, x2, y2), outline=color, width=3)
                dibujo.rectangle(
                    (
                        texto_x,
                        texto_y,
                        texto_x + ancho_texto + 10,
                        texto_y + alto_texto + 8,
                    ),
                    fill=color,
                )
                dibujo.text(
                    (texto_x + 5, texto_y + 3),
                    etiqueta,
                    fill=(255, 255, 255),
                    font=fuente,
                )

            if not detecciones:
                mensaje = "Enfoca un objeto compatible con el modelo"
                dibujo.rectangle((8, 8, 360, 34), fill=(40, 40, 40))
                dibujo.text(
                    (12, 11),
                    mensaje,
                    fill=(255, 255, 255),
                    font=fuente,
                )

            cuadro = cv2.cvtColor(np.asarray(imagen), cv2.COLOR_RGB2BGR)
            cv2.imshow(nombre_ventana, redimensionar_frame(cuadro, cv2))
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
    finally:
        camara.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Detecta y clasifica residuos con la cámara web."
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Índice de cámara (predeterminado: 0)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Pesos YOLO locales opcionales; por defecto descarga el modelo de "
            "residuos de Hugging Face"
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Confianza mínima entre 0 y 1 (predeterminado: 0.35)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamaño de inferencia (predeterminado: 640)",
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Ancho solicitado a la cámara"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Alto solicitado a la cámara"
    )
    parser.add_argument(
        "--history",
        type=int,
        choices=(3, 4, 5),
        default=5,
        help="Fotogramas usados para suavizar detecciones (predeterminado: 5)",
    )
    parser.add_argument(
        "--no-autofocus",
        action="store_false",
        dest="autofocus",
        help="Desactiva autofocus si la cámara no lo soporta o causa problemas",
    )
    parser.set_defaults(autofocus=True)
    parser.add_argument(
        "--brightness",
        type=float,
        default=None,
        help="Brillo de cámara (rango dependiente del controlador; opcional)",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="Exposición de cámara (valor dependiente del controlador; opcional)",
    )
    argumentos = parser.parse_args()

    if not 0 <= argumentos.confidence <= 1:
        parser.error("--confidence debe estar entre 0 y 1.")
    if argumentos.imgsz <= 0:
        parser.error("--imgsz debe ser mayor que cero.")
    if argumentos.width <= 0 or argumentos.height <= 0:
        parser.error("--width y --height deben ser mayores que cero.")

    try:
        modelo = argumentos.model or obtener_ruta_modelo()
        analizar_camara(
            argumentos.camera,
            modelo,
            argumentos.confidence,
            argumentos.imgsz,
            argumentos.width,
            argumentos.height,
            argumentos.autofocus,
            argumentos.brightness,
            argumentos.exposure,
            argumentos.history,
        )
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nEjecución interrumpida.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())