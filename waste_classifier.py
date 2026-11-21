"""Detección de residuos con cámara web y un YOLO especializado en residuos."""

import argparse
import math
import platform
import sys
import threading
import time
import unicodedata
from collections import deque
from pathlib import Path


ID_REPOSITORIO = "kendrickfff/waste-classification-yolov8-ken"
ARCHIVO_MODELO = "yolov8n-waste-12cls-best.pt"
MODELO_OBJETOS = "yolo11n.pt"
INTERVALO_INFERENCIA = 0.2
HISTORIAL_CATEGORIAS = 6
VOTOS_CAMBIO_CATEGORIA = 4
UMBRAL_CAMBIO_CONFIANZA = 0.60
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
    "biological": "Basura Orgánica",
    "organic": "Basura Orgánica",
    "food": "Basura Orgánica",
    "fruit": "Basura Orgánica",
    "vegetable": "Basura Orgánica",
    "peel": "Basura Orgánica",
    "banana": "Basura Orgánica",
    "orange": "Basura Orgánica",
    "lemon": "Basura Orgánica",
    "lime": "Basura Orgánica",
    "apple": "Basura Orgánica",
    "tomato": "Basura Orgánica",
    "avocado": "Basura Orgánica",
    "jalapeno": "Basura Orgánica",
    "chili": "Basura Orgánica",
    "chile": "Basura Orgánica",
    "pepper": "Basura Orgánica",
    "bell pepper": "Basura Orgánica",
    "cucumber": "Basura Orgánica",
    "zucchini": "Basura Orgánica",
    "broccoli": "Basura Orgánica",
    "cauliflower": "Basura Orgánica",
    "cabbage": "Basura Orgánica",
    "corn": "Basura Orgánica",
    "carrot": "Basura Orgánica",
    "mushroom": "Basura Orgánica",
    "plant": "Basura Orgánica",
    "leaf": "Basura Orgánica",
    "sandwich": "Basura Orgánica",
    "hot dog": "Basura Orgánica",
    "pizza": "Basura Orgánica",
    "donut": "Basura Orgánica",
    "cake": "Basura Orgánica",
    "plastic": "Basura Inorgánica",
    "paper": "Basura Inorgánica",
    "paper towel": "Basura Inorgánica",
    "toilet tissue": "Basura Inorgánica",
    "cardboard": "Basura Inorgánica",
    "metal": "Basura Inorgánica",
    "glass": "Basura Inorgánica",
    "brown glass": "Basura Inorgánica",
    "green glass": "Basura Inorgánica",
    "white glass": "Basura Inorgánica",
    "battery": "Basura Inorgánica",
    "clothes": "Basura Inorgánica",
    "shoes": "Basura Inorgánica",
    "trash": "Basura Inorgánica",
    "bottle": "Basura Inorgánica",
    "cup": "Basura Inorgánica",
    "wine glass": "Basura Inorgánica",
    "can": "Basura Inorgánica",
    "book": "Basura Inorgánica",
    "scissors": "Basura Inorgánica",
    "cell phone": "Basura Inorgánica",
    "keyboard": "Basura Inorgánica",
    "remote": "Basura Inorgánica",
    "mouse": "Basura Inorgánica",
    "laptop": "Basura Inorgánica",
    "tv": "Basura Inorgánica",
    "fork": "Basura Inorgánica",
    "knife": "Basura Inorgánica",
    "spoon": "Basura Inorgánica",
    "bowl": "Basura Inorgánica",
    "backpack": "Basura Inorgánica",
    "handbag": "Basura Inorgánica",
    "suitcase": "Basura Inorgánica",
    "toothbrush": "Basura Inorgánica",
    "plastic bag": "Basura Inorgánica",
    "carton": "Basura Inorgánica",
}

COLORES = {
    "Basura Orgánica": (16, 185, 129),
    "Basura Inorgánica": (59, 130, 246),
}


def normalizar_nombre_clase(nombre_clase):
    normalizado = unicodedata.normalize("NFKD", nombre_clase.strip().lower())
    normalizado = "".join(
        caracter for caracter in normalizado if not unicodedata.combining(caracter)
    )
    return " ".join(normalizado.replace("_", " ").replace("-", " ").split())


def clasificar_clase(nombre_clase):
    """Mapea clases de ambos detectores a las dos categorías públicas."""
    normalizado = normalizar_nombre_clase(nombre_clase)
    if normalizado in CLASES_RESIDUOS:
        return CLASES_RESIDUOS[normalizado]
    palabras = set(normalizado.split())
    if any(
        palabra in palabras or palabra in normalizado
        for palabra in (
            "organic",
            "biological",
            "food",
            "fruit",
            "peel",
            "vegetable",
            "produce",
            "banana",
            "orange",
            "lemon",
            "lime",
            "apple",
            "tomato",
            "avocado",
            "jalapeno",
            "chili",
            "chile",
            "pepper",
            "cucumber",
            "zucchini",
            "broccoli",
            "cauliflower",
            "cabbage",
            "corn",
            "carrot",
            "mushroom",
            "plant",
            "leaf",
        )
    ):
        return "Basura Orgánica"
    return None


def supera_umbral(nombre_clase, confianza_deteccion, umbral_base):
    """Aplica umbrales específicos a ropa y vidrio."""
    umbral_clase = UMBRALES_POR_CLASE.get(
        normalizar_nombre_clase(nombre_clase), umbral_base
    )
    return confianza_deteccion >= max(umbral_base, umbral_clase)


def porcentaje_piel_en_caja(frame_bgr, caja, cv2):
    """Calcula piel en el 60% central de la caja para excluir el fondo."""
    region = recortar_centro_caja(frame_bgr, caja)
    if region is None:
        return 0.0

    region_ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
    mascara_piel = cv2.inRange(region_ycrcb, (0, 133, 77), (255, 173, 127))
    return cv2.countNonZero(mascara_piel) / (region.shape[0] * region.shape[1])


def recortar_centro_caja(frame, caja, fraccion=0.6):
    """Extrae el porcentaje central solicitado de una caja XYXY."""
    alto, ancho = frame.shape[:2]
    x1 = max(0, min(ancho, int(round(caja[0]))))
    y1 = max(0, min(alto, int(round(caja[1]))))
    x2 = max(0, min(ancho, int(round(caja[2]))))
    y2 = max(0, min(alto, int(round(caja[3]))))
    if x2 <= x1 or y2 <= y1:
        return None

    margen_x = int((x2 - x1) * (1 - fraccion) / 2)
    margen_y = int((y2 - y1) * (1 - fraccion) / 2)
    centro = frame[
        y1 + margen_y : y2 - margen_y,
        x1 + margen_x : x2 - margen_x,
    ]
    return centro if centro.size else None


def analizar_apariencia_color(frame_bgr, caja, cv2):
    """Obtiene evidencia cromática del recorte central, sin contar reflejos."""
    import numpy as np

    region = recortar_centro_caja(frame_bgr, caja)
    if region is None:
        return {
            "evidencia_organica": 0.0,
            "evidencia_inorganica": 0.0,
            "inorganico_fuerte": False,
            "piel": 0.0,
        }

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue = hsv[:, :, 0] * 2
    saturation = hsv[:, :, 1] / 255
    value = hsv[:, :, 2] / 255
    pixeles = value.size
    reflejos = (value > 0.9) & (saturation < 0.24)
    validos = ~reflejos

    verdes = (
        (hue >= 35)
        & (hue <= 165)
        & (saturation >= 0.18)
        & (value > 0.1)
    )
    calidos = (
        ((hue <= 40) | (hue >= 355))
        & (saturation >= 0.3)
        & (value > 0.12)
    )
    amarillos = (
        (hue >= 20)
        & (hue <= 70)
        & (saturation >= 0.3)
        & (value > 0.12)
    )
    marron_oscuro = (
        (hue >= 5)
        & (hue <= 50)
        & (saturation >= 0.2)
        & (value <= 0.48)
    )
    organicos = (verdes | calidos | amarillos | marron_oscuro) & validos
    elegibles = (saturation >= 0.08) & (value >= 0.07) & validos
    cantidad_organica = int(np.count_nonzero(organicos))
    cantidad_elegible = int(np.count_nonzero(elegibles))
    proporcion_cromatica = (
        cantidad_organica / cantidad_elegible if cantidad_elegible else 0.0
    )
    cobertura_organica = min(1.0, cantidad_organica / max(1, pixeles) / 0.3)
    evidencia_base = 0.65 * math.sqrt(
        proporcion_cromatica * cobertura_organica
    )

    cantidad_marron = int(np.count_nonzero(marron_oscuro & validos))
    textura = 0.0
    if cantidad_marron:
        gris = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        varianza_bordes = float(cv2.Laplacian(gris, cv2.CV_32F).var())
        textura = min(
            0.12,
            (varianza_bordes / 400)
            * (cantidad_marron / max(1, pixeles))
            * 0.3,
        )
    evidencia_organica = min(0.65, evidencia_base + textura)

    neutrales = (value >= 0.62) & (saturation <= 0.2)
    rosas = (
        (hue >= 320)
        & (hue <= 355)
        & (saturation >= 0.1)
        & (saturation <= 0.65)
        & (value >= 0.45)
    )
    proporcion_neutral = float(np.count_nonzero(neutrales) / pixeles)
    proporcion_rosa = float(np.count_nonzero(rosas) / pixeles)
    inorganico_fuerte = proporcion_neutral >= 0.62 or proporcion_rosa >= 0.3

    return {
        "evidencia_organica": evidencia_organica,
        "evidencia_inorganica": (
            max(proporcion_neutral, proporcion_rosa) if inorganico_fuerte else 0.0
        ),
        "inorganico_fuerte": inorganico_fuerte,
        "piel": porcentaje_piel_en_caja(frame_bgr, caja, cv2),
    }


def decidir_categoria(
    categoria_residuos,
    confianza_residuos,
    categoria_objeto,
    confianza_objeto,
    apariencia,
    es_clase_biologica=False,
):
    """Fusiona modelos semánticos, color y el filtro de piel."""
    if apariencia["inorganico_fuerte"]:
        return (
            "Basura Inorgánica",
            max(0.61, apariencia["evidencia_inorganica"]),
        )
    if categoria_objeto:
        return categoria_objeto, confianza_objeto

    evidencia_organica = apariencia["evidencia_organica"]
    if (
        categoria_residuos == "Basura Orgánica"
        and es_clase_biologica
        and apariencia["piel"] > UMBRAL_PIEL_BIOLOGICO
        and evidencia_organica < 0.3
    ):
        return None
    if (
        categoria_residuos == "Basura Inorgánica"
        and evidencia_organica >= 0.3
    ):
        return "Basura Orgánica", evidencia_organica
    if categoria_residuos:
        return categoria_residuos, confianza_residuos
    if evidencia_organica >= 0.3:
        return "Basura Orgánica", evidencia_organica
    return None


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
    """Ajusta el fotograma preservando su relación de aspecto."""
    import numpy as np

    alto, ancho = frame.shape[:2]
    escala = min(ANCHO_VENTANA / ancho, ALTO_VENTANA / alto)
    ancho_nuevo = max(1, round(ancho * escala))
    alto_nuevo = max(1, round(alto * escala))
    interpolacion = (
        cv2.INTER_AREA
        if escala < 1
        else cv2.INTER_LINEAR
    )
    redimensionado = cv2.resize(
        frame, (ancho_nuevo, alto_nuevo), interpolation=interpolacion
    )
    lienzo = np.zeros((ALTO_VENTANA, ANCHO_VENTANA, 3), dtype=frame.dtype)
    x = (ANCHO_VENTANA - ancho_nuevo) // 2
    y = (ALTO_VENTANA - alto_nuevo) // 2
    lienzo[y : y + alto_nuevo, x : x + ancho_nuevo] = redimensionado
    return lienzo


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
    """Suaviza cajas y cambia de categoría solo con evidencia temporal suficiente."""

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
            rastro["categorias"].append(deteccion["categoria"])
            rastro["evidencias"].append(
                (deteccion["categoria"], deteccion["confianza"])
            )
            rastro["clase"] = deteccion["clase"]
            rastro["objeto"] = deteccion["objeto"]
            rastro["ausencias"] = 0
            categoria_nueva = deteccion["categoria"]
            if categoria_nueva != rastro["categoria"]:
                votos = sum(
                    categoria == categoria_nueva
                    for categoria in rastro["categorias"]
                )
                if (
                    votos >= VOTOS_CAMBIO_CATEGORIA
                    or deteccion["confianza"] > UMBRAL_CAMBIO_CONFIANZA
                ):
                    rastro["categoria"] = categoria_nueva
            rastros_usados.add(indice_rastro)
            detecciones_usadas.add(indice_deteccion)

        for indice_rastro, rastro in enumerate(self.rastros):
            if indice_rastro not in rastros_usados:
                rastro["ausencias"] += 1
                rastro["categorias"].append(None)
                rastro["evidencias"].append((None, 0.0))

        for indice_deteccion, deteccion in enumerate(detecciones):
            if indice_deteccion in detecciones_usadas:
                continue
            self.rastros.append(
                {
                    "clase": deteccion["clase"],
                    "categoria": deteccion["categoria"],
                    "objeto": deteccion["objeto"],
                    "categorias": deque(
                        [deteccion["categoria"]],
                        maxlen=HISTORIAL_CATEGORIAS,
                    ),
                    "evidencias": deque(
                        [(deteccion["categoria"], deteccion["confianza"])],
                        maxlen=HISTORIAL_CATEGORIAS,
                    ),
                    "cajas": deque(
                        [tuple(deteccion["caja"])], maxlen=self.historial
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
            confianzas = [
                confianza
                for categoria, confianza in rastro["evidencias"]
                if categoria == rastro["categoria"]
            ]
            salida.append(
                {
                    "categoria": rastro["categoria"],
                    "objeto": rastro["objeto"],
                    "caja": self._caja_promedio(rastro),
                    "confianza": sum(confianzas) / len(confianzas),
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


def extraer_detecciones(resultado, umbral):
    """Convierte una salida Ultralytics en detecciones de las dos categorías."""
    detecciones = []
    if resultado.boxes is None:
        return detecciones

    nombres = resultado.names
    for caja in resultado.boxes:
        id_clase = int(caja.cls.item())
        nombre = nombres[id_clase]
        categoria = clasificar_clase(nombre)
        confianza = float(caja.conf.item())
        if categoria is None or confianza < umbral:
            continue
        if not supera_umbral(nombre, confianza, umbral):
            continue

        caja_xyxy = tuple(float(valor) for valor in caja.xyxy[0].tolist())
        if caja_ocupa_demasiada_area(caja_xyxy, resultado.orig_shape):
            continue
        detecciones.append(
            {
                "clase": normalizar_nombre_clase(nombre),
                "categoria": categoria,
                "objeto": "",
                "caja": caja_xyxy,
                "confianza": confianza,
            }
        )
    return detecciones


def fusionar_detecciones(especializadas, objetos, frame, cv2):
    """Prioriza clases reales de COCO y usa color solo como evidencia auxiliar."""
    fusionadas = []
    objetos_usados = set()

    for especializada in especializadas:
        candidata = None
        indice_candidata = None
        mejor_iou = 0.05
        for indice, objeto in enumerate(objetos):
            if indice in objetos_usados:
                continue
            iou = interseccion_sobre_union(
                especializada["caja"], objeto["caja"]
            )
            if iou > mejor_iou:
                mejor_iou = iou
                candidata = objeto
                indice_candidata = indice

        if indice_candidata is not None:
            objetos_usados.add(indice_candidata)
        apariencia = analizar_apariencia_color(
            frame, especializada["caja"], cv2
        )
        categoria_objeto = candidata["categoria"] if candidata else None
        confianza_objeto = candidata["confianza"] if candidata else 0.0
        decision = decidir_categoria(
            especializada["categoria"],
            especializada["confianza"],
            categoria_objeto,
            confianza_objeto,
            apariencia,
            es_clase_biologica=especializada["clase"] == "biological",
        )
        if decision is None:
            continue

        categoria, confianza = decision
        fusionadas.append(
            {
                **especializada,
                "categoria": categoria,
                "confianza": confianza,
            }
        )

    for indice, objeto in enumerate(objetos):
        if indice in objetos_usados:
            continue
        apariencia = analizar_apariencia_color(frame, objeto["caja"], cv2)
        decision = decidir_categoria(
            None,
            0.0,
            objeto["categoria"],
            objeto["confianza"],
            apariencia,
        )
        if decision is None:
            continue
        categoria, confianza = decision
        fusionadas.append(
            {
                **objeto,
                "categoria": categoria,
                "confianza": confianza,
            }
        )

    return fusionadas


class MotorDeteccion:
    """Combina el detector de residuos, YOLO-COCO y el análisis del recorte."""

    def __init__(
        self,
        modelo,
        modelo_objetos,
        confianza,
        tamano,
        historial,
    ):
        import cv2
        from ultralytics import YOLO

        self.cv2 = cv2
        self.confianza = confianza
        self.tamano = tamano
        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        self.detector_residuos = YOLO(modelo or obtener_ruta_modelo())
        self.detector_objetos = YOLO(modelo_objetos)
        self.suavizador = SuavizadorTemporal(historial=historial)

    @staticmethod
    def _inferir_modelo(modelo, cuadro, confianza, tamano):
        resultados = modelo.predict(
            source=cuadro,
            conf=confianza,
            imgsz=tamano,
            agnostic_nms=True,
            iou=0.45,
            device="cpu",
            verbose=False,
        )
        return resultados[0]

    def detectar(self, cuadro):
        cuadro_mejorado = preprocesar_cuadro(
            cuadro,
            self.cv2,
            self.clahe,
        )
        resultado_residuos = self._inferir_modelo(
            self.detector_residuos,
            cuadro_mejorado,
            self.confianza,
            self.tamano,
        )
        resultado_objetos = self._inferir_modelo(
            self.detector_objetos,
            cuadro,
            max(0.15, min(0.35, self.confianza * 0.6)),
            min(640, self.tamano),
        )
        especializadas = extraer_detecciones(
            resultado_residuos,
            self.confianza,
        )
        objetos = extraer_detecciones(
            resultado_objetos,
            max(0.15, min(0.35, self.confianza * 0.6)),
        )
        fusionadas = fusionar_detecciones(
            especializadas,
            objetos,
            cuadro,
            self.cv2,
        )
        return self.suavizador.update(fusionadas)


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


def cargar_fuente_pil():
    from PIL import ImageFont

    try:
        return ImageFont.truetype("arial.ttf", 16)
    except OSError:
        return ImageFont.load_default()


def limitar_caja_visible(caja, ancho, alto):
    """Mantiene la caja dentro de un margen seguro de la imagen."""
    x1 = max(12, min(ancho - 12, int(round(caja[0]))))
    y1 = max(38, min(alto - 12, int(round(caja[1]))))
    x2 = max(12, min(ancho - 12, int(round(caja[2]))))
    y2 = max(38, min(alto - 12, int(round(caja[3]))))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def dibujar_detecciones(cuadro, detecciones, cv2, fuente=None):
    """Dibuja solo categoría y porcentaje con caja y etiqueta recortadas."""
    import numpy as np
    from PIL import Image, ImageDraw

    if fuente is None:
        fuente = cargar_fuente_pil()
    imagen = Image.fromarray(cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB))
    dibujo = ImageDraw.Draw(imagen)
    for deteccion in detecciones:
        caja = limitar_caja_visible(
            deteccion["caja"],
            imagen.width,
            imagen.height,
        )
        if caja is None:
            continue

        x1, y1, x2, y2 = caja
        categoria = deteccion["categoria"]
        color = COLORES[categoria]
        etiqueta = f"{categoria} · {deteccion['confianza']:.0%}"
        limites = dibujo.textbbox((0, 0), etiqueta, font=fuente)
        ancho_texto = limites[2] - limites[0]
        alto_texto = limites[3] - limites[1]
        ancho_pastilla = min(imagen.width - 24, ancho_texto + 12)
        texto_x = min(max(12, x1), imagen.width - 12 - ancho_pastilla)
        if y1 <= 38:
            texto_y = max(12, y1 + 8)
        else:
            texto_y = max(12, y1 - alto_texto - 8)
        texto_y = min(texto_y, imagen.height - 12 - alto_texto - 6)

        dibujo.rectangle((x1, y1, x2, y2), outline=color, width=3)
        dibujo.rectangle(
            (
                texto_x,
                texto_y,
                texto_x + ancho_pastilla,
                texto_y + alto_texto + 6,
            ),
            fill=color,
        )
        dibujo.text(
            (texto_x + 6, texto_y + 3),
            etiqueta,
            fill=(255, 255, 255),
            font=fuente,
        )

    return cv2.cvtColor(np.asarray(imagen), cv2.COLOR_RGB2BGR)


class ServicioCamara:
    """Mantiene captura e inferencia independientes para ventana y MJPEG."""

    def __init__(
        self,
        indice=0,
        modelo=None,
        modelo_objetos=MODELO_OBJETOS,
        confianza=0.35,
        tamano=640,
        ancho=640,
        alto=480,
        autofocus=True,
        brillo=None,
        exposicion=None,
        historial=5,
    ):
        self.indice = indice
        self.modelo = modelo
        self.modelo_objetos = modelo_objetos
        self.confianza = confianza
        self.tamano = tamano
        self.ancho = ancho
        self.alto = alto
        self.autofocus = autofocus
        self.brillo = brillo
        self.exposicion = exposicion
        self.historial = historial
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._frame_evento = threading.Event()
        self._detener_evento = threading.Event()
        self._generacion = 0
        self._frame = None
        self._version_frame = 0
        self._detecciones = []
        self._estado = "detenido"
        self._mensaje = "La cámara Python está detenida."
        self._error = None
        self._camara = None
        self._motor = None
        self._camara_lista = False
        self._modelo_listo = False
        self._captura_thread = None
        self._inferencia_thread = None
        self._fuente = None

    def start(self):
        with self._lifecycle_lock:
            with self._lock:
                if self._estado not in ("detenido", "error"):
                    return
                self._generacion += 1
                generacion = self._generacion
                self._detener_evento.clear()
                self._frame_evento.clear()
                self._frame = None
                self._detecciones = []
                self._error = None
                self._camara_lista = False
                self._modelo_listo = False
                self._estado = "iniciando"
                self._mensaje = "Abriendo cámara y cargando modelos YOLO."
                self._captura_thread = threading.Thread(
                    target=self._capturar,
                    args=(generacion,),
                    name="bioklasifik-captura",
                    daemon=True,
                )
                self._inferencia_thread = threading.Thread(
                    target=self._inferir,
                    args=(generacion,),
                    name="bioklasifik-inferencia",
                    daemon=True,
                )
                self._captura_thread.start()
                self._inferencia_thread.start()

    def stop(self):
        with self._lifecycle_lock:
            self._detener_evento.set()
            self._frame_evento.set()
            with self._lock:
                self._generacion += 1
                camara = self._camara
                threads = (self._captura_thread, self._inferencia_thread)
            if camara is not None:
                camara.release()
            for thread in threads:
                if thread and thread is not threading.current_thread():
                    thread.join(timeout=3)
            with self._lock:
                self._camara = None
                self._frame = None
                self._detecciones = []
                self._camara_lista = False
                self._modelo_listo = False
                self._motor = None
                self._estado = "detenido"
                self._mensaje = "La cámara Python está detenida."
                self._captura_thread = None
                self._inferencia_thread = None

    def set_confianza(self, confianza):
        with self._lock:
            self.confianza = confianza
            if self._motor:
                self._motor.confianza = confianza

    def _actualizar_estado(self):
        if self._estado == "error":
            return
        if self._camara_lista and self._modelo_listo:
            self._estado = "listo"
            self._mensaje = "Cámara Python activa; inferencia YOLO en segundo plano."
        elif self._camara_lista:
            self._estado = "cargando_modelo"
            self._mensaje = "Cámara activa; cargando modelos de clasificación."
        elif self._modelo_listo:
            self._estado = "abriendo_camara"
            self._mensaje = "Modelos listos; abriendo cámara."

    def _fallar(self, error, generacion):
        mensaje = str(error).strip() or error.__class__.__name__
        with self._lock:
            if (
                generacion != self._generacion
                or self._detener_evento.is_set()
            ):
                return
            self._error = mensaje
            self._estado = "error"
            self._mensaje = f"Error del motor Python: {mensaje}"
        self._detener_evento.set()
        self._frame_evento.set()

    def _capturar(self, generacion):
        import cv2

        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        camara = cv2.VideoCapture(self.indice, backend)
        if not camara.isOpened():
            camara.release()
            self._fallar(
                RuntimeError(
                    f"No se pudo abrir la cámara {self.indice}; prueba --camera 1."
                ),
                generacion,
            )
            return

        with self._lock:
            if generacion != self._generacion:
                camara.release()
                return
            self._camara = camara
        try:
            configurar_camara(
                camara,
                cv2,
                self.ancho,
                self.alto,
                self.autofocus,
                self.brillo,
                self.exposicion,
            )
            camara.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            with self._lock:
                if (
                    generacion != self._generacion
                    or self._detener_evento.is_set()
                ):
                    return
                self._camara_lista = True
                self._actualizar_estado()

            while (
                not self._detener_evento.is_set()
                and generacion == self._generacion
            ):
                leido, cuadro = camara.read()
                if not leido:
                    if not self._detener_evento.is_set():
                        self._fallar(
                            RuntimeError("La cámara dejó de entregar fotogramas."),
                            generacion,
                        )
                    return
                with self._lock:
                    if generacion != self._generacion:
                        return
                    self._frame = cuadro
                    self._version_frame += 1
                self._frame_evento.set()
        except (OSError, RuntimeError, cv2.error) as error:
            if not self._detener_evento.is_set():
                self._fallar(error, generacion)
        finally:
            camara.release()
            with self._lock:
                if generacion == self._generacion:
                    self._camara = None
                    self._camara_lista = False

    def _inferir(self, generacion):
        try:
            motor = MotorDeteccion(
                self.modelo,
                self.modelo_objetos,
                self.confianza,
                self.tamano,
                self.historial,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            self._fallar(error, generacion)
            return

        with self._lock:
            if (
                generacion != self._generacion
                or self._detener_evento.is_set()
            ):
                return
            motor.confianza = self.confianza
            self._motor = motor
            self._modelo_listo = True
            self._actualizar_estado()

        version_procesada = -1
        ultima_inferencia = 0.0
        while (
            not self._detener_evento.is_set()
            and generacion == self._generacion
        ):
            if not self._frame_evento.wait(timeout=0.1):
                continue
            with self._lock:
                self._frame_evento.clear()
                if self._frame is None or self._version_frame == version_procesada:
                    continue
                cuadro = self._frame.copy()
                version_procesada = self._version_frame

            espera = INTERVALO_INFERENCIA - (time.monotonic() - ultima_inferencia)
            if espera > 0 and self._detener_evento.wait(espera):
                return
            ultima_inferencia = time.monotonic()
            try:
                detecciones = motor.detectar(cuadro)
            except (OSError, RuntimeError, ValueError, motor.cv2.error) as error:
                self._fallar(error, generacion)
                return
            with self._lock:
                if generacion == self._generacion:
                    self._detecciones = detecciones

    def obtener_snapshot(self):
        with self._lock:
            cuadro = self._frame.copy() if self._frame is not None else None
            return cuadro, [dict(item) for item in self._detecciones]

    def obtener_estado(self):
        with self._lock:
            principales = sorted(
                self._detecciones,
                key=lambda deteccion: deteccion["confianza"],
                reverse=True,
            )
            deteccion = None
            if principales:
                principal = principales[0]
                confianza = round(principal["confianza"], 4)
                deteccion = {
                    "category": principal["categoria"],
                    "confidence": confianza,
                    "percentage": round(confianza * 100),
                    "label": f"{principal['categoria']} · {confianza:.0%}",
                }
            return {
                "python_backend": True,
                "state": self._estado,
                "message": self._mensaje,
                "error": self._error,
                "detection": deteccion,
            }


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
    modelo_objetos=MODELO_OBJETOS,
):
    """Muestra la cámara; captura e inferencia se ejecutan en hilos separados."""
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Faltan dependencias. Instálalas con: "
            "python -m pip install -r requirements.txt"
        ) from error

    servicio = ServicioCamara(
        indice=indice,
        modelo=modelo,
        modelo_objetos=modelo_objetos,
        confianza=confianza,
        tamano=tamano,
        ancho=ancho,
        alto=alto,
        autofocus=autofocus,
        brillo=brillo,
        exposicion=exposicion,
        historial=historial,
    )
    nombre_ventana = "Clasificador de residuos - q/ESC para salir"
    fuente = cargar_fuente_pil()
    try:
        servicio.start()
        cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(nombre_ventana, ANCHO_VENTANA, ALTO_VENTANA)
        print(
            "Cámara activa. Captura e inferencia están separadas; "
            "presiona q o ESC para salir."
        )
        while True:
            cuadro, detecciones = servicio.obtener_snapshot()
            estado = servicio.obtener_estado()
            if estado["state"] == "error":
                raise RuntimeError(estado["error"])
            if cuadro is None:
                cuadro = np.zeros((ALTO_VENTANA, ANCHO_VENTANA, 3), dtype=np.uint8)
                cv2.putText(
                    cuadro,
                    estado["message"][:80],
                    (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
            else:
                cuadro = dibujar_detecciones(
                    cuadro,
                    detecciones,
                    cv2,
                    fuente,
                )
            cv2.imshow(nombre_ventana, redimensionar_frame(cuadro, cv2))
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
    finally:
        servicio.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Clasifica residuos con la interfaz web local o una ventana OpenCV."
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
        "--object-model",
        default=MODELO_OBJETOS,
        help="Modelo YOLO-COCO para reconocer objetos y alimentos.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Inicia Flask y abre la interfaz web (modo predeterminado).",
    )
    parser.add_argument(
        "--desktop",
        action="store_true",
        help="Usa la ventana clásica de OpenCV en lugar de la interfaz web.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Dirección local del servidor web (predeterminado: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Puerto del servidor web (predeterminado: 5000).",
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
        help=(
            "Historial para suavizar las cajas (la categoría usa 6 fotogramas; "
            "predeterminado: 5)"
        ),
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
    if not 1 <= argumentos.port <= 65535:
        parser.error("--port debe estar entre 1 y 65535.")

    try:
        if not argumentos.desktop:
            from app import ejecutar_servidor

            ejecutar_servidor(
                indice=argumentos.camera,
                modelo=argumentos.model,
                modelo_objetos=argumentos.object_model,
                confianza=argumentos.confidence,
                tamano=argumentos.imgsz,
                ancho=argumentos.width,
                alto=argumentos.height,
                autofocus=argumentos.autofocus,
                brillo=argumentos.brightness,
                exposicion=argumentos.exposure,
                historial=argumentos.history,
                host=argumentos.host,
                port=argumentos.port,
            )
        else:
            analizar_camara(
                argumentos.camera,
                argumentos.model,
                argumentos.confidence,
                argumentos.imgsz,
                argumentos.width,
                argumentos.height,
                argumentos.autofocus,
                argumentos.brightness,
                argumentos.exposure,
                argumentos.history,
                argumentos.object_model,
            )
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except ImportError as error:
        print(
            f"Falta una dependencia para el modo solicitado: {error}. "
            "Instala con: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\nEjecución interrumpida.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())