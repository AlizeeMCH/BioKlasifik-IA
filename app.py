"""Servidor local Flask para la interfaz web y el motor de detección Python."""

import math
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_from_directory,
    stream_with_context,
)

from waste_classifier import (
    ALTO_VENTANA,
    ANCHO_VENTANA,
    ServicioCamara,
    cargar_fuente_pil,
    dibujar_detecciones,
    redimensionar_frame,
)


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"


def crear_app(servicio=None):
    """Crea las rutas de interfaz, stream MJPEG y estado del detector."""
    servicio = servicio or ServicioCamara()
    aplicacion = Flask(
        __name__,
        static_folder=str(DIST),
        static_url_path="",
    )
    aplicacion.extensions["servicio_camara"] = servicio

    @aplicacion.get("/")
    def index():
        return send_from_directory(DIST, "index.html")

    @aplicacion.get("/api/status")
    def estado():
        return jsonify(servicio.obtener_estado())

    @aplicacion.post("/api/start")
    def iniciar_camara():
        servicio.start()
        return jsonify(servicio.obtener_estado()), 202

    @aplicacion.post("/api/stop")
    def detener_camara():
        servicio.stop()
        return jsonify(servicio.obtener_estado())

    @aplicacion.post("/api/config")
    def configurar():
        payload = request.get_json()
        confianza = payload.get("confidence") if isinstance(payload, dict) else None
        if (
            isinstance(confianza, bool)
            or not isinstance(confianza, (int, float))
            or not math.isfinite(confianza)
            or not 0 <= confianza <= 1
        ):
            return jsonify({"error": "confidence debe estar entre 0 y 1."}), 400
        servicio.set_confianza(float(confianza))
        return jsonify({"confidence": float(confianza)})

    @aplicacion.get("/video_feed")
    def video_feed():
        def generar():
            import cv2
            import numpy as np

            fuente = cargar_fuente_pil()
            while True:
                cuadro, detecciones = servicio.obtener_snapshot()
                if cuadro is None:
                    cuadro = np.zeros(
                        (ALTO_VENTANA, ANCHO_VENTANA, 3),
                        dtype=np.uint8,
                    )
                    mensaje = servicio.obtener_estado()["message"][:80]
                    cv2.putText(
                        cuadro,
                        mensaje,
                        (14, 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                else:
                    cuadro = dibujar_detecciones(cuadro, detecciones, cv2, fuente)
                cuadro = redimensionar_frame(cuadro, cv2)
                codificado, imagen = cv2.imencode(
                    ".jpg",
                    cuadro,
                    [cv2.IMWRITE_JPEG_QUALITY, 82],
                )
                if not codificado:
                    raise RuntimeError("OpenCV no pudo codificar el fotograma MJPEG.")
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + imagen.tobytes()
                    + b"\r\n"
                )
                time.sleep(1 / 30)

        return Response(
            stream_with_context(generar()),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    return aplicacion


def construir_frontend():
    """Genera los archivos estáticos que Flask servirá, sin tocar Vercel."""
    ejecutable_npm = shutil.which("npm.cmd") if sys.platform == "win32" else None
    ejecutable_npm = ejecutable_npm or shutil.which("npm")
    if not ejecutable_npm:
        raise RuntimeError(
            "No se encontró npm. Instala Node.js y ejecuta npm install "
            "antes de iniciar el servidor web."
        )
    try:
        subprocess.run(
            [ejecutable_npm, "run", "build"],
            cwd=ROOT,
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "No se pudo ejecutar npm. Revisa la instalación de Node.js."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Falló npm run build; revisa el error de Vite antes de abrir Flask."
        ) from error


def ejecutar_servidor(
    indice=0,
    modelo=None,
    modelo_objetos="yolo11n.pt",
    confianza=0.35,
    tamano=640,
    ancho=640,
    alto=480,
    autofocus=True,
    brillo=None,
    exposicion=None,
    historial=5,
    host="127.0.0.1",
    port=5000,
):
    """Construye la web, inicia la cámara y abre Flask en el navegador."""
    construir_frontend()
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
    aplicacion = crear_app(servicio)
    url = f"http://{host}:{port}/"
    threading.Timer(1.0, webbrowser_open, args=(url,)).start()
    print(f"BioKlasifik IA con YOLO local: {url}")
    try:
        aplicacion.run(
            host=host,
            port=port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    finally:
        servicio.stop()


def webbrowser_open(url):
    import webbrowser

    if not webbrowser.open(url):
        print(f"No se pudo abrir el navegador automáticamente. Visita {url}.")


if __name__ == "__main__":
    try:
        ejecutar_servidor()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
