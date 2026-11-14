import unittest

import numpy as np

from app import crear_app


class ServicioFalso:
    def __init__(self):
        self.started = False
        self.confianza = 0.35

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def set_confianza(self, confianza):
        self.confianza = confianza

    def obtener_estado(self):
        return {
            "python_backend": True,
            "state": "listo" if self.started else "detenido",
            "message": "Estado de prueba",
            "error": None,
            "detection": None,
        }

    def obtener_snapshot(self):
        if not self.started:
            return None, []
        return np.zeros((480, 640, 3), dtype=np.uint8), []


class AppTests(unittest.TestCase):
    def setUp(self):
        self.servicio = ServicioFalso()
        self.app = crear_app(self.servicio)
        self.app.testing = True
        self.client = self.app.test_client()

    def test_api_status_indica_motor_python(self):
        respuesta = self.client.get("/api/status")

        self.assertEqual(respuesta.status_code, 200)
        self.assertTrue(respuesta.json["python_backend"])

    def test_sirve_el_build_estatico_de_vite(self):
        with self.client.get("/") as respuesta:
            self.assertEqual(respuesta.status_code, 200)
            self.assertIn(b"BioKlasifik IA", respuesta.data)
            self.assertIn(b"/assets/", respuesta.data)

    def test_start_stop_y_configuracion_de_confianza(self):
        self.assertEqual(self.client.post("/api/start").status_code, 202)
        respuesta = self.client.post(
            "/api/config",
            json={"confidence": 0.6},
        )
        self.assertEqual(respuesta.status_code, 200)
        self.assertEqual(self.servicio.confianza, 0.6)

        respuesta_invalida = self.client.post(
            "/api/config",
            json={"confidence": 1.5},
        )
        self.assertEqual(respuesta_invalida.status_code, 400)
        self.assertEqual(self.client.post("/api/stop").status_code, 200)
        self.assertFalse(self.servicio.started)

    def test_video_feed_entrega_frames_mjpeg(self):
        self.servicio.start()
        respuesta = self.client.get("/video_feed", buffered=False)
        primer_frame = next(respuesta.response)

        self.assertIn(b"--frame", primer_frame)
        self.assertIn(b"image/jpeg", primer_frame)
        respuesta.close()


if __name__ == "__main__":
    unittest.main()
