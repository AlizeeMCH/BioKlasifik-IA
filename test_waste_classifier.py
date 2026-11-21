import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import waste_classifier
from waste_classifier import (
    ServicioCamara,
    SuavizadorTemporal,
    analizar_apariencia_color,
    caja_ocupa_demasiada_area,
    clasificar_clase,
    decidir_categoria,
    limitar_caja_visible,
    porcentaje_piel_en_caja,
    redimensionar_frame,
    supera_umbral,
)


class ClasificacionTests(unittest.TestCase):
    def test_busca_pesos_en_la_carpeta_del_script_y_no_en_la_red(self):
        with tempfile.TemporaryDirectory() as temporal:
            carpeta = Path(temporal)
            peso = carpeta / waste_classifier.ARCHIVO_MODELO
            peso.touch()

            with patch.object(waste_classifier, "__file__", str(carpeta / "waste_classifier.py")):
                ruta = waste_classifier.obtener_ruta_modelo()

        self.assertEqual(ruta, str(peso.resolve()))

    def test_resuelve_el_modelo_secundario_junto_al_script(self):
        with tempfile.TemporaryDirectory() as temporal:
            carpeta = Path(temporal)
            peso = carpeta / "yolo11n.pt"
            peso.touch()

            with patch.object(
                waste_classifier,
                "__file__",
                str(carpeta / "waste_classifier.py"),
            ):
                ruta = waste_classifier.resolver_ruta_modelo("yolo11n.pt")

        self.assertEqual(ruta, str(peso.resolve()))

    def test_falla_con_mensaje_claro_si_falta_un_peso_local(self):
        with tempfile.TemporaryDirectory() as temporal:
            with patch.object(
                waste_classifier,
                "__file__",
                str(Path(temporal) / "waste_classifier.py"),
            ), patch.object(Path, "cwd", return_value=Path(temporal)):
                with self.assertRaisesRegex(RuntimeError, "no descarga"):
                    waste_classifier.obtener_ruta_modelo()

    def test_cli_predeterminado_no_importa_la_aplicacion_web(self):
        import builtins
        import sys

        importar_original = builtins.__import__

        def importar_sin_app(nombre, *argumentos, **opciones):
            if nombre == "app":
                raise AssertionError("El modo Python no debe importar app.py")
            return importar_original(nombre, *argumentos, **opciones)

        with (
            patch.object(sys, "argv", ["waste_classifier.py"]),
            patch.object(waste_classifier, "analizar_camara") as analizar,
            patch("builtins.__import__", side_effect=importar_sin_app),
        ):
            self.assertEqual(waste_classifier.main(), 0)

        analizar.assert_called_once()

    def test_clasifica_residuos_organicos(self):
        self.assertEqual(
            clasificar_clase("biological"),
            "Basura Orgánica",
        )
        self.assertEqual(
            clasificar_clase("food scraps"),
            "Basura Orgánica",
        )
        for clase in (
            "banana",
            "green chili",
            "lemon",
            "tomato",
            "avocado",
        ):
            with self.subTest(clase=clase):
                self.assertEqual(clasificar_clase(clase), "Basura Orgánica")

    def test_clasifica_residuos_inorganicos(self):
        clases = (
            "plastic",
            "paper",
            "cardboard",
            "metal",
            "glass",
            "brown-glass",
            "green-glass",
            "white-glass",
            "battery",
            "clothes",
            "shoes",
            "trash",
            "bottle",
            "paper towel",
        )
        for clase in clases:
            with self.subTest(clase=clase):
                self.assertEqual(
                    clasificar_clase(clase),
                    "Basura Inorgánica",
                )

    def test_normaliza_etiquetas_y_clasifica_alias_organicos(self):
        self.assertEqual(
            clasificar_clase("BIological-waste"),
            "Basura Orgánica",
        )

    def test_ignora_clases_no_mapeadas(self):
        self.assertIsNone(clasificar_clase("unknown gadget"))

    def test_umbral_especifico_reduce_falsos_positivos_de_ropa(self):
        self.assertFalse(supera_umbral("clothes", 0.607, 0.35))
        self.assertTrue(supera_umbral("clothes", 0.65, 0.35))
        self.assertTrue(supera_umbral("paper", 0.36, 0.35))

    def test_umbral_de_vidrio_filtra_confianzas_bajas(self):
        self.assertFalse(supera_umbral("green-glass", 0.38, 0.35))
        self.assertFalse(supera_umbral("brown-glass", 0.54, 0.35))
        self.assertTrue(supera_umbral("white-glass", 0.55, 0.35))

    def test_porcentaje_de_piel_en_caja(self):
        ycrcb = np.empty((10, 10, 3), dtype=np.uint8)
        ycrcb[:5, :, :] = (180, 150, 110)
        ycrcb[5:, :, :] = (180, 128, 128)
        frame_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        porcentaje = porcentaje_piel_en_caja(frame_bgr, (0, 0, 10, 10), cv2)

        self.assertGreater(porcentaje, 0.45)

    def test_descarta_cajas_gigantes_y_redimensiona_ventana(self):
        self.assertTrue(caja_ocupa_demasiada_area((0, 0, 90, 90), (100, 100, 3)))
        self.assertFalse(caja_ocupa_demasiada_area((0, 0, 80, 80), (100, 100, 3)))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(redimensionar_frame(frame, cv2).shape, (600, 800, 3))

    def test_conserva_aspecto_y_limita_cajas_y_etiquetas(self):
        frame = np.zeros((640, 480, 3), dtype=np.uint8)
        redimensionado = redimensionar_frame(frame, cv2)
        self.assertEqual(redimensionado.shape, (600, 800, 3))
        self.assertEqual(limitar_caja_visible((-10, -20, 500, 700), 480, 640), (12, 38, 468, 628))
        self.assertIsNone(limitar_caja_visible((0, 0, 10, 30), 480, 640))

    def test_analiza_el_color_en_el_centro_y_ignora_reflejos(self):
        frame = np.full((100, 100, 3), (30, 60, 90), dtype=np.uint8)
        frame[20:80, 20:80] = (30, 80, 20)
        frame[47:53, 47:53] = (255, 255, 255)
        apariencia = analizar_apariencia_color(
            frame,
            (0, 0, 100, 100),
            cv2,
        )
        decision = decidir_categoria(
            "Basura Inorgánica",
            0.9,
            None,
            0,
            apariencia,
        )
        self.assertGreaterEqual(apariencia["evidencia_organica"], 0.3)
        self.assertEqual(decision[0], "Basura Orgánica")

    def test_material_blanco_y_plastico_rosa_se_forzan_a_inorganico(self):
        blanco = np.full((80, 80, 3), 248, dtype=np.uint8)
        rosa = np.full((80, 80, 3), (165, 150, 220), dtype=np.uint8)
        for frame in (blanco, rosa):
            apariencia = analizar_apariencia_color(frame, (0, 0, 80, 80), cv2)
            decision = decidir_categoria(
                "Basura Orgánica",
                0.8,
                None,
                0,
                apariencia,
            )
            self.assertTrue(apariencia["inorganico_fuerte"])
            self.assertEqual(decision[0], "Basura Inorgánica")

    def test_filtra_piel_sin_descartar_comida_con_evidencia_cromatica(self):
        apariencia_piel = {
            "evidencia_organica": 0.0,
            "evidencia_inorganica": 0.0,
            "inorganico_fuerte": False,
            "piel": 0.7,
        }
        self.assertIsNone(
            decidir_categoria(
                "Basura Orgánica",
                0.8,
                None,
                0,
                apariencia_piel,
                es_clase_biologica=True,
            )
        )

    def test_suaviza_caja_y_confianza_entre_fotogramas(self):
        suavizador = SuavizadorTemporal(historial=5)
        deteccion = {
            "clase": "bottle",
            "categoria": "Basura Inorgánica",
            "objeto": "Botella",
            "caja": (10, 10, 110, 110),
            "confianza": 0.5,
        }
        suavizador.update([deteccion])
        deteccion["caja"] = (12, 10, 112, 110)
        deteccion["confianza"] = 0.7

        resultado = suavizador.update([deteccion])[0]

        self.assertEqual(resultado["caja"], (11.0, 10.0, 111.0, 110.0))
        self.assertAlmostEqual(resultado["confianza"], 0.6)

    def test_conserva_deteccion_durante_dos_fotogramas_perdidos(self):
        suavizador = SuavizadorTemporal()
        deteccion = {
            "clase": "bottle",
            "categoria": "Basura Inorgánica",
            "objeto": "Botella",
            "caja": (10, 10, 110, 110),
            "confianza": 0.7,
        }
        suavizador.update([deteccion])

        self.assertEqual(len(suavizador.update([])), 1)
        self.assertEqual(len(suavizador.update([])), 1)
        self.assertEqual(suavizador.update([]), [])

    def test_exige_cuatro_de_seis_frames_para_cambiar_de_categoria(self):
        suavizador = SuavizadorTemporal(historial=6)

        def deteccion(categoria, confianza=0.55):
            return {
                "clase": categoria,
                "categoria": categoria,
                "objeto": "",
                "caja": (10, 10, 110, 110),
                "confianza": confianza,
            }

        self.assertEqual(
            suavizador.update([deteccion("Basura Orgánica")])[0]["categoria"],
            "Basura Orgánica",
        )
        for _ in range(3):
            self.assertEqual(
                suavizador.update([deteccion("Basura Inorgánica")])[0]["categoria"],
                "Basura Orgánica",
            )
        self.assertEqual(
            suavizador.update([deteccion("Basura Inorgánica")])[0]["categoria"],
            "Basura Inorgánica",
        )

    def test_confianza_superior_a_60_por_ciento_permite_cambio_inmediato(self):
        suavizador = SuavizadorTemporal()
        caja = (10, 10, 110, 110)
        suavizador.update(
            [
                {
                    "clase": "Basura Orgánica",
                    "categoria": "Basura Orgánica",
                    "objeto": "",
                    "caja": caja,
                    "confianza": 0.55,
                }
            ]
        )
        resultado = suavizador.update(
            [
                {
                    "clase": "Basura Inorgánica",
                    "categoria": "Basura Inorgánica",
                    "objeto": "",
                    "caja": caja,
                    "confianza": 0.61,
                }
            ]
        )
        self.assertEqual(resultado[0]["categoria"], "Basura Inorgánica")

    def test_fotogramas_sin_deteccion_cuentan_dentro_de_la_ventana(self):
        suavizador = SuavizadorTemporal(historial=6)
        caja = (10, 10, 110, 110)

        def deteccion(categoria):
            return {
                "clase": categoria,
                "categoria": categoria,
                "objeto": "",
                "caja": caja,
                "confianza": 0.55,
            }

        suavizador.update([deteccion("Basura Orgánica")])
        suavizador.update([deteccion("Basura Inorgánica")])
        suavizador.update([deteccion("Basura Inorgánica")])
        suavizador.update([])
        suavizador.update([])
        resultado = suavizador.update([deteccion("Basura Inorgánica")])

        self.assertEqual(resultado[0]["categoria"], "Basura Orgánica")

    def test_captura_sigue_fluida_mientras_el_motor_infiere(self):
        class CamaraFalsa:
            def __init__(self):
                self.lecturas = 0
                self.liberada = False

            def isOpened(self):
                return True

            def set(self, _propiedad, _valor):
                return True

            def get(self, propiedad):
                if propiedad == cv2.CAP_PROP_FRAME_WIDTH:
                    return 640
                if propiedad == cv2.CAP_PROP_FRAME_HEIGHT:
                    return 480
                return 0

            def read(self):
                if self.liberada:
                    return False, None
                self.lecturas += 1
                time.sleep(0.002)
                return True, np.zeros((480, 640, 3), dtype=np.uint8)

            def release(self):
                self.liberada = True

        camara = CamaraFalsa()

        class MotorFalso:
            def __init__(self, *argumentos):
                self.cv2 = cv2
                self.confianza = argumentos[2]

            def detectar(self, _cuadro):
                time.sleep(0.25)
                return [
                    {
                        "clase": "banana",
                        "categoria": "Basura Orgánica",
                        "objeto": "",
                        "caja": (20, 40, 300, 300),
                        "confianza": 0.72,
                    }
                ]

        servicio = ServicioCamara()
        try:
            with (
                patch("cv2.VideoCapture", return_value=camara),
                patch.object(waste_classifier, "MotorDeteccion", MotorFalso),
            ):
                servicio.start()
                limite = time.monotonic() + 3
                estado = servicio.obtener_estado()
                while (
                    estado["detection"] is None
                    and time.monotonic() < limite
                ):
                    time.sleep(0.01)
                    estado = servicio.obtener_estado()
                cuadro, _ = servicio.obtener_snapshot()

            self.assertEqual(estado["state"], "listo")
            self.assertEqual(
                estado["detection"]["category"],
                "Basura Orgánica",
            )
            self.assertIsNotNone(cuadro)
            self.assertGreater(camara.lecturas, 20)
        finally:
            servicio.stop()
        self.assertTrue(camara.liberada)


if __name__ == "__main__":
    unittest.main()
