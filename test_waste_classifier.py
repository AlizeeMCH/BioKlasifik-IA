import unittest

import cv2
import numpy as np

from waste_classifier import (
    SuavizadorTemporal,
    caja_ocupa_demasiada_area,
    clasificar_clase,
    porcentaje_piel_en_caja,
    redimensionar_frame,
    supera_umbral,
)


class ClasificacionTests(unittest.TestCase):
    def test_clasifica_residuos_organicos(self):
        self.assertEqual(
            clasificar_clase("biological"),
            ("Basura Orgánica", "Orgánico"),
        )
        self.assertEqual(
            clasificar_clase("food scraps"),
            ("Basura Orgánica", "Orgánico"),
        )

    def test_clasifica_residuos_inorganicos(self):
        clases = {
            "plastic": "Plástico",
            "paper": "Papel",
            "cardboard": "Cartón",
            "metal": "Metal",
            "glass": "Vidrio",
            "brown-glass": "Vidrio marrón",
            "green-glass": "Vidrio verde",
            "white-glass": "Vidrio transparente",
            "battery": "Batería",
            "clothes": "Residuo Inorgánico",
            "shoes": "Calzado",
            "trash": "Residuo mixto",
        }
        for clase, material in clases.items():
            with self.subTest(clase=clase):
                self.assertEqual(
                    clasificar_clase(clase),
                    ("Basura Inorgánica", material),
                )

    def test_normaliza_etiquetas_y_clasifica_alias_organicos(self):
        self.assertEqual(
            clasificar_clase("BIological-waste"),
            ("Basura Orgánica", "Orgánico"),
        )

    def test_ignora_clases_no_mapeadas(self):
        self.assertIsNone(clasificar_clase("cell phone"))

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


if __name__ == "__main__":
    unittest.main()
