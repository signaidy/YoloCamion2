"""Detector de carriles para ETS2 — Volvo FH16 primera persona.

Usa detección por brillo (escala de grises) en lugar de HSV,
lo que es más robusto con el renderizado de ETS2 donde las marcas
viales son consistentemente brillantes sobre asfalto oscuro.
"""
from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np


@dataclass
class EstadoCarril:
    desviacion: float          # -1 a +1
    confianza: float           # 0-1
    linea_izq_detectada: bool
    linea_der_detectada: bool


# Posición esperada de la línea derecha cuando el camión está centrado
_POS_DER_CENTRADO = 0.68
_ESCALA_DER = 0.28


class DetectorCarriles:
    """Detecta marcas de carril usando brillo sobre asfalto.

    ETS2 renderiza las marcas viales como zonas brillantes (>160 de brillo)
    sobre asfalto oscuro (<80). Esto es más confiable que HSV para gráficos.
    """

    def __init__(
        self,
        zona_roi: tuple[float, float, float, float] = (0.15, 0.52, 0.85, 0.83),
        suavizado: int = 8,
        zona_muerta: float = 0.07,
    ):
        self._roi = zona_roi
        self._hist: deque[float] = deque(maxlen=suavizado)
        self._zona_muerta = zona_muerta

    def detectar(self, frame: np.ndarray) -> EstadoCarril:
        h, w = frame.shape[:2]
        xi, yt, xd, yb = self._roi
        x1, y1 = int(w * xi), int(h * yt)
        x2, y2 = int(w * xd), int(h * yb)
        roi = frame[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        # Convertir a escala de grises y aplicar CLAHE para equalizar
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Máscara de zonas brillantes (marcas viales blancas/amarillas)
        _, mask_bright = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

        # Eliminar ruido fino (reflejos, bordes de objetos)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        mask_bright = cv2.morphologyEx(mask_bright, cv2.MORPH_OPEN, kernel)

        edges = cv2.Canny(mask_bright, 30, 90)

        # maxLineGap grande para conectar líneas discontinuas (marcas de guión)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=15,
            minLineLength=15,
            maxLineGap=150,
        )

        # Zonas estrictas sin solapamiento
        DER_MIN = 0.54   # línea derecha: 54-96% del ancho de ROI
        IZQ_MAX = 0.44   # línea izquierda: 12-44%
        IZQ_MIN = 0.10   # excluye carril contrario muy a la izquierda

        left_xs, right_xs = [], []

        if lines is not None:
            for seg in lines:
                xa, ya, xb, yb_l = seg[0]
                if xb == xa:
                    continue
                slope = (yb_l - ya) / (xb - xa)

                if abs(slope) < 0.18:   # casi horizontal → ignorar
                    continue

                x_bot = xa + (roi_h - ya) / slope if slope != 0 else xa

                if slope > 0.18 and DER_MIN * roi_w < x_bot < 0.97 * roi_w:
                    right_xs.append(x_bot)
                elif slope < -0.18 and IZQ_MIN * roi_w < x_bot < IZQ_MAX * roi_w:
                    left_xs.append(x_bot)

        izq_det = bool(left_xs)
        der_det = bool(right_xs)

        desviacion_raw = 0.0
        confianza = 0.0

        if der_det and izq_det:
            centro = (np.median(left_xs) + np.median(right_xs)) / 2
            desviacion_raw = (centro / roi_w - 0.50) / 0.38
            confianza = 1.0
        elif der_det:
            pos_norm = np.median(right_xs) / roi_w
            desviacion_raw = (pos_norm - _POS_DER_CENTRADO) / _ESCALA_DER
            confianza = 0.85
        elif izq_det:
            pos_norm = np.median(left_xs) / roi_w
            desviacion_raw = -(0.32 - pos_norm) / _ESCALA_DER
            confianza = 0.5
        else:
            self._hist.append(0.0)
            return EstadoCarril(0.0, 0.0, False, False)

        desviacion_raw = float(np.clip(desviacion_raw, -1.0, 1.0))
        if abs(desviacion_raw) < self._zona_muerta:
            desviacion_raw = 0.0

        self._hist.append(desviacion_raw)
        suave = float(np.mean(self._hist))

        return EstadoCarril(
            desviacion=suave,
            confianza=confianza,
            linea_izq_detectada=izq_det,
            linea_der_detectada=der_det,
        )
