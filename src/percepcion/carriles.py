"""Detector de carriles por visión — mantiene el camión centrado en su carril.

Usa HoughLinesP sobre una máscara de marcas blancas/amarillas en la zona
frontal inferior del frame (excluye el tablero y los espejos laterales).

Devuelve desviacion en [-1, +1]:
  -1 = completamente a la izquierda del carril
   0 = centrado
  +1 = completamente a la derecha del carril
"""
from dataclasses import dataclass
from collections import deque
from typing import Optional

import cv2
import numpy as np


@dataclass
class EstadoCarril:
    desviacion: float          # -1 a +1, 0 = centrado
    confianza: float           # 0-1, qué tan seguros estamos de la detección
    linea_izq_detectada: bool
    linea_der_detectada: bool


class DetectorCarriles:
    """Detecta marcas de carril y calcula la desviación lateral del vehículo.

    ROI optimizada para cámara primera persona del Volvo FH16 en ETS2:
    - Excluye el tablero del camión (parte inferior)
    - Excluye los espejos laterales
    - Se enfoca en la zona donde las líneas del carril son más claras
    """

    def __init__(
        self,
        zona_roi: tuple[float, float, float, float] = (0.18, 0.56, 0.82, 0.84),
        suavizado: int = 6,
        zona_muerta: float = 0.07,
    ):
        # zona_roi: (x_izq, y_top, x_der, y_bot) como fracción del frame
        self._roi = zona_roi
        self._historial: deque[float] = deque(maxlen=suavizado)
        self._zona_muerta = zona_muerta

    def detectar(self, frame: np.ndarray) -> EstadoCarril:
        h, w = frame.shape[:2]
        xi, yt, xd, yb = self._roi
        x1, y1 = int(w * xi), int(h * yt)
        x2, y2 = int(w * xd), int(h * yb)
        roi = frame[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        # Máscara de marcas blancas y amarillas en HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_blanca   = cv2.inRange(hsv, (0,   0,  160), (180,  45, 255))
        mask_amarilla = cv2.inRange(hsv, (15,  70,  80), (38,  255, 255))
        mask = cv2.bitwise_or(mask_blanca, mask_amarilla)

        # Dilatar ligeramente para conectar líneas discontinuas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)

        edges = cv2.Canny(mask, 30, 100)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=20, minLineLength=20, maxLineGap=80
        )

        left_xs, right_xs = [], []

        if lines is not None:
            for seg in lines:
                x_a, y_a, x_b, y_b = seg[0]
                if x_b == x_a:
                    continue
                slope = (y_b - y_a) / (x_b - x_a)
                if abs(slope) < 0.25:        # líneas casi horizontales → ignorar
                    continue
                # Proyectar hasta la parte baja de la ROI
                x_bot = x_a + (roi_h - y_a) / slope if slope != 0 else x_a

                if slope < 0 and x_bot < roi_w * 0.6:   # línea izquierda
                    left_xs.append(x_bot)
                elif slope > 0 and x_bot > roi_w * 0.4: # línea derecha
                    right_xs.append(x_bot)

        izq_det = bool(left_xs)
        der_det = bool(right_xs)

        if izq_det and der_det:
            centro_carril = (np.median(left_xs) + np.median(right_xs)) / 2
            confianza = 1.0
        elif izq_det:
            centro_carril = np.median(left_xs) + roi_w * 0.38
            confianza = 0.6
        elif der_det:
            centro_carril = np.median(right_xs) - roi_w * 0.38
            confianza = 0.6
        else:
            self._historial.append(0.0)
            return EstadoCarril(0.0, 0.0, False, False)

        desviacion_raw = (centro_carril - roi_w / 2) / (roi_w / 2)
        desviacion_raw = float(np.clip(desviacion_raw, -1.0, 1.0))

        # Zona muerta: pequeñas desviaciones no generan corrección
        if abs(desviacion_raw) < self._zona_muerta:
            desviacion_raw = 0.0

        self._historial.append(desviacion_raw)
        desviacion_suave = float(np.mean(self._historial))

        return EstadoCarril(
            desviacion=desviacion_suave,
            confianza=confianza,
            linea_izq_detectada=izq_det,
            linea_der_detectada=der_det,
        )
