"""Detector de carriles para ETS2 — Volvo FH16 en primera persona.

Estrategia específica para ETS2 (conducción por la derecha):
- Línea derecha (borde de carril): aparece en el 55-95% derecho del frame
- Línea izquierda (centro de vía): aparece en el 10-45% izquierdo del frame
- Líneas del sentido contrario: aparecen muy a la izquierda (<15%) → excluidas
- Referencia primaria: línea derecha (más estable y sin ambigüedad)

Devuelve desviacion en [-1, +1]:
  negativo = camión está muy a la derecha → girar izquierda (A)
  cero     = centrado
  positivo = camión está muy a la izquierda → girar derecha (D)
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
# (como fracción del ancho de la ROI). En ETS2 con Volvo FH16 es ~0.68.
_POS_LINEA_DER_CENTRADO = 0.68
# Margen máximo de desviación normalizada antes de saturar
_ESCALA_DER = 0.30


class DetectorCarriles:
    def __init__(
        self,
        zona_roi: tuple[float, float, float, float] = (0.18, 0.56, 0.82, 0.84),
        suavizado: int = 7,
        zona_muerta: float = 0.07,
    ):
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

        # Detección de marcas blancas y amarillas
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_blanca   = cv2.inRange(hsv, (0,   0,  155), (180,  50, 255))
        mask_amarilla = cv2.inRange(hsv, (14,  65,  75), (38,  255, 255))
        mask = cv2.bitwise_or(mask_blanca, mask_amarilla)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)

        edges = cv2.Canny(mask, 30, 100)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=20, minLineLength=18, maxLineGap=90
        )

        # Zonas de búsqueda estrictas (sin solapamiento para evitar ambigüedad)
        # Línea derecha: 55-95% del ancho → borde derecho de nuestro carril
        # Línea izquierda: 15-44% del ancho → centro de vía (excluye carril contrario)
        ZONA_DER_MIN = 0.55
        ZONA_IZQ_MAX = 0.44
        ZONA_IZQ_MIN = 0.12   # excluye líneas del carril contrario (muy a la izq)

        left_xs, right_xs = [], []

        if lines is not None:
            for seg in lines:
                xa, ya, xb, yb_l = seg[0]
                if xb == xa:
                    continue
                slope = (yb_l - ya) / (xb - xa)

                # Descartar líneas casi horizontales (señalización lateral, sombras)
                if abs(slope) < 0.20:
                    continue

                # Proyectar al fondo de la ROI para clasificar por posición
                x_bot = xa + (roi_h - ya) / slope if slope != 0 else xa

                if slope > 0.20 and ZONA_DER_MIN * roi_w < x_bot < 0.97 * roi_w:
                    right_xs.append(x_bot)
                elif slope < -0.20 and ZONA_IZQ_MIN * roi_w < x_bot < ZONA_IZQ_MAX * roi_w:
                    left_xs.append(x_bot)

        izq_det = bool(left_xs)
        der_det = bool(right_xs)

        desviacion_raw = 0.0
        confianza = 0.0

        if der_det and izq_det:
            # Ambas líneas: usar el centro del carril
            centro = (np.median(left_xs) + np.median(right_xs)) / 2
            desviacion_raw = (centro / roi_w - 0.50) / 0.40
            confianza = 1.0

        elif der_det:
            # Solo línea derecha (caso más común y confiable)
            # Si está a la derecha del punto esperado → camión muy a la izquierda → desviar +
            # Si está a la izquierda del punto esperado → camión muy a la derecha → desviar -
            pos_norm = np.median(right_xs) / roi_w
            desviacion_raw = (pos_norm - _POS_LINEA_DER_CENTRADO) / _ESCALA_DER
            confianza = 0.85

        elif izq_det:
            # Solo línea izquierda — menos confiable
            pos_norm = np.median(left_xs) / roi_w
            # Posición esperada de línea izq cuando centrado ≈ 0.32
            desviacion_raw = -(_POS_LINEA_DER_CENTRADO - 1.0 + pos_norm) / _ESCALA_DER
            confianza = 0.5

        else:
            self._historial.append(0.0)
            return EstadoCarril(0.0, 0.0, False, False)

        desviacion_raw = float(np.clip(desviacion_raw, -1.0, 1.0))

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
