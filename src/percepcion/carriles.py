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


# Posición esperada de la línea derecha cuando el camión está centrado.
# Calibrado para Volvo FH16 en ETS2: la línea derecha aparece en ~0.72 del
# ancho de ROI cuando el camión está centrado (no 0.68 como la cámara genérica).
_POS_DER_CENTRADO = 0.72
_ESCALA_DER = 0.28


class DetectorCarriles:
    """Detecta marcas de carril usando brillo sobre asfalto.

    Soporta dos modos:
    - Día (modo_noche=False): ROI media, umbral ~110, zona amplia
    - Noche (modo_noche=True): ROI baja (zona iluminada por faros), umbral ~80

    Detección automática de noche si el brillo medio del frame < 60.
    """

    def __init__(
        self,
        zona_roi: tuple[float, float, float, float] = (0.15, 0.52, 0.85, 0.83),
        zona_roi_noche: tuple[float, float, float, float] = (0.20, 0.72, 0.80, 0.90),
        suavizado: int = 4,
        zona_muerta: float = 0.07,
        modo_noche: bool | None = None,   # None = automático
        sesgo_lateral: float = 0.0,       # offset de calibración (+= vira dcha, -= vira izq)
    ):
        self._roi_dia = zona_roi
        self._roi_noche = zona_roi_noche
        self._hist: deque[float] = deque(maxlen=suavizado)
        self._zona_muerta = zona_muerta
        self._modo_noche_forzado = modo_noche
        self._sesgo_lateral = sesgo_lateral

    def _es_noche(self, frame: np.ndarray) -> bool:
        if self._modo_noche_forzado is not None:
            return self._modo_noche_forzado
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) < 55.0

    def detectar(self, frame: np.ndarray) -> EstadoCarril:
        h, w = frame.shape[:2]
        noche = self._es_noche(frame)
        roi_coords = self._roi_noche if noche else self._roi_dia
        umbral = 75 if noche else 110
        xi, yt, xd, yb = roi_coords
        x1, y1 = int(w * xi), int(h * yt)
        x2, y2 = int(w * xd), int(h * yb)
        roi = frame[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        # Convertir a escala de grises y aplicar CLAHE para equalizar
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        _, mask_bright = cv2.threshold(gray, umbral, 255, cv2.THRESH_BINARY)

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

                if abs(slope) < 0.12:   # casi horizontal → ignorar (0.12 captura líneas en curvas)
                    continue

                x_bot = xa + (roi_h - ya) / slope if slope != 0 else xa

                if slope > 0.12 and DER_MIN * roi_w < x_bot < 0.97 * roi_w:
                    right_xs.append(x_bot)
                elif slope < -0.12 and IZQ_MIN * roi_w < x_bot < IZQ_MAX * roi_w:
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

        desviacion_raw -= self._sesgo_lateral
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

    def debug_frame(self, frame: np.ndarray) -> np.ndarray:
        """Retorna una copia del frame con la ROI y las líneas detectadas anotadas.

        Verde = línea izquierda, Rojo = línea derecha, Cyan = centro de carril,
        Amarillo = rectángulo ROI. Útil para calibrar _POS_DER_CENTRADO y sesgo_lateral.
        """
        h, w = frame.shape[:2]
        noche = self._es_noche(frame)
        roi_coords = self._roi_noche if noche else self._roi_dia
        umbral = 75 if noche else 110
        xi, yt, xd, yb = roi_coords
        x1, y1 = int(w * xi), int(h * yt)
        x2, y2 = int(w * xd), int(h * yb)
        roi = frame[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, mask_bright = cv2.threshold(gray, umbral, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        mask_bright = cv2.morphologyEx(mask_bright, cv2.MORPH_OPEN, kernel)
        edges = cv2.Canny(mask_bright, 30, 90)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15,
                                minLineLength=15, maxLineGap=150)

        DER_MIN = 0.54
        IZQ_MAX = 0.44
        IZQ_MIN = 0.10
        left_xs, right_xs = [], []

        out = frame.copy()
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)  # ROI = amarillo

        if lines is not None:
            for seg in lines:
                xa, ya, xb_s, yb_s = seg[0]
                if xb_s == xa:
                    continue
                slope = (yb_s - ya) / (xb_s - xa)
                if abs(slope) < 0.12:
                    continue
                x_bot = xa + (roi_h - ya) / slope if slope != 0 else xa
                abs_xa, abs_ya = xa + x1, ya + y1
                abs_xb, abs_yb = xb_s + x1, yb_s + y1
                if slope > 0.12 and DER_MIN * roi_w < x_bot < 0.97 * roi_w:
                    right_xs.append(x_bot)
                    cv2.line(out, (abs_xa, abs_ya), (abs_xb, abs_yb), (0, 0, 255), 2)
                elif slope < -0.12 and IZQ_MIN * roi_w < x_bot < IZQ_MAX * roi_w:
                    left_xs.append(x_bot)
                    cv2.line(out, (abs_xa, abs_ya), (abs_xb, abs_yb), (0, 255, 0), 2)
                else:
                    cv2.line(out, (abs_xa, abs_ya), (abs_xb, abs_yb), (128, 128, 128), 1)

        if right_xs:
            rx = int(np.median(right_xs)) + x1
            cv2.circle(out, (rx, y2 - 5), 8, (0, 0, 255), -1)
            pos_norm = np.median(right_xs) / roi_w
            label = f"DER pos={pos_norm:.2f} desv={((pos_norm - _POS_DER_CENTRADO) / _ESCALA_DER):.2f}"
            cv2.putText(out, label, (rx - 60, y2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if left_xs:
            lx = int(np.median(left_xs)) + x1
            cv2.circle(out, (lx, y2 - 5), 8, (0, 255, 0), -1)

        if right_xs and left_xs:
            cx_roi = (np.median(left_xs) + np.median(right_xs)) / 2
            cx = int(cx_roi) + x1
            cv2.circle(out, (cx, y2 - 5), 6, (255, 255, 0), -1)
            desv = (cx_roi / roi_w - 0.50) / 0.38
            cv2.putText(out, f"centro desv={desv:.2f}", (cx - 40, y2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        modo = "NOCHE" if noche else "DIA"
        cv2.putText(out, f"modo={modo} izq={len(left_xs)} der={len(right_xs)}",
                    (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out
