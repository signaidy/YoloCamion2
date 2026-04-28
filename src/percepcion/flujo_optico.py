"""Estimador de flujo optico denso restringido a un ROI.

Pure-vision (RNF-07): el flujo se calcula a partir de pixeles consecutivos
del frame, dentro de una region acotada para no comer FPS. Devuelve un mapa
(H, W, 2) de velocidades (u, v) en px por segundo.

Backend: cv2.calcOpticalFlowFarneback (denso, robusto, suficiente para
~30 FPS en GPU integrada). Si el presupuesto de FPS se rompe en la
Tarea 1.6, migrar a Lucas-Kanade disperso sobre Shi-Tomasi.
"""
from typing import Optional

import cv2
import numpy as np


# Parametros Farneback calibrados para ETS2 (asfalto/marcas/vehiculos):
#   pyr_scale=0.5, levels=3, winsize=21, iterations=3, poly_n=5, poly_sigma=1.1
_FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=21,
    iterations=3,
    poly_n=5,
    poly_sigma=1.1,
    flags=0,
)


class EstimadorFlujoOptico:
    """Calcula flujo denso entre frames consecutivos, opcionalmente solo en un ROI.

    Uso:
        est = EstimadorFlujoOptico(roi=(x1, y1, x2, y2))
        flujo = est.calcular(frame_bgr, timestamp=t)   # (H, W, 2) en px/s
    """

    def __init__(
        self,
        roi: Optional[tuple[int, int, int, int]] = None,
    ):
        self._roi = roi
        self._gris_anterior: Optional[np.ndarray] = None
        self._t_anterior: Optional[float] = None
        self._shape_full: Optional[tuple[int, int]] = None

    def _a_gris(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _recortar_roi(self, gris: np.ndarray) -> np.ndarray:
        if self._roi is None:
            return gris
        x1, y1, x2, y2 = self._roi
        h, w = gris.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        return gris[y1:y2, x1:x2]

    def calcular(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """Devuelve mapa (H, W, 2) de flujo en px/s. Cero fuera del ROI."""
        gris_full = self._a_gris(frame)
        if self._shape_full is None:
            self._shape_full = gris_full.shape[:2]

        gris_actual = self._recortar_roi(gris_full)

        if self._gris_anterior is None or self._t_anterior is None:
            self._gris_anterior = gris_actual.copy()
            self._t_anterior = timestamp
            return np.zeros((*self._shape_full, 2), dtype=np.float32)

        dt = timestamp - self._t_anterior
        if dt <= 0:
            self._gris_anterior = gris_actual.copy()
            self._t_anterior = timestamp
            return np.zeros((*self._shape_full, 2), dtype=np.float32)

        # Por seguridad si el ROI cambia (no esperado): reiniciar
        if gris_actual.shape != self._gris_anterior.shape:
            self._gris_anterior = gris_actual.copy()
            self._t_anterior = timestamp
            return np.zeros((*self._shape_full, 2), dtype=np.float32)

        # Farneback devuelve flujo en px/frame -> normalizar a px/s
        flujo_local = cv2.calcOpticalFlowFarneback(
            self._gris_anterior, gris_actual, None, **_FARNEBACK_PARAMS
        )
        flujo_local = flujo_local / dt

        # Reinsertar en el frame completo (cero fuera del ROI)
        flujo_full = np.zeros((*self._shape_full, 2), dtype=np.float32)
        if self._roi is not None:
            x1, y1, x2, y2 = self._roi
            h, w = self._shape_full
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            flujo_full[y1:y2, x1:x2] = flujo_local
        else:
            flujo_full = flujo_local

        self._gris_anterior = gris_actual.copy()
        self._t_anterior = timestamp
        return flujo_full

    def reset(self) -> None:
        self._gris_anterior = None
        self._t_anterior = None


def promediar_flujo_en_caja(
    flujo: np.ndarray,
    caja: tuple[int, int, int, int],
) -> tuple[float, float]:
    """Devuelve el flujo promedio (u, v) dentro de la caja, recortando a limites.

    Si la caja queda completamente fuera del frame, devuelve (0, 0).
    """
    h, w = flujo.shape[:2]
    x1, y1, x2, y2 = caja
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return (0.0, 0.0)

    region = flujo[y1:y2, x1:x2]
    u = float(region[..., 0].mean())
    v = float(region[..., 1].mean())
    return (u, v)
