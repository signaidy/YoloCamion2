"""Estimadores de flujo optico restringidos a un ROI.

Pure-vision (RNF-07). Dos backends:

  EstimadorFlujoOptico   Farneback denso. Mapa (H,W,2) en px/s.
                         Costo ~134 ms/frame a 1920x1080 -- usar solo cuando
                         el pipeline soporta <= 5 FPS (analisis offline).

  EstimadorFlujoOpticoLK Lucas-Kanade disperso sobre puntos Shi-Tomasi.
                         Costo despreciable (~5 ms). Devuelve un mapa
                         disperso (H,W,2) interpolado a partir de los puntos
                         rastreados; es el backend RECOMENDADO para runtime.

Benchmark de referencia (commit con benchmark_fps.py):
  Farneback denso:  4.9 FPS / 199 ms
  Lucas-Kanade:    15.8 FPS /  59 ms
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


_SHI_TOMASI = dict(maxCorners=120, qualityLevel=0.01, minDistance=10, blockSize=7)
_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


class EstimadorFlujoOpticoLK:
    """Flujo optico disperso (Lucas-Kanade) sobre puntos Shi-Tomasi del ROI.

    Mantiene la misma API de salida que EstimadorFlujoOptico: mapa
    (H_full, W_full, 2) en px/s. Donde no hay puntos rastreados, el mapa es
    cero. Para densificar localmente alrededor de un bounding box, usar
    `promediar_flujo_en_caja(flujo, caja)`: la funcion ignora celdas en cero
    y promedia solo donde hay senal real.
    """

    def __init__(
        self,
        roi: Optional[tuple[int, int, int, int]] = None,
        radio_pintar: int = 6,
    ):
        self._roi = roi
        self._radio = radio_pintar
        self._gris_anterior: Optional[np.ndarray] = None
        self._t_anterior: Optional[float] = None
        self._puntos: Optional[np.ndarray] = None
        self._shape_full: Optional[tuple[int, int]] = None

    def _a_gris(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _recortar(self, gris: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        if self._roi is None:
            return gris, (0, 0)
        x1, y1, x2, y2 = self._roi
        h, w = gris.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        return gris[y1:y2, x1:x2], (x1, y1)

    def _re_sembrar(self, recorte: np.ndarray) -> None:
        self._puntos = cv2.goodFeaturesToTrack(recorte, mask=None, **_SHI_TOMASI)

    def _flujo_vacio(self) -> np.ndarray:
        return np.zeros((*self._shape_full, 2), dtype=np.float32)

    def calcular(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        gris_full = self._a_gris(frame)
        if self._shape_full is None:
            self._shape_full = gris_full.shape[:2]

        recorte, (ox, oy) = self._recortar(gris_full)

        if self._gris_anterior is None or self._puntos is None:
            self._gris_anterior = recorte.copy()
            self._t_anterior = timestamp
            self._re_sembrar(recorte)
            return self._flujo_vacio()

        if recorte.shape != self._gris_anterior.shape:
            self._gris_anterior = recorte.copy()
            self._t_anterior = timestamp
            self._re_sembrar(recorte)
            return self._flujo_vacio()

        dt = timestamp - self._t_anterior
        if dt <= 0:
            self._gris_anterior = recorte.copy()
            self._t_anterior = timestamp
            return self._flujo_vacio()

        nuevos, st, _ = cv2.calcOpticalFlowPyrLK(
            self._gris_anterior, recorte, self._puntos, None, **_LK_PARAMS
        )

        flujo_full = self._flujo_vacio()
        if nuevos is not None and st is not None:
            mascara = st.flatten() == 1
            antes = self._puntos.reshape(-1, 2)[mascara]
            despues = nuevos.reshape(-1, 2)[mascara]
            for (xa, ya), (xd, yd) in zip(antes, despues):
                u = (xd - xa) / dt
                v = (yd - ya) / dt
                px = int(round(xd)) + ox
                py = int(round(yd)) + oy
                # Pintar disco pequeno con el vector para densificar
                x1 = max(0, px - self._radio)
                y1 = max(0, py - self._radio)
                x2 = min(self._shape_full[1], px + self._radio + 1)
                y2 = min(self._shape_full[0], py + self._radio + 1)
                flujo_full[y1:y2, x1:x2, 0] = u
                flujo_full[y1:y2, x1:x2, 1] = v

            # Re-sembrar si quedan pocos puntos validos
            if mascara.sum() < 20:
                self._re_sembrar(recorte)
            else:
                self._puntos = nuevos[mascara].reshape(-1, 1, 2)
        else:
            self._re_sembrar(recorte)

        self._gris_anterior = recorte.copy()
        self._t_anterior = timestamp
        return flujo_full

    def reset(self) -> None:
        self._gris_anterior = None
        self._t_anterior = None
        self._puntos = None


def promediar_flujo_en_caja(
    flujo: np.ndarray,
    caja: tuple[int, int, int, int],
    ignorar_ceros: bool = True,
) -> tuple[float, float]:
    """Devuelve el flujo promedio (u, v) dentro de la caja, recortando a limites.

    Si `ignorar_ceros` (default), promedia solo celdas con (u,v) != (0,0).
    Esto es importante para flujo disperso (LK), donde la mayoria del mapa
    es cero y un mean() trivial subestimaria la magnitud real.
    """
    h, w = flujo.shape[:2]
    x1, y1, x2, y2 = caja
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return (0.0, 0.0)

    region = flujo[y1:y2, x1:x2]
    if ignorar_ceros:
        mascara = (region[..., 0] != 0) | (region[..., 1] != 0)
        if mascara.sum() == 0:
            return (0.0, 0.0)
        u = float(region[..., 0][mascara].mean())
        v = float(region[..., 1][mascara].mean())
    else:
        u = float(region[..., 0].mean())
        v = float(region[..., 1].mean())
    return (u, v)
