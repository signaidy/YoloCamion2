import time
from pathlib import Path
from typing import Optional

import cv2

from src.tipos import Cuadro
from src.fuente.base import FuenteCuadros


class FuenteVideo(FuenteCuadros):
    """Lee cuadros desde un archivo de video local (mp4, avi, etc.)."""

    def __init__(self, ruta: str | Path, loop: bool = False):
        self._ruta = str(ruta)
        self._loop = loop
        self._cap: Optional[cv2.VideoCapture] = None
        self._indice = 0
        self._t_inicio = 0.0
        self._activa = False

    def iniciar(self) -> None:
        self._cap = cv2.VideoCapture(self._ruta)
        if not self._cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {self._ruta}")
        self._indice = 0
        self._t_inicio = time.monotonic()
        self._activa = True

    def siguiente(self) -> Optional[Cuadro]:
        if not self._activa or self._cap is None:
            return None

        ok, frame = self._cap.read()
        if not ok:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok:
                    return None
            else:
                self._activa = False
                return None

        ahora = time.monotonic()
        elapsed = ahora - self._t_inicio
        fps = self._indice / elapsed if elapsed > 0 else 0.0
        self._indice += 1

        return Cuadro(
            imagen=frame,
            timestamp=ahora,
            indice=self._indice,
            fps_instantaneo=fps,
        )

    def cerrar(self) -> None:
        self._activa = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def esta_activa(self) -> bool:
        return self._activa
