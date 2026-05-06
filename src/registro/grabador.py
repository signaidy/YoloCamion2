"""Graba video MP4 con overlays de detecciones, acción actual y estado FSM."""
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.tipos import Accion, Seguimiento

logger = logging.getLogger(__name__)

_COLOR_CAJA = {
    "vehiculo":    (0, 165, 255),
    "motocicleta": (0, 165, 255),
    "peaton":      (0, 0, 255),
    "semaforo":    (255, 255, 0),
    "senal_alto":  (0, 255, 255),
    "desconocido": (128, 128, 128),
}
_COLOR_ACCION = {
    Accion.ALTO_TOTAL:    (0, 0, 255),
    Accion.FRENAR_FUERTE: (0, 0, 200),
    Accion.FRENAR_SUAVE:  (0, 165, 255),
    Accion.MANTENER:      (255, 255, 255),
    Accion.ACELERAR:      (0, 255, 0),
    Accion.REBASAR_IZQ:   (255, 0, 255),
    Accion.REBASAR_DER:   (255, 0, 200),
}


class GrabadorVideo:
    def __init__(self, ruta_base: str | Path, fps: float = 30.0):
        ruta_base = Path(ruta_base)
        ruta_base.mkdir(parents=True, exist_ok=True)
        nombre = f"grabacion_{int(time.time())}.mp4"
        self._ruta = ruta_base / nombre
        self._fps = fps
        self._writer: Optional[cv2.VideoWriter] = None

    def iniciar(self, ancho: int, alto: int) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._ruta), fourcc, self._fps, (ancho, alto))
        logger.info("Grabador iniciado: %s (%dx%d @ %.0f fps)", self._ruta, ancho, alto, self._fps)

    def escribir_frame(
        self,
        frame: np.ndarray,
        seguimientos: list[Seguimiento],
        accion: Accion,
        estado_fsm: str,
        fps_actual: float,
        regla: int,
    ) -> None:
        if self._writer is None:
            return

        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # Cajas de detección
        for seg in seguimientos:
            x1, y1, x2, y2 = seg.caja
            color = _COLOR_CAJA.get(seg.clase.value, (128, 128, 128))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            etiqueta = f"{seg.clase.value} #{seg.id_seguimiento} ({seg.confianza:.0%})"
            cv2.putText(canvas, etiqueta, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Overlay HUD
        color_accion = _COLOR_ACCION.get(accion, (255, 255, 255))
        cv2.rectangle(canvas, (0, 0), (420, 90), (0, 0, 0), -1)
        cv2.putText(canvas, f"Accion: {accion.value}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_accion, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Estado: {estado_fsm}  R{regla}", (8, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"FPS: {fps_actual:.1f}", (8, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1, cv2.LINE_AA)

        self._writer.write(canvas)

    def cerrar(self) -> None:
        if self._writer is not None:
            self._writer.release()
            logger.info("Grabador cerrado: %s", self._ruta)

    @property
    def ruta(self) -> Path:
        return self._ruta
