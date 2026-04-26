import time
from typing import Optional

from src.tipos import Cuadro
from src.fuente.base import FuenteCuadros


class FuentePantalla(FuenteCuadros):
    """Captura cuadros en tiempo real usando DXcam (DXGI Desktop Duplication).

    Requiere Windows + GPU con soporte DirectX. Alcanza ~240 FPS en capturas
    de pantalla completa sin compresión.
    """

    def __init__(self, monitor: int = 0, region: Optional[tuple] = None):
        self._monitor = monitor
        self._region = region  # (left, top, right, bottom) o None para pantalla completa
        self._camara = None
        self._indice = 0
        self._t_inicio = 0.0
        self._activa = False

    def iniciar(self) -> None:
        import dxcam
        self._camara = dxcam.create(device_idx=self._monitor, region=self._region)
        self._camara.start(target_fps=60, video_mode=True)
        self._indice = 0
        self._t_inicio = time.monotonic()
        self._activa = True

    def siguiente(self) -> Optional[Cuadro]:
        if not self._activa or self._camara is None:
            return None

        frame = self._camara.get_latest_frame()
        if frame is None:
            return None

        import cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ahora = time.monotonic()
        elapsed = ahora - self._t_inicio
        fps = self._indice / elapsed if elapsed > 0 else 0.0
        self._indice += 1

        return Cuadro(
            imagen=frame_bgr,
            timestamp=ahora,
            indice=self._indice,
            fps_instantaneo=fps,
        )

    def cerrar(self) -> None:
        self._activa = False
        if self._camara is not None:
            self._camara.stop()
            self._camara = None

    @property
    def esta_activa(self) -> bool:
        return self._activa
