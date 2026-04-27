import time
from typing import Optional

import cv2
import numpy as np

from src.tipos import Cuadro
from src.fuente.base import FuenteCuadros


def _escalar_con_recorte(frame: np.ndarray, objetivo: tuple[int, int]) -> np.ndarray:
    """Escala manteniendo el aspect ratio del destino recortando el centro.

    Si la pantalla es 16:10 (2560x1600) y el destino es 16:9 (1920x1080),
    recorta las bandas superiores/inferiores para no distorsionar la imagen.
    """
    w_dst, h_dst = objetivo
    h_src, w_src = frame.shape[:2]

    ratio_dst = w_dst / h_dst
    ratio_src = w_src / h_src

    if abs(ratio_src - ratio_dst) < 0.01:
        # Misma relación de aspecto — escalar directo
        return cv2.resize(frame, (w_dst, h_dst), interpolation=cv2.INTER_LINEAR)

    if ratio_src > ratio_dst:
        # Fuente más ancha → recortar lados
        h_new = h_src
        w_new = int(h_src * ratio_dst)
        x0 = (w_src - w_new) // 2
        recortado = frame[:, x0:x0 + w_new]
    else:
        # Fuente más alta → recortar arriba/abajo (caso 2560x1600 → 16:9)
        w_new = w_src
        h_new = int(w_src / ratio_dst)
        y0 = (h_src - h_new) // 2
        recortado = frame[y0:y0 + h_new, :]

    return cv2.resize(recortado, (w_dst, h_dst), interpolation=cv2.INTER_LINEAR)


class FuentePantalla(FuenteCuadros):
    """Captura cuadros en tiempo real.

    Intenta dxcam primero (más rápido, usa DXGI). Si falla (común en laptops
    con GPU dual), cae automáticamente a mss (más compatible, ~30-60 FPS).

    Si el juego corre a una resolución diferente a las ROI calibradas (1920x1080),
    usa escalar_a=(1920, 1080) para redimensionar cada frame automáticamente.
    """

    def __init__(
        self,
        monitor: int = 0,
        region: Optional[tuple] = None,
        escalar_a: Optional[tuple[int, int]] = None,
    ):
        self._monitor = monitor
        self._region = region
        self._escalar_a = escalar_a
        self._camara = None
        self._mss = None          # backend alternativo
        self._usar_mss = False
        self._indice = 0
        self._t_inicio = 0.0
        self._activa = False

    def iniciar(self) -> None:
        self._indice = 0
        self._t_inicio = time.monotonic()

        # Intentar dxcam
        try:
            import dxcam
            self._camara = dxcam.create(device_idx=self._monitor, region=self._region)
            self._camara.start(target_fps=60, video_mode=True)
            self._usar_mss = False
            self._activa = True
            import logging
            logging.getLogger(__name__).info("Captura: backend dxcam activo")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "dxcam no disponible (%s) — usando mss como fallback", e
            )
            self._camara = None
            self._iniciar_mss()

    def _iniciar_mss(self) -> None:
        import mss as mss_lib
        self._mss = mss_lib.MSS()
        monitors = self._mss.monitors

        # monitors[0] = todos combinados; monitors[1..] = monitores individuales
        # Buscamos el monitor primario (is_primary=True); si no está marcado, usamos
        # el de mayor área como heurística.
        mon = None
        for m in monitors[1:]:
            if m.get("is_primary", False):
                mon = m
                break
        if mon is None:
            mon = max(monitors[1:], key=lambda m: m["width"] * m["height"])

        import logging
        logging.getLogger(__name__).info(
            "mss: usando monitor primario %dx%d en (%d,%d)",
            mon["width"], mon["height"], mon["left"], mon["top"]
        )

        if self._region is not None:
            l, t, r, b = self._region
            self._mss_region = {"left": l, "top": t, "width": r - l, "height": b - t}
        else:
            self._mss_region = mon

        self._usar_mss = True
        self._activa = True
        import logging
        logging.getLogger(__name__).info(
            "Captura: backend mss activo — región %s", self._mss_region
        )

    def siguiente(self) -> Optional[Cuadro]:
        if not self._activa:
            return None

        if self._usar_mss:
            frame_bgr = self._capturar_mss()
        else:
            frame_bgr = self._capturar_dxcam()

        if frame_bgr is None:
            return None

        if self._escalar_a is not None:
            frame_bgr = _escalar_con_recorte(frame_bgr, self._escalar_a)

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

    def _capturar_dxcam(self) -> Optional[np.ndarray]:
        frame = self._camara.get_latest_frame()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def _capturar_mss(self) -> Optional[np.ndarray]:
        captura = self._mss.grab(self._mss_region)
        # mss devuelve BGRA
        return cv2.cvtColor(np.array(captura), cv2.COLOR_BGRA2BGR)

    def cerrar(self) -> None:
        self._activa = False
        if self._camara is not None:
            try:
                self._camara.stop()
            except Exception:
                pass
            self._camara = None
        if self._mss is not None:
            self._mss.close()
            self._mss = None

    @property
    def esta_activa(self) -> bool:
        return self._activa
