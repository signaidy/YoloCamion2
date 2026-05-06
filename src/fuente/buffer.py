"""Buffer de captura en hilo separado.

Captura frames en background mientras el hilo principal corre YOLO.
Así mss no bloquea la inferencia y el FPS sube significativamente.
"""
import threading
import time
from typing import Optional

from src.tipos import Cuadro
from src.fuente.base import FuenteCuadros


class FuenteConBuffer(FuenteCuadros):
    """Envuelve cualquier FuenteCuadros y captura en un hilo dedicado."""

    def __init__(self, fuente: FuenteCuadros):
        self._fuente = fuente
        self._ultimo: Optional[Cuadro] = None
        self._lock = threading.Lock()
        self._hilo: Optional[threading.Thread] = None
        self._activa = False

    def iniciar(self) -> None:
        self._fuente.iniciar()
        self._activa = True
        self._hilo = threading.Thread(target=self._loop, daemon=True, name="captura-buf")
        self._hilo.start()

    def _loop(self) -> None:
        while self._activa:
            cuadro = self._fuente.siguiente()
            if cuadro is not None:
                with self._lock:
                    self._ultimo = cuadro
            else:
                time.sleep(0.001)

    def siguiente(self) -> Optional[Cuadro]:
        with self._lock:
            return self._ultimo

    def cerrar(self) -> None:
        self._activa = False
        self._fuente.cerrar()

    @property
    def esta_activa(self) -> bool:
        return self._activa and self._fuente.esta_activa
