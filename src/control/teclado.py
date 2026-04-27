import logging
import time

import pydirectinput

from src.tipos import ComandoControl
from src.control.base import Controlador

logger = logging.getLogger(__name__)

# Con teclado el control es binario (tecla pulsada o no).
# MANTENER envía acelerador=0.3 — subimos el umbral a 0.55 para que
# MANTENER NO presione W (el camión cruza por inercia/freno motor).
# Solo ACELERAR (0.6) y GIRAR (0.2 con volante 0.5) presionan W.
_UMBRAL_AVANZAR = 0.55
_UMBRAL_FRENAR  = 0.15   # cualquier freno real activa S
_UMBRAL_GIRAR   = 0.25


class ControladorTeclado(Controlador):
    """Emula controles binarizados con pydirectinput (W/A/S/D).

    Diseñado para ejecutarse como Administrador (requerido cuando ETS2
    corre con privilegios elevados).
    """

    def __init__(self):
        self._teclas_activas: set[str] = set()

    def aplicar(self, cmd: ComandoControl) -> None:
        deseadas: set[str] = set()

        if cmd.acelerador >= _UMBRAL_AVANZAR:
            deseadas.add("w")
        if cmd.freno >= _UMBRAL_FRENAR:
            deseadas.add("s")
        if cmd.volante <= -_UMBRAL_GIRAR:
            deseadas.add("a")
        if cmd.volante >= _UMBRAL_GIRAR:
            deseadas.add("d")

        # No presionar W y S al mismo tiempo (conflicto)
        if "w" in deseadas and "s" in deseadas:
            deseadas.discard("w")

        for tecla in self._teclas_activas - deseadas:
            pydirectinput.keyUp(tecla)
        for tecla in deseadas - self._teclas_activas:
            pydirectinput.keyDown(tecla)

        self._teclas_activas = deseadas

    def liberar(self) -> None:
        for tecla in list(self._teclas_activas):
            pydirectinput.keyUp(tecla)
        self._teclas_activas.clear()
        logger.info("Teclado liberado")

    def cerrar(self) -> None:
        self.liberar()
