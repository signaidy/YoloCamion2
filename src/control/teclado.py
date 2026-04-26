import logging

import pydirectinput

from src.tipos import ComandoControl
from src.control.base import Controlador

logger = logging.getLogger(__name__)

_UMBRAL_AVANZAR = 0.2
_UMBRAL_FRENAR  = 0.2
_UMBRAL_GIRAR   = 0.3


class ControladorTeclado(Controlador):
    """Emula controles binarizados con pydirectinput (W/A/S/D).

    Fallback sin driver adicional cuando vgamepad no está disponible.
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
