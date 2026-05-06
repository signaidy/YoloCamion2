import logging

from src.tipos import ComandoControl
from src.control.base import Controlador

logger = logging.getLogger(__name__)


class ControladorNulo(Controlador):
    """No envía controles reales — solo registra. Útil para probar sobre video."""

    def aplicar(self, cmd: ComandoControl) -> None:
        logger.debug(
            "ControladorNulo | acel=%.2f freno=%.2f volante=%.2f",
            cmd.acelerador, cmd.freno, cmd.volante,
        )

    def liberar(self) -> None:
        logger.info("ControladorNulo | liberar() — sin efecto real")

    def cerrar(self) -> None:
        pass
