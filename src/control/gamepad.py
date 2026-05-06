import logging

from src.tipos import ComandoControl
from src.control.base import Controlador

logger = logging.getLogger(__name__)


class ControladorGamepad(Controlador):
    """Emula un gamepad Xbox 360 virtual usando vgamepad + ViGEmBus.

    RT = acelerador, LT = freno, stick_x = volante.
    """

    def __init__(self):
        self._gamepad = None

    def iniciar(self) -> None:
        import vgamepad as vg
        self._gamepad = vg.VX360Gamepad()
        logger.info("Gamepad virtual inicializado")

    def aplicar(self, cmd: ComandoControl) -> None:
        if self._gamepad is None:
            raise RuntimeError("Llama a iniciar() antes de aplicar()")

        self._gamepad.right_trigger(value=int(cmd.acelerador * 255))
        self._gamepad.left_trigger(value=int(cmd.freno * 255))
        self._gamepad.left_joystick_float(x_value_float=cmd.volante, y_value_float=0.0)
        self._gamepad.update()

    def liberar(self) -> None:
        if self._gamepad is not None:
            self._gamepad.right_trigger(value=0)
            self._gamepad.left_trigger(value=0)
            self._gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            self._gamepad.update()
            logger.info("Gamepad virtual liberado")

    def cerrar(self) -> None:
        self.liberar()
