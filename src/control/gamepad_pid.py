"""Controlador de gamepad 100% analogico con tres PIDs internos.

Capa 3 del refactor pure-vision. Reemplaza al ControladorGamepad pasthrough
sin suavizado y al ControladorTeclado digital. Todo va a vgamepad como
flotantes continuos para evitar oscilacion del remolque.

Tres PIDs:
  pid_volante     setpoint=0  medicion=desviacion_volante  -> stick_x
  pid_velocidad   setpoint=vel_obj  medicion=vel_actual    -> RT/LT split

Bypass de emergencia:
  Cuando setpoint.freno_objetivo >= 0.9, se ignoran los PIDs y se aplica
  LT directo (presion de freno proporcional). Los integradores se resetean
  para no acumular durante la frenada.

La velocidad actual se actualiza externamente con `actualizar_velocidad_actual()`
desde el bucle del piloto, que la deriva del flujo optico (Tarea 3.3, pure-vision).
NO consultamos telemetria interna (RNF-07).
"""
import logging
from dataclasses import dataclass

from src.control.base import Controlador
from src.control.pid import PIDController
from src.tipos import ComandoControl, SetpointControl

logger = logging.getLogger(__name__)


@dataclass
class ConfigPID:
    kp: float
    ki: float
    kd: float


# Defaults calibrados para ETS2 Volvo FH16 (ajustables en config/default.yaml)
# Calibracion final en pista en Fase 5.
_CFG_VOLANTE_DEFAULT  = ConfigPID(kp=0.65, ki=0.015, kd=0.12)
_CFG_VELOCIDAD_DEFAULT = ConfigPID(kp=0.65, ki=0.05, kd=0.04)

_FRENO_EMERGENCIA = 0.9   # umbral del setpoint para bypass total (cancela PID acel.)
_FRENO_DIRECTO_MIN = 0.05  # cualquier freno_objetivo>=esto se aplica como LT directo


class ControladorGamepadPID(Controlador):
    """Gamepad Xbox virtual (vgamepad) controlado via tres PIDs."""

    def __init__(
        self,
        cfg_volante: ConfigPID = _CFG_VOLANTE_DEFAULT,
        cfg_velocidad: ConfigPID = _CFG_VELOCIDAD_DEFAULT,
    ):
        self._pid_vol = PIDController(
            cfg_volante.kp, cfg_volante.ki, cfg_volante.kd, limite=1.0
        )
        self._pid_vel = PIDController(
            cfg_velocidad.kp, cfg_velocidad.ki, cfg_velocidad.kd, limite=1.0
        )
        self._gamepad = None
        self._t_ultimo: float | None = None
        self._vel_actual: float = 0.0   # 0..1 normalizado, fuente visual

    def iniciar(self) -> None:
        import vgamepad as vg
        self._gamepad = vg.VX360Gamepad()
        logger.info("ControladorGamepadPID: gamepad virtual iniciado")

    def actualizar_velocidad_actual(self, velocidad_norm: float) -> None:
        """Llamar cada frame con la velocidad propia visual (0..1)."""
        self._vel_actual = max(0.0, min(1.0, float(velocidad_norm)))

    # ── Compatibilidad con la API vieja (acepta ComandoControl) ─────────────
    def aplicar(self, sp_o_cmd) -> None:
        if isinstance(sp_o_cmd, ComandoControl):
            sp = SetpointControl(
                velocidad_objetivo_norm=sp_o_cmd.acelerador,
                freno_objetivo=sp_o_cmd.freno,
                desviacion_volante=sp_o_cmd.volante,
            )
        elif isinstance(sp_o_cmd, SetpointControl):
            sp = sp_o_cmd
        else:
            raise TypeError(
                f"aplicar() espera SetpointControl o ComandoControl, "
                f"recibio {type(sp_o_cmd).__name__}"
            )
        self._aplicar_setpoint(sp)

    def _aplicar_setpoint(self, sp: SetpointControl) -> None:
        if self._gamepad is None:
            raise RuntimeError("Llamar a iniciar() antes de aplicar()")

        import time as _t
        ahora = _t.monotonic()
        if self._t_ultimo is None:
            dt = 0.033   # asumimos ~30 FPS para el primer frame
        else:
            dt = max(0.001, ahora - self._t_ultimo)
        self._t_ultimo = ahora

        # ── Volante: PID hacia desviacion=0 ──────────────────────────────────
        # Pure Pursuit ahora devuelve el error posicional normalizado en el 
        # look-ahead point. Usamos el PID para llevar este error a cero
        # de forma suave (evitando volantazos u oscilaciones de lado a lado).
        stick_x = self._pid_vol.calcular(
            setpoint=0.0, medicion=sp.desviacion_volante, dt=dt
        )
        
        self._gamepad.left_joystick_float(
            x_value_float=float(stick_x), y_value_float=0.0
        )

        # ── Velocidad / Frenado ─────────────────────────────────────────────
        # Throttle OPEN-LOOP (feedforward del setpoint). Razón: la velocidad
        # propia visual via flujo óptico LK no está calibrada todavía y reporta
        # ~0.4-0.7 incluso con el camión parado (ruido de fondo, sway de cabina,
        # textura del capó). Eso engañaba al PID a creer que ya estaba a
        # velocidad y a no aplicar throttle. Hasta calibrar el flujo óptico,
        # mapeamos directamente velocidad_objetivo_norm -> RT.
        # El freno del FSM se aplica directo: si freno_objetivo >= umbral,
        # se anula el RT y se aplica LT proporcional.
        if sp.freno_objetivo >= _FRENO_DIRECTO_MIN:
            rt_aplicado = 0
            lt_aplicado = int(min(1.0, sp.freno_objetivo) * 255)
            self._pid_vel.reset()
        else:
            rt_aplicado = int(min(1.0, max(0.0, sp.velocidad_objetivo_norm)) * 255)
            lt_aplicado = 0

        self._gamepad.right_trigger(value=rt_aplicado)
        self._gamepad.left_trigger(value=lt_aplicado)
        self._ultimo_rt = rt_aplicado
        self._ultimo_lt = lt_aplicado
        self._ultimo_stick = float(stick_x)

        self._gamepad.update()

    @property
    def ultimo_comando_aplicado(self) -> tuple[int, int, float]:
        """(rt, lt, stick_x) del ultimo aplicar(); util para debug-piloto."""
        return self._ultimo_rt, self._ultimo_lt, self._ultimo_stick

    def liberar(self) -> None:
        if self._gamepad is None:
            return
        self._gamepad.right_trigger(value=0)
        self._gamepad.left_trigger(value=0)
        self._gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self._gamepad.update()
        self._pid_vol.reset()
        self._pid_vel.reset()
        logger.info("ControladorGamepadPID: ejes liberados")

    def cerrar(self) -> None:
        self.liberar()
