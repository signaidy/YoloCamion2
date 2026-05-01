"""Tests del ControladorGamepadPID con vgamepad mockeado.

El controlador combina:
  - Setpoint del FSM (velocidad_objetivo_norm, freno_objetivo, desviacion_volante)
  - Velocidad propia visual estimada externamente (Tarea 3.3)

Volante:
  - desviacion_volante se aplica como comando de stick: - izquierda, + derecha
  - bypass directo de freno cuando freno_objetivo >= 0.9 (emergencia)
"""
import time
from unittest.mock import MagicMock, patch

import pytest

from src.tipos import SetpointControl


@pytest.fixture
def mock_gamepad():
    """Patch vgamepad.VX360Gamepad y devuelve la instancia mock + el ctrl."""
    with patch.dict("sys.modules", {"vgamepad": MagicMock()}):
        import sys
        vg_mod = sys.modules["vgamepad"]
        gp = MagicMock()
        vg_mod.VX360Gamepad = MagicMock(return_value=gp)

        # Importar despues de patchear para que el modulo agarre el mock
        from src.control.gamepad_pid import ControladorGamepadPID
        ctrl = ControladorGamepadPID()
        ctrl.iniciar()
        yield ctrl, gp


def _sp(vel=0.0, freno=0.0, vol=0.0) -> SetpointControl:
    return SetpointControl(
        velocidad_objetivo_norm=vel,
        freno_objetivo=freno,
        desviacion_volante=vol,
    )


def test_iniciar_crea_gamepad_virtual(mock_gamepad):
    ctrl, gp = mock_gamepad
    assert ctrl._gamepad is gp


def test_liberar_pone_todos_los_ejes_a_cero(mock_gamepad):
    ctrl, gp = mock_gamepad
    ctrl.liberar()
    gp.right_trigger.assert_called_with(value=0)
    gp.left_trigger.assert_called_with(value=0)
    gp.left_joystick_float.assert_called_with(x_value_float=0.0, y_value_float=0.0)
    gp.update.assert_called()


def test_freno_emergencia_bypassa_pid_y_aplica_lt_maximo(mock_gamepad):
    ctrl, gp = mock_gamepad
    # Sembrar integrales no-cero para verificar que se resetean
    ctrl._pid_vel._integral = 5.0
    ctrl.actualizar_velocidad_actual(0.5)
    ctrl.aplicar(_sp(freno=1.0))

    # LT debe estar al maximo o casi
    llamadas_lt = [c.kwargs.get("value", 0) for c in gp.left_trigger.call_args_list]
    assert max(llamadas_lt) >= 240
    # RT debe estar en cero
    llamadas_rt = [c.kwargs.get("value", 0) for c in gp.right_trigger.call_args_list]
    assert llamadas_rt[-1] == 0
    # PID de velocidad reseteado tras emergencia
    assert ctrl._pid_vel._integral == pytest.approx(0.0)


def test_freno_en_standstill_no_activa_reversa(mock_gamepad):
    ctrl, gp = mock_gamepad
    ctrl.actualizar_velocidad_actual(0.0)
    ctrl.aplicar(_sp(freno=1.0))

    llamadas_lt = [c.kwargs.get("value", 0) for c in gp.left_trigger.call_args_list]
    llamadas_rt = [c.kwargs.get("value", 0) for c in gp.right_trigger.call_args_list]
    assert llamadas_lt[-1] == 0
    assert llamadas_rt[-1] == 0


def test_velocidad_objetivo_alta_y_actual_baja_dispara_throttle(mock_gamepad):
    ctrl, gp = mock_gamepad
    ctrl.actualizar_velocidad_actual(0.0)        # detenido
    # Aplicar varias veces para que el PID acumule
    for _ in range(5):
        ctrl.aplicar(_sp(vel=0.8))
        time.sleep(0.005)
    # En algun momento RT debio activarse
    rt_max = max(c.kwargs.get("value", 0) for c in gp.right_trigger.call_args_list)
    assert rt_max > 0


def test_velocidad_objetivo_cero_y_actual_alta_rueda_sin_reversa(mock_gamepad):
    """Sin freno explicito, exceso de velocidad debe soltar acelerador sin LT."""
    ctrl, gp = mock_gamepad
    ctrl.actualizar_velocidad_actual(0.8)        # vamos rapido
    for _ in range(5):
        ctrl.aplicar(_sp(vel=0.0))
        time.sleep(0.005)
    lt_max = max(c.kwargs.get("value", 0) for c in gp.left_trigger.call_args_list)
    rt_max = max(c.kwargs.get("value", 0) for c in gp.right_trigger.call_args_list)
    assert lt_max == 0
    assert rt_max == 0


def test_desviacion_positiva_gira_stick_a_la_derecha(mock_gamepad):
    ctrl, gp = mock_gamepad
    ctrl.aplicar(_sp(vol=0.5))
    xs = [c.kwargs.get("x_value_float", 0.0)
          for c in gp.left_joystick_float.call_args_list]
    assert max(xs) > 0


def test_desviacion_negativa_gira_stick_a_la_izquierda(mock_gamepad):
    ctrl, gp = mock_gamepad
    ctrl.aplicar(_sp(vol=-0.5))
    xs = [c.kwargs.get("x_value_float", 0.0)
          for c in gp.left_joystick_float.call_args_list]
    assert min(xs) < 0


def test_desviacion_pequena_se_aplica_proporcional(mock_gamepad):
    ctrl, gp = mock_gamepad
    ctrl.aplicar(_sp(vol=0.06))
    xs = [c.kwargs.get("x_value_float", 0.0)
          for c in gp.left_joystick_float.call_args_list]
    assert xs[-1] == pytest.approx(0.06)


def test_setpoint_neutro_mantiene_stick_centrado(mock_gamepad):
    ctrl, gp = mock_gamepad
    ctrl.aplicar(_sp(vol=0.0))
    xs = [c.kwargs.get("x_value_float", 0.0)
          for c in gp.left_joystick_float.call_args_list]
    assert all(abs(x) < 0.05 for x in xs)


def test_cerrar_libera_los_ejes(mock_gamepad):
    ctrl, gp = mock_gamepad
    ctrl.cerrar()
    # cerrar() llama liberar()
    gp.right_trigger.assert_called_with(value=0)
    gp.left_trigger.assert_called_with(value=0)


def test_aplicar_sin_iniciar_lanza_error():
    """Si nadie llamo iniciar(), aplicar() debe fallar explicitamente."""
    from src.control.gamepad_pid import ControladorGamepadPID
    ctrl = ControladorGamepadPID()
    with pytest.raises(RuntimeError):
        ctrl.aplicar(_sp())
