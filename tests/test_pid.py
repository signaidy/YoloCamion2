"""Tests del PIDController generico.

Anti-windup por clamping del termino integral cuando la salida satura.
Limites simetricos en [-limite, +limite].
"""
import pytest

from src.control.pid import PIDController


def test_respuesta_proporcional_pura():
    """Ki=0 Kd=0 -> salida = Kp * error."""
    pid = PIDController(kp=2.0, ki=0.0, kd=0.0, limite=10.0)
    salida = pid.calcular(setpoint=5.0, medicion=3.0, dt=0.1)
    assert salida == pytest.approx(4.0)   # 2.0 * (5 - 3)


def test_salida_satura_en_limite_positivo():
    pid = PIDController(kp=100.0, ki=0.0, kd=0.0, limite=1.0)
    salida = pid.calcular(setpoint=10.0, medicion=0.0, dt=0.1)
    assert salida == pytest.approx(1.0)


def test_salida_satura_en_limite_negativo():
    pid = PIDController(kp=100.0, ki=0.0, kd=0.0, limite=1.0)
    salida = pid.calcular(setpoint=0.0, medicion=10.0, dt=0.1)
    assert salida == pytest.approx(-1.0)


def test_anti_windup_acota_la_integral():
    """Con setpoint inalcanzable, el integral no debe crecer indefinidamente."""
    pid = PIDController(kp=1.0, ki=10.0, kd=0.0, limite=1.0)
    for _ in range(200):
        pid.calcular(setpoint=10.0, medicion=0.0, dt=0.1)
    # Integral acotada por limite/Ki = 0.1
    assert abs(pid._integral) <= 1.0 / 10.0 + 1e-6


def test_dt_no_positivo_devuelve_cero_sin_actualizar():
    pid = PIDController(kp=1.0, ki=1.0, kd=1.0, limite=10.0)
    pid.calcular(setpoint=5.0, medicion=0.0, dt=0.1)
    integral_antes = pid._integral
    err_antes = pid._error_anterior
    salida = pid.calcular(setpoint=5.0, medicion=0.0, dt=0.0)
    assert salida == pytest.approx(0.0)
    assert pid._integral == integral_antes
    assert pid._error_anterior == err_antes


def test_reset_limpia_integral_y_error_anterior():
    pid = PIDController(kp=1.0, ki=5.0, kd=1.0, limite=10.0)
    pid.calcular(setpoint=5.0, medicion=0.0, dt=0.1)
    assert pid._integral != 0.0
    pid.reset()
    assert pid._integral == pytest.approx(0.0)
    assert pid._error_anterior == pytest.approx(0.0)


def test_termino_derivativo_amortigua_cambio_brusco_de_error():
    """Con Kd>0 y un cambio brusco de error, la salida del primer paso
    debe ser distinta de la del segundo (donde el error ya es estable)."""
    pid = PIDController(kp=1.0, ki=0.0, kd=2.0, limite=100.0)
    s1 = pid.calcular(setpoint=10.0, medicion=0.0, dt=0.1)
    s2 = pid.calcular(setpoint=10.0, medicion=0.0, dt=0.1)
    assert s1 != pytest.approx(s2)
    # s1 = Kp*err + Kd*err/dt = 10 + 2*10/0.1 = 210 -> sat. al limite 100
    # s2 = Kp*err + Kd*0/dt = 10
    assert s1 > s2


def test_setpoint_alcanzado_da_salida_cero():
    pid = PIDController(kp=2.0, ki=0.0, kd=0.0, limite=1.0)
    salida = pid.calcular(setpoint=5.0, medicion=5.0, dt=0.1)
    assert salida == pytest.approx(0.0)


def test_limite_se_aplica_a_la_suma_p_i_d():
    """Aunque P+I+D excedan el limite, la salida queda saturada."""
    pid = PIDController(kp=10.0, ki=10.0, kd=10.0, limite=2.0)
    for _ in range(20):
        s = pid.calcular(setpoint=100.0, medicion=0.0, dt=0.1)
        assert -2.0 <= s <= 2.0
