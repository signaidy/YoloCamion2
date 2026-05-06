import pytest

from src.control.carril_steering_policy import comando_direccion_por_carril


def test_zona_muerta_devuelve_cero():
    assert comando_direccion_por_carril(0.004, "ll") == 0.0


def test_signo_positivo_gira_hacia_izquierda_en_gamepad():
    cmd = comando_direccion_por_carril(0.10, "ll")
    assert cmd < 0.0


def test_fuente_ll_recibe_mas_autoridad_que_da_para_offset_persistente():
    cmd_ll = comando_direccion_por_carril(0.14, "ll", velocidad_kmh=25)
    cmd_da = comando_direccion_por_carril(0.14, "da", velocidad_kmh=25)

    assert abs(cmd_ll) > abs(cmd_da)
    assert cmd_ll < 0.0
    assert cmd_da < 0.0


def test_fuente_da_sigue_limitada():
    cmd = comando_direccion_por_carril(0.30, "da", velocidad_kmh=30)
    assert cmd == -0.18


def test_fuente_ll_permita_mayor_comando_sin_saturar_pronto():
    cmd = comando_direccion_por_carril(0.16, "ll", velocidad_kmh=25)
    assert cmd < -0.25


def test_baja_velocidad_amortigua_comando():
    cmd_lento = comando_direccion_por_carril(0.12, "ll", velocidad_kmh=8)
    cmd_rapido = comando_direccion_por_carril(0.12, "ll", velocidad_kmh=25)

    assert abs(cmd_lento) < abs(cmd_rapido)
    assert cmd_lento < 0.0


def test_fuente_da_a_baja_velocidad_tiene_cap_mas_conservador():
    cmd_lento = comando_direccion_por_carril(0.30, "da", velocidad_kmh=4)
    cmd_rapido = comando_direccion_por_carril(0.30, "da", velocidad_kmh=25)

    assert abs(cmd_lento) < abs(cmd_rapido)
    assert cmd_lento == pytest.approx(-0.13333333333333333)
