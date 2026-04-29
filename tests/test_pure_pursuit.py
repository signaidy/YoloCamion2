# tests/test_pure_pursuit.py
import numpy as np
import pytest

from src.control.pure_pursuit import PurePursuitVisual


def test_carril_perdido_devuelve_True_y_error_cero_inicial():
    """Máscara vacía sin memoria previa → (0.0, True)."""
    pp = PurePursuitVisual()
    mascara = np.zeros((480, 640), dtype=np.uint8)
    error, perdido = pp.calcular_giro(mascara)
    assert perdido is True
    assert error == pytest.approx(0.0)


def test_decaimiento_memoria_tras_perder_carril():
    """Tras detectar error, máscara vacía devuelve error_anterior * 0.85."""
    pp = PurePursuitVisual()
    m1 = np.zeros((480, 640), dtype=np.uint8)
    m1[200:480, 50:250] = 1          # área a la izquierda → error positivo
    error_1, perdido_1 = pp.calcular_giro(m1)
    assert not perdido_1
    assert error_1 > 0

    m2 = np.zeros((480, 640), dtype=np.uint8)
    error_2, perdido_2 = pp.calcular_giro(m2)
    assert perdido_2 is True
    assert error_2 == pytest.approx(error_1 * 0.85, rel=0.05)


def test_via_ancha_bias_derecho_genera_error_negativo():
    """Vía de ancho completo (dos carriles): bias sitúa objetivo en carril derecho."""
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    m[100:480, 0:640] = 1            # toda la vía visible
    error, perdido = pp.calcular_giro(m)
    assert not perdido
    # Target cae a la derecha del centro → dx < 0 → error < 0 → PID gira derecha
    assert error < -0.10


def test_area_solo_izquierda_error_positivo():
    """Área manejable solo a la izquierda → camión debe girar izquierda (error > 0)."""
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    m[100:480, 0:200] = 1
    error, perdido = pp.calcular_giro(m)
    assert not perdido
    assert error > 0.10


def test_area_solo_derecha_error_negativo():
    """Área manejable solo a la derecha → camión debe girar derecha (error < 0)."""
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    m[100:480, 440:640] = 1
    error, perdido = pp.calcular_giro(m)
    assert not perdido
    assert error < -0.10


def test_error_acotado_entre_menos1_y_1():
    """El error normalizado nunca sale del rango [-1, 1]."""
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    m[100:480, 0:10] = 1             # franja extrema → dx muy grande
    error, _ = pp.calcular_giro(m)
    assert -1.0 <= error <= 1.0


def test_ultimo_punto_debug_none_cuando_carril_perdido():
    """Sin carril visible, ultimo_punto_debug debe ser None."""
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    pp.calcular_giro(m)
    assert pp.ultimo_punto_debug is None


def test_ultimo_punto_debug_dentro_de_la_imagen():
    """Con carril visible, ultimo_punto_debug cae dentro de los límites de la imagen."""
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    m[100:480, 200:500] = 1
    pp.calcular_giro(m)
    assert pp.ultimo_punto_debug is not None
    x, y = pp.ultimo_punto_debug
    assert 0 <= x < 640
    assert 0 <= y < 480
