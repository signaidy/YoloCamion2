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


def test_via_ancha_sin_sesgo_error_cercano_a_cero():
    """Vía simétrica (ancho completo): sin sesgo, centroide ≈ centro → error ≈ 0."""
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    m[100:480, 0:640] = 1            # área verde simétrica
    error, perdido = pp.calcular_giro(m)
    assert not perdido
    # Sin sesgo, centroide = 319 (x_camion=320) → error ≈ 0
    assert abs(error) < 0.05


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


def test_ll_mask_simetrico_error_cero():
    """Líneas equidistantes del centro del frame → centro_carril = x_camion → error ≈ 0."""
    pp = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[100:480, 0:640] = 1
    ll = np.zeros((480, 640), dtype=np.uint8)
    ll[300:480, 215:225] = 1   # borde izquierdo  (max = 224)
    ll[300:480, 415:425] = 1   # borde derecho    (min = 415)
    # centro_carril = (224 + 415) / 2 = 319.5 ≈ x_camion=320 → error ≈ 0
    error, perdido = pp.calcular_giro(da, ll)
    assert not perdido
    assert abs(error) < 0.05


def test_ll_mask_corrige_bias_da_mask():
    """
    da_mask asimétrica (centroide ≈ 420 → error negativo).
    ll_mask dice centro ≈ 320 → error ≈ 0.
    """
    pp_con_ll = PurePursuitVisual()
    pp_sin_ll = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[200:480, 200:640] = 1   # centroide ≈ 419 (a la derecha del frame)
    ll = np.zeros((480, 640), dtype=np.uint8)
    ll[300:480, 215:225] = 1   # borde izq ≈ 220
    ll[300:480, 415:425] = 1   # borde der ≈ 420 → centro ≈ 320

    error_con, _ = pp_con_ll.calcular_giro(da, ll)
    error_sin, _ = pp_sin_ll.calcular_giro(da)

    assert abs(error_con) < 0.05    # ll_mask: camión centrado
    assert error_sin < -0.10        # da_mask sola: sesgo negativo persistente


def test_ll_mask_vacio_usa_da_mask():
    """ll_mask sin píxeles → resultado idéntico a no pasar ll_mask."""
    pp1 = PurePursuitVisual()
    pp2 = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[100:480, 50:300] = 1    # área asimétrica
    ll = np.zeros((480, 640), dtype=np.uint8)   # vacía

    e1, _ = pp1.calcular_giro(da)
    e2, _ = pp2.calcular_giro(da, ll)
    assert e1 == pytest.approx(e2, rel=0.01)


def test_ll_mask_pocos_pixeles_no_activa_ll():
    """Menos de _MIN_LL_PIXELES (15) por lado → cae a da_mask, misma señal."""
    pp_ll = PurePursuitVisual()
    pp_da = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[100:480, 0:640] = 1     # da simétrica
    ll = np.zeros((480, 640), dtype=np.uint8)
    ll[350, 100] = 1            # 1 pixel izq — insuficiente (< 15)
    ll[350, 540] = 1            # 1 pixel der — insuficiente (< 15)

    e_ll, _ = pp_ll.calcular_giro(da, ll)
    e_da, _ = pp_da.calcular_giro(da)
    assert e_ll == pytest.approx(e_da)
