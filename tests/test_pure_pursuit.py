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


def test_via_ancha_con_sesgo_derecho_error_negativo():
    """Vía simétrica con _BIAS_FRAC=0.30: centroide desplazado a la derecha → error < 0 (girar derecha).

    _BIAS_FRAC=0.30 modela conducción europea (carril derecho): en vías bidireccionales
    donde da_mask cubre ambos carriles, el centroide de la mitad derecha del área verde
    corresponde al carril del conductor, no a la línea central.
    """
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    m[100:480, 0:640] = 1            # área verde simétrica
    error, perdido = pp.calcular_giro(m)
    assert not perdido
    # Con BIAS_FRAC=0.30, centroide queda a la derecha de x_camion → error negativo
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


def test_ll_mask_simetrico_aplica_offset_de_cabina():
    """Lineas equidistantes del centro visual dan error casi cero."""
    pp = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[100:480, 0:640] = 1
    ll = np.zeros((480, 640), dtype=np.uint8)
    ll[300:480, 215:225] = 1
    ll[300:480, 415:425] = 1
    # centro_carril visual ~= x_camion; el offset de cabina pide margen a la derecha.
    error, perdido = pp.calcular_giro(da, ll)
    assert not perdido
    assert abs(error) < 0.05


def test_ll_mask_corrige_bias_da_mask():
    """
    da_mask asimétrica (centroide ≈ 420 → error negativo).
    ll_mask dice centro visual ≈ 320 → error con offset de cabina.
    """
    pp_con_ll = PurePursuitVisual()
    pp_sin_ll = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[200:480, 200:640] = 1   # centroide ≈ 419 (a la derecha del frame)
    ll = np.zeros((480, 640), dtype=np.uint8)
    ll[300:480, 215:225] = 1
    ll[300:480, 415:425] = 1

    error_con, _ = pp_con_ll.calcular_giro(da, ll)
    error_sin, _ = pp_sin_ll.calcular_giro(da)

    assert abs(error_con) < 0.05
    assert abs(error_sin) < 0.05


def test_ll_mask_multicarril_ignora_linea_exterior():
    """Una linea exterior extra no debe mover el centro del carril activo."""
    pp = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[100:480, 0:640] = 1
    ll = np.zeros((480, 640), dtype=np.uint8)
    ll[250:480, 25:35] = 1     # linea exterior de otro carril
    ll[250:480, 215:225] = 1
    ll[250:480, 415:425] = 1

    error, perdido = pp.calcular_giro(da, ll)

    assert not perdido
    assert abs(error) < 0.05


def test_ll_mask_multicarril_elige_par_adyacente_central():
    """Con varias lineas visibles, usa el par de carril mas cercano al eje visual."""
    pp = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[100:480, 0:640] = 1
    ll = np.zeros((480, 640), dtype=np.uint8)
    for x1, x2 in [(55, 65), (215, 225), (415, 425), (575, 585)]:
        ll[300:480, x1:x2] = 1

    error, perdido = pp.calcular_giro(da, ll)

    assert not perdido
    assert abs(error) < 0.05


def test_ll_mask_empate_prefiere_par_a_la_derecha():
    """En empate de carriles visibles, evita elegir el carril pegado a la mediana."""
    pp = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[100:480, 0:640] = 1
    ll = np.zeros((480, 640), dtype=np.uint8)
    for x1, x2 in [(95, 105), (295, 305), (335, 345), (535, 545)]:
        ll[300:480, x1:x2] = 1

    centro = pp._centro_desde_ll_pares(ll, [340, 360, 380], 320, 640)

    assert centro == pytest.approx(439.5)


def test_ll_mask_no_sigue_punto_previo_en_otro_carril():
    """Un punto previo malo no debe cambiar el carril usado por ll_mask."""
    pp = PurePursuitVisual()
    pp._ultimo_punto = (80, 360)
    da = np.zeros((480, 640), dtype=np.uint8)
    da[100:480, 0:640] = 1
    ll = np.zeros((480, 640), dtype=np.uint8)
    ll[250:480, 25:35] = 1
    ll[250:480, 215:225] = 1
    ll[250:480, 415:425] = 1

    error, perdido = pp.calcular_giro(da, ll)

    assert not perdido
    assert abs(error) < 0.05


def test_ll_mask_rechaza_salto_a_carril_adyacente():
    """Con carril bloqueado, una deteccion lejana no debe cambiar de carril."""
    pp = PurePursuitVisual()
    pp._x_ancla_carril = 320.0
    pp._ultimo_error = 0.12
    da = np.zeros((480, 640), dtype=np.uint8)
    ll = np.zeros((480, 640), dtype=np.uint8)
    ll[250:480, 25:35] = 1
    ll[250:480, 135:145] = 1

    error, perdido = pp.calcular_giro(da, ll)

    assert perdido
    assert error == pytest.approx(0.12 * pp._DECAY)
    assert pp._x_ancla_carril == pytest.approx(320.0)


def test_validacion_de_carril_satura_salto_sin_mover_ancla():
    """Un centro de carril muy lejano se limita, pero no arrastra el ancla."""
    pp = PurePursuitVisual()
    pp._x_ancla_carril = 320.0

    centro = pp._validar_y_actualizar_ancla(120.0, 640)

    assert centro == pytest.approx(320.0 - 640 * pp._MAX_SALTO_CARRIL_FRAC)
    assert pp._x_ancla_carril == pytest.approx(320.0)


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
