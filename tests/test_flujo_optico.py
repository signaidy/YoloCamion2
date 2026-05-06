"""Tests del EstimadorFlujoOptico.

Pure-vision (RNF-07): el flujo se calcula en el frame, restringido al ROI
frontal por presupuesto de FPS, y devuelve un campo (u,v) en px/s.
"""
import numpy as np
import pytest

from src.percepcion.flujo_optico import (
    EstimadorFlujoOptico,
    EstimadorFlujoOpticoLK,
    promediar_flujo_en_caja,
)


def _frame_negro(h: int = 360, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _frame_con_cuadrado(
    cx: int, cy: int, lado: int = 60, h: int = 360, w: int = 640
) -> np.ndarray:
    """Frame BGR con un cuadrado blanco en (cx,cy)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    x1 = max(0, cx - lado // 2)
    y1 = max(0, cy - lado // 2)
    x2 = min(w, cx + lado // 2)
    y2 = min(h, cy + lado // 2)
    img[y1:y2, x1:x2] = 255
    return img


def test_primer_frame_devuelve_flujo_cero():
    """Sin frame anterior no hay flujo medible."""
    est = EstimadorFlujoOptico()
    flujo = est.calcular(_frame_negro(), timestamp=0.0)
    assert flujo.shape[2] == 2          # (H, W, 2): canales u y v
    assert np.allclose(flujo, 0.0)


def test_frame_repetido_da_flujo_cero():
    est = EstimadorFlujoOptico()
    f = _frame_con_cuadrado(320, 180)
    est.calcular(f, timestamp=0.0)
    flujo = est.calcular(f, timestamp=0.1)
    assert np.allclose(flujo, 0.0, atol=0.5)


def test_objeto_que_se_mueve_a_la_derecha_genera_flujo_positivo_en_x():
    est = EstimadorFlujoOptico()
    f1 = _frame_con_cuadrado(300, 180)
    f2 = _frame_con_cuadrado(340, 180)        # +40 px en x
    est.calcular(f1, timestamp=0.0)
    flujo = est.calcular(f2, timestamp=0.5)   # 0.5s -> 80 px/s

    # Promedio dentro de la caja del objeto en el frame 2
    u_medio, v_medio = promediar_flujo_en_caja(flujo, (310, 150, 370, 210))
    assert u_medio > 5.0, f"u_medio={u_medio} debio ser positivo y > umbral"
    assert abs(v_medio) < 20.0


def test_objeto_que_baja_genera_flujo_positivo_en_y():
    est = EstimadorFlujoOptico()
    f1 = _frame_con_cuadrado(320, 150)
    f2 = _frame_con_cuadrado(320, 200)
    est.calcular(f1, timestamp=0.0)
    flujo = est.calcular(f2, timestamp=0.5)

    u_medio, v_medio = promediar_flujo_en_caja(flujo, (290, 170, 350, 230))
    assert v_medio > 5.0
    assert abs(u_medio) < 20.0


def test_promediar_flujo_en_caja_fuera_de_rango_devuelve_cero():
    flujo = np.zeros((100, 100, 2), dtype=np.float32)
    u, v = promediar_flujo_en_caja(flujo, (200, 200, 300, 300))   # fuera
    assert u == pytest.approx(0.0)
    assert v == pytest.approx(0.0)


def test_promediar_flujo_recorta_caja_a_limites_del_frame():
    flujo = np.ones((100, 100, 2), dtype=np.float32) * 3.0
    u, v = promediar_flujo_en_caja(flujo, (-50, -50, 50, 50))
    assert u == pytest.approx(3.0)
    assert v == pytest.approx(3.0)


def test_estimador_con_roi_solo_calcula_dentro_del_roi():
    """Si pasamos un ROI=(x1,y1,x2,y2), todo lo de afuera queda en cero."""
    est = EstimadorFlujoOptico(roi=(100, 100, 540, 260))
    f1 = _frame_con_cuadrado(50, 50)        # cuadrado FUERA del ROI
    f2 = _frame_con_cuadrado(80, 50)
    est.calcular(f1, timestamp=0.0)
    flujo = est.calcular(f2, timestamp=0.5)

    # Fuera del ROI: cero garantizado
    u, v = promediar_flujo_en_caja(flujo, (40, 40, 90, 90))
    assert u == pytest.approx(0.0)
    assert v == pytest.approx(0.0)


def test_estimador_calcula_flujo_dentro_del_roi():
    est = EstimadorFlujoOptico(roi=(100, 100, 540, 260))
    f1 = _frame_con_cuadrado(300, 180)        # dentro del ROI
    f2 = _frame_con_cuadrado(340, 180)
    est.calcular(f1, timestamp=0.0)
    flujo = est.calcular(f2, timestamp=0.5)

    u, v = promediar_flujo_en_caja(flujo, (310, 150, 370, 210))
    assert u > 5.0


def test_reset_limpia_frame_anterior():
    est = EstimadorFlujoOptico()
    f1 = _frame_con_cuadrado(300, 180)
    f2 = _frame_con_cuadrado(340, 180)
    est.calcular(f1, timestamp=0.0)
    est.reset()
    flujo = est.calcular(f2, timestamp=0.5)
    # Sin frame previo -> flujo cero
    assert np.allclose(flujo, 0.0)


# ── Tests del backend Lucas-Kanade disperso ─────────────────────────────────


def _frame_con_textura(h: int = 360, w: int = 640) -> np.ndarray:
    """Frame con textura aleatoria para que Shi-Tomasi encuentre puntos."""
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return img


def test_lk_primer_frame_devuelve_flujo_cero():
    est = EstimadorFlujoOpticoLK()
    flujo = est.calcular(_frame_con_textura(), timestamp=0.0)
    assert flujo.shape[2] == 2
    assert np.allclose(flujo, 0.0)


def test_lk_desplazamiento_horizontal_genera_u_positivo():
    """Frame que se desplaza a la derecha debe producir u>0 en los puntos."""
    est = EstimadorFlujoOpticoLK()
    base = _frame_con_textura()
    # f2 = base desplazada 12 px a la derecha (np.roll)
    f2 = np.roll(base, 12, axis=1)
    est.calcular(base, timestamp=0.0)
    flujo = est.calcular(f2, timestamp=0.5)        # 0.5s -> 24 px/s

    # Tomar el promedio de celdas no-cero (LK es disperso)
    u, v = promediar_flujo_en_caja(flujo, (0, 0, flujo.shape[1], flujo.shape[0]))
    assert u > 5.0, f"u_medio={u} debio ser positivo"


def test_lk_con_roi_ignora_fuera_del_roi():
    est = EstimadorFlujoOpticoLK(roi=(100, 100, 540, 260))
    base = _frame_con_textura()
    f2 = np.roll(base, 8, axis=1)
    est.calcular(base, timestamp=0.0)
    flujo = est.calcular(f2, timestamp=0.5)

    # Fuera del ROI: cero
    u, v = promediar_flujo_en_caja(flujo, (10, 10, 90, 90))
    assert u == pytest.approx(0.0)
    assert v == pytest.approx(0.0)


def test_lk_reset_re_siembra_puntos():
    est = EstimadorFlujoOpticoLK()
    base = _frame_con_textura()
    est.calcular(base, timestamp=0.0)
    est.reset()
    flujo = est.calcular(base, timestamp=0.5)
    assert np.allclose(flujo, 0.0)


def test_promediar_ignora_ceros_de_un_mapa_disperso():
    """Mapa con un solo punto no-cero: el promedio debe ser ese punto, no /N."""
    flujo = np.zeros((100, 100, 2), dtype=np.float32)
    flujo[50, 50] = (10.0, 5.0)
    u, v = promediar_flujo_en_caja(flujo, (0, 0, 100, 100), ignorar_ceros=True)
    assert u == pytest.approx(10.0)
    assert v == pytest.approx(5.0)


def test_promediar_sin_ignorar_ceros_promedia_todo_el_area():
    flujo = np.zeros((100, 100, 2), dtype=np.float32)
    flujo[50, 50] = (10.0, 5.0)
    u, v = promediar_flujo_en_caja(flujo, (0, 0, 100, 100), ignorar_ceros=False)
    # 10/(100*100) = 0.001
    assert u == pytest.approx(0.001, abs=1e-4)
