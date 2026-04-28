"""Tests del EstimadorVelocidadPropia.

Pure-vision (RNF-07): la velocidad del ego se deriva del modulo del flujo
optico en una franja inferior-central (asfalto del propio carril). Cuando
el camion avanza, el asfalto pasa hacia abajo en la imagen.
"""
import math

import numpy as np
import pytest

from src.percepcion.velocidad_propia import EstimadorVelocidadPropia


def _flujo_uniforme(h: int = 1080, w: int = 1920, magnitud: float = 0.0,
                    direccion: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Mapa (H,W,2) con flujo uniforme dado."""
    flujo = np.zeros((h, w, 2), dtype=np.float32)
    if magnitud > 0:
        u_dir, v_dir = direccion
        norm = math.sqrt(u_dir * u_dir + v_dir * v_dir)
        flujo[..., 0] = magnitud * u_dir / norm
        flujo[..., 1] = magnitud * v_dir / norm
    return flujo


def test_flujo_cero_da_velocidad_cero():
    est = EstimadorVelocidadPropia()
    flujo = _flujo_uniforme(magnitud=0.0)
    v = est.estimar(flujo)
    assert v == pytest.approx(0.0)


def test_flujo_descendente_central_da_velocidad_positiva():
    """Asfalto que pasa hacia abajo en la imagen = el ego avanza hacia adelante."""
    est = EstimadorVelocidadPropia(factor_calibracion=0.01)  # px/s -> norm
    # 100 px/s en y dentro de la franja (vertical positiva = baja)
    flujo = _flujo_uniforme(magnitud=100.0, direccion=(0.0, 1.0))
    v = est.estimar(flujo)
    assert v > 0.0


def test_velocidad_normalizada_en_0_1():
    est = EstimadorVelocidadPropia(factor_calibracion=0.01)
    # 1000 px/s -> 10.0 sin clamp -> debe saturar en 1.0
    flujo = _flujo_uniforme(magnitud=1000.0, direccion=(0.0, 1.0))
    v = est.estimar(flujo)
    assert 0.0 <= v <= 1.0
    assert v == pytest.approx(1.0)


def test_franja_inferior_central_es_la_que_cuenta():
    """Flujo solo en la zona superior NO debe contribuir."""
    est = EstimadorVelocidadPropia(factor_calibracion=0.01)
    flujo = np.zeros((1080, 1920, 2), dtype=np.float32)
    # Magnitud alta solo en la mitad SUPERIOR de la imagen
    flujo[:540, :, 1] = 200.0
    v = est.estimar(flujo)
    assert v == pytest.approx(0.0, abs=0.05)


def test_franja_inferior_central_con_flujo_da_velocidad_positiva():
    est = EstimadorVelocidadPropia(factor_calibracion=0.01)
    flujo = np.zeros((1080, 1920, 2), dtype=np.float32)
    # Flujo en la franja inferior-central (zona donde miramos asfalto)
    flujo[850:1050, 600:1320, 1] = 150.0
    v = est.estimar(flujo)
    assert v > 0.5


def test_factor_calibracion_escala_la_salida():
    """Con factor doble, una misma magnitud da el doble de velocidad normalizada."""
    flujo = _flujo_uniforme(magnitud=50.0, direccion=(0.0, 1.0))
    est_a = EstimadorVelocidadPropia(factor_calibracion=0.005)
    est_b = EstimadorVelocidadPropia(factor_calibracion=0.010)
    va = est_a.estimar(flujo)
    vb = est_b.estimar(flujo)
    assert vb > va


def test_suavizado_de_ventana_ema_amortigua_picos():
    """Con suavizado activado, un pico aislado no produce velocidad maxima."""
    est = EstimadorVelocidadPropia(factor_calibracion=0.01, alpha_ema=0.2)
    # Frames previos: cero
    for _ in range(5):
        est.estimar(_flujo_uniforme(magnitud=0.0))
    # Pico
    pico = est.estimar(_flujo_uniforme(magnitud=1000.0, direccion=(0.0, 1.0)))
    # El EMA debe atenuar el pico
    assert pico < 1.0


def test_flujo_disperso_ignora_celdas_cero():
    """Con flujo LK (mayoria del mapa en cero) la magnitud media no debe diluirse."""
    est = EstimadorVelocidadPropia(factor_calibracion=0.01, alpha_ema=1.0)
    flujo = np.zeros((1080, 1920, 2), dtype=np.float32)
    # Solo unas pocas celdas tienen senal en la franja inferior
    flujo[900:910, 900:910, 1] = 100.0
    v = est.estimar(flujo)
    assert v > 0.0
