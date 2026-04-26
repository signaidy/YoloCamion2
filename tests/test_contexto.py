import numpy as np
import pytest

from src.tipos import Clase, EstadoSemaforo, Region, Seguimiento
from src.percepcion.contexto import AnalizadorContexto, _ROI_DEFAULT
from src.percepcion.semaforo import clasificar_semaforo


def _seg(clase, caja, area=10000, edad=5, track_id=1, confianza=0.9):
    return Seguimiento(clase=clase, caja=caja, confianza=confianza, area=area,
                       id_seguimiento=track_id, edad=edad)


# ── Tests de contexto ──────────────────────────────────────────────────────────

def test_frente_cercano_ocupado_cuando_vehiculo_grande_en_roi():
    ctx = AnalizadorContexto()
    roi = _ROI_DEFAULT[Region.FRENTE_CERCANO]
    cx, cy = (roi[0] + roi[2]) // 2, (roi[1] + roi[3]) // 2
    seg = _seg(Clase.VEHICULO, (cx - 50, cy - 50, cx + 50, cy + 50), area=10000)
    estado = ctx.analizar([seg])
    assert estado.frente_cercano_ocupado is True


def test_frente_no_ocupado_cuando_vehiculo_pequeño():
    ctx = AnalizadorContexto()
    roi = _ROI_DEFAULT[Region.FRENTE_CERCANO]
    cx, cy = (roi[0] + roi[2]) // 2, (roi[1] + roi[3]) // 2
    seg = _seg(Clase.VEHICULO, (cx - 10, cy - 10, cx + 10, cy + 10), area=400)
    estado = ctx.analizar([seg])
    assert estado.frente_cercano_ocupado is False


def test_peaton_en_riesgo_cuando_peaton_en_frente():
    ctx = AnalizadorContexto()
    roi = _ROI_DEFAULT[Region.FRENTE_CERCANO]
    cx, cy = (roi[0] + roi[2]) // 2, (roi[1] + roi[3]) // 2
    seg = _seg(Clase.PEATON, (cx - 20, cy - 40, cx + 20, cy + 40))
    estado = ctx.analizar([seg])
    assert estado.peaton_en_riesgo is True


def test_espejo_izq_requiere_edad_minima():
    ctx = AnalizadorContexto()
    roi = _ROI_DEFAULT[Region.ESPEJO_IZQ]
    cx, cy = (roi[0] + roi[2]) // 2, (roi[1] + roi[3]) // 2
    seg_joven = _seg(Clase.VEHICULO, (cx - 50, cy - 50, cx + 50, cy + 50), edad=1)
    estado = ctx.analizar([seg_joven])
    assert estado.espejo_izq_ocupado is False

    seg_maduro = _seg(Clase.VEHICULO, (cx - 50, cy - 50, cx + 50, cy + 50), edad=5)
    estado2 = ctx.analizar([seg_maduro])
    assert estado2.espejo_izq_ocupado is True


def test_senal_alto_detectada():
    ctx = AnalizadorContexto()
    roi = _ROI_DEFAULT[Region.FRENTE_CERCANO]
    cx, cy = (roi[0] + roi[2]) // 2, (roi[1] + roi[3]) // 2
    seg = _seg(Clase.SENAL_ALTO, (cx - 20, cy - 20, cx + 20, cy + 20))
    estado = ctx.analizar([seg])
    assert estado.senal_alto_cercana is True


def test_escena_vacia_da_defaults_seguros():
    ctx = AnalizadorContexto()
    estado = ctx.analizar([])
    assert estado.frente_cercano_ocupado is False
    assert estado.peaton_en_riesgo is False
    assert estado.semaforo_visible is None
    assert estado.vehiculos_totales == 0


# ── Tests de semáforo por color HSV ───────────────────────────────────────────

def _imagen_color_bgr(bgr: tuple, size: int = 60) -> np.ndarray:
    img = np.full((size, size, 3), bgr, dtype=np.uint8)
    return img


def test_semaforo_detecta_rojo():
    img = _imagen_color_bgr((0, 0, 200))  # rojo puro BGR
    estado = clasificar_semaforo(img, (0, 0, 60, 60))
    assert estado == EstadoSemaforo.ROJO


def test_semaforo_detecta_verde():
    img = _imagen_color_bgr((0, 200, 0))  # verde puro BGR
    estado = clasificar_semaforo(img, (0, 0, 60, 60))
    assert estado == EstadoSemaforo.VERDE


def test_semaforo_desconocido_en_imagen_gris():
    img = _imagen_color_bgr((128, 128, 128))
    estado = clasificar_semaforo(img, (0, 0, 60, 60))
    assert estado == EstadoSemaforo.DESCONOCIDO
