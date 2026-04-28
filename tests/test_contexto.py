import math

import numpy as np
import pytest

from src.tipos import Clase, EstadoSemaforo, FisicaVisual, Region, Seguimiento
from src.percepcion.contexto import AnalizadorContexto, _ROI_DEFAULT
from src.percepcion.fisica import EstimadorFisicaVisual
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


# ── Tests TTC integrado en AnalizadorContexto (Fase 1.4) ───────────────────────


def test_escena_vacia_da_ttc_infinito_y_sin_critico():
    ctx = AnalizadorContexto()
    estado = ctx.analizar([])
    assert math.isinf(estado.ttc_minimo_frente_s)
    assert estado.vehiculo_critico_id is None


def test_vehiculo_que_se_acerca_en_frente_cercano_baja_el_ttc():
    """Bbox creciendo en frente_cercano debe poblar ttc_minimo y vehiculo_critico."""
    estimador = EstimadorFisicaVisual()
    ctx = AnalizadorContexto(estimador_fisica=estimador)
    roi = _ROI_DEFAULT[Region.FRENTE_CERCANO]
    cx, cy = (roi[0] + roi[2]) // 2, (roi[1] + roi[3]) // 2

    # Frame 1: vehiculo de 100x100
    seg1 = _seg(Clase.VEHICULO, (cx - 50, cy - 50, cx + 50, cy + 50),
                area=10000, track_id=7, edad=1)
    estado1 = ctx.analizar([seg1], timestamp=0.0)
    assert math.isinf(estado1.ttc_minimo_frente_s)  # primera muestra

    # Frame 2 (0.3s despues): vehiculo de 130x130 (acercandose)
    seg2 = _seg(Clase.VEHICULO, (cx - 65, cy - 65, cx + 65, cy + 65),
                area=16900, track_id=7, edad=2)
    estado2 = ctx.analizar([seg2], timestamp=0.3)

    assert estado2.ttc_minimo_frente_s < math.inf
    assert estado2.ttc_minimo_frente_s > 0
    assert estado2.vehiculo_critico_id == 7


def test_vehiculo_fuera_del_frente_no_aporta_al_ttc_minimo():
    """Un vehiculo en espejo izq creciendo no debe alterar ttc_minimo_frente_s."""
    ctx = AnalizadorContexto(estimador_fisica=EstimadorFisicaVisual())
    roi = _ROI_DEFAULT[Region.ESPEJO_IZQ]
    cx, cy = (roi[0] + roi[2]) // 2, (roi[1] + roi[3]) // 2

    seg1 = _seg(Clase.VEHICULO, (cx - 50, cy - 50, cx + 50, cy + 50),
                area=10000, track_id=3, edad=5)
    ctx.analizar([seg1], timestamp=0.0)
    seg2 = _seg(Clase.VEHICULO, (cx - 70, cy - 70, cx + 70, cy + 70),
                area=19600, track_id=3, edad=6)
    estado = ctx.analizar([seg2], timestamp=0.3)

    assert math.isinf(estado.ttc_minimo_frente_s)
    assert estado.vehiculo_critico_id is None


def test_vehiculo_que_se_aleja_no_baja_el_ttc():
    """Bbox decreciendo en frente_cercano: TTC sigue siendo infinito."""
    ctx = AnalizadorContexto(estimador_fisica=EstimadorFisicaVisual())
    roi = _ROI_DEFAULT[Region.FRENTE_CERCANO]
    cx, cy = (roi[0] + roi[2]) // 2, (roi[1] + roi[3]) // 2

    seg1 = _seg(Clase.VEHICULO, (cx - 80, cy - 80, cx + 80, cy + 80),
                area=25600, track_id=2, edad=1)
    ctx.analizar([seg1], timestamp=0.0)
    seg2 = _seg(Clase.VEHICULO, (cx - 50, cy - 50, cx + 50, cy + 50),
                area=10000, track_id=2, edad=2)
    estado = ctx.analizar([seg2], timestamp=0.3)

    assert math.isinf(estado.ttc_minimo_frente_s)


def test_dos_vehiculos_en_frente_se_reporta_el_de_menor_ttc():
    """Si hay varios vehiculos en frente, gana el TTC mas bajo."""
    ctx = AnalizadorContexto(estimador_fisica=EstimadorFisicaVisual())
    roi = _ROI_DEFAULT[Region.FRENTE_CERCANO]
    cx, cy = (roi[0] + roi[2]) // 2, (roi[1] + roi[3]) // 2

    # id=10 crece poco (TTC alto), id=20 crece mucho (TTC bajo)
    a1 = _seg(Clase.VEHICULO, (cx - 100, cy - 50, cx, cy + 50),
              area=10000, track_id=10, edad=5)
    b1 = _seg(Clase.VEHICULO, (cx, cy - 50, cx + 100, cy + 50),
              area=10000, track_id=20, edad=5)
    ctx.analizar([a1, b1], timestamp=0.0)

    a2 = _seg(Clase.VEHICULO, (cx - 105, cy - 52, cx, cy + 52),
              area=10920, track_id=10, edad=6)        # crece poco
    b2 = _seg(Clase.VEHICULO, (cx, cy - 75, cx + 150, cy + 75),
              area=22500, track_id=20, edad=6)        # crece mucho
    estado = ctx.analizar([a2, b2], timestamp=0.3)

    assert estado.vehiculo_critico_id == 20
    assert estado.ttc_minimo_frente_s < math.inf


def test_sin_estimador_fisica_el_contexto_funciona_sin_ttc():
    """Compatibilidad: si no se inyecta estimador, ttc_minimo queda en infinito."""
    ctx = AnalizadorContexto()  # sin estimador_fisica
    roi = _ROI_DEFAULT[Region.FRENTE_CERCANO]
    cx, cy = (roi[0] + roi[2]) // 2, (roi[1] + roi[3]) // 2
    seg = _seg(Clase.VEHICULO, (cx - 50, cy - 50, cx + 50, cy + 50), area=10000)
    estado = ctx.analizar([seg])
    assert math.isinf(estado.ttc_minimo_frente_s)
    assert estado.vehiculo_critico_id is None
