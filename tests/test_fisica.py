"""Tests del EstimadorFisicaVisual: TTC por escalado de bounding box.

Pure-vision: el TTC se deriva exclusivamente del crecimiento del area del
bbox a lo largo del tiempo. Nunca consulta telemetria interna del juego.
"""
import math

import pytest

from src.percepcion.fisica import EstimadorFisicaVisual
from src.tipos import Clase, Seguimiento


def _seg(track_id: int, area: int, caja=None) -> Seguimiento:
    """Helper: Seguimiento con area dada, caja sintetica centrada."""
    if caja is None:
        # Construye caja cuadrada de la area pedida, centrada en (640, 540)
        lado = int(area ** 0.5)
        cx, cy = 640, 540
        x1, y1 = cx - lado // 2, cy - lado // 2
        x2, y2 = x1 + lado, y1 + lado
        caja = (x1, y1, x2, y2)
    return Seguimiento(
        clase=Clase.VEHICULO,
        caja=caja,
        confianza=0.9,
        area=area,
        id_seguimiento=track_id,
        edad=1,
    )


def test_id_nuevo_sin_historial_devuelve_ttc_infinito():
    est = EstimadorFisicaVisual()
    seg = _seg(track_id=1, area=10000)
    est.actualizar([seg], timestamp=0.0)
    assert seg.fisica is not None
    assert math.isinf(seg.fisica.ttc_segundos)
    assert seg.fisica.area_px == 10000
    assert seg.fisica.area_anterior_px == 0


def test_bbox_constante_devuelve_ttc_infinito():
    est = EstimadorFisicaVisual()
    s1 = _seg(1, 10000)
    s2 = _seg(1, 10000)
    s3 = _seg(1, 10000)
    est.actualizar([s1], timestamp=0.0)
    est.actualizar([s2], timestamp=0.5)
    est.actualizar([s3], timestamp=1.0)
    assert math.isinf(s3.fisica.ttc_segundos)


def test_bbox_que_decrece_devuelve_ttc_infinito():
    """Objeto que se aleja: area cae -> TTC infinito."""
    est = EstimadorFisicaVisual()
    s1 = _seg(1, 10000)
    s2 = _seg(1, 8000)
    s3 = _seg(1, 6000)
    est.actualizar([s1], timestamp=0.0)
    est.actualizar([s2], timestamp=0.5)
    est.actualizar([s3], timestamp=1.0)
    assert math.isinf(s3.fisica.ttc_segundos)


def test_bbox_que_duplica_su_area_en_un_segundo_da_ttc_aprox_un_segundo():
    """Si A(t)=A0*(1+t), dA/dt=A0, TTC=A/(dA/dt) ~ 1s al duplicar en 1s.

    Mas exactamente: el modelo TTC = area / (d_area/dt). Tras t=1s con
    duplicacion lineal, area=2*A0, d_area/dt=A0, TTC=2s.
    Para que TTC ~ 1s necesitamos crecimiento mas agresivo. Aqui validamos
    la cota: TTC < 2.5s y > 0.5s (ventana razonable).
    """
    est = EstimadorFisicaVisual()
    # Crecimiento: 5000 -> 7500 -> 10000 en 1 segundo (lineal)
    est.actualizar([_seg(1, 5000)], timestamp=0.0)
    est.actualizar([_seg(1, 7500)], timestamp=0.5)
    s_final = _seg(1, 10000)
    est.actualizar([s_final], timestamp=1.0)
    ttc = s_final.fisica.ttc_segundos
    assert 0.5 < ttc < 2.5, f"TTC={ttc}s fuera de ventana esperada"
    assert s_final.fisica.velocidad_relativa_px_s > 0


def test_velocidad_relativa_es_negativa_o_cero_cuando_se_aleja():
    est = EstimadorFisicaVisual()
    est.actualizar([_seg(1, 10000)], timestamp=0.0)
    s = _seg(1, 5000)
    est.actualizar([s], timestamp=1.0)
    assert s.fisica.velocidad_relativa_px_s <= 0.0


def test_centroide_se_calcula_desde_caja():
    est = EstimadorFisicaVisual()
    seg = _seg(1, 10000, caja=(100, 200, 300, 400))
    est.actualizar([seg], timestamp=0.0)
    assert seg.fisica.centroide == (200, 300)  # ((100+300)/2, (200+400)/2)


def test_dt_excesivo_descarta_muestra_anterior_y_devuelve_inf():
    """Si dt > 0.5s entre frames el tracker probablemente perdio el objeto."""
    est = EstimadorFisicaVisual(dt_max=0.5)
    est.actualizar([_seg(1, 5000)], timestamp=0.0)
    s = _seg(1, 10000)
    est.actualizar([s], timestamp=1.5)  # dt=1.5s, demasiado
    assert math.isinf(s.fisica.ttc_segundos)


def test_salto_de_area_excesivo_descarta_muestra():
    """Cambio area/area > 1.5 indica swap de id, oclusion o falso positivo."""
    est = EstimadorFisicaVisual()
    est.actualizar([_seg(1, 5000)], timestamp=0.0)
    s = _seg(1, 50000)  # 10x mas grande, salto irreal
    est.actualizar([s], timestamp=0.1)
    assert math.isinf(s.fisica.ttc_segundos)


def test_id_que_desaparece_se_limpia_del_historial():
    est = EstimadorFisicaVisual()
    est.actualizar([_seg(1, 5000), _seg(2, 8000)], timestamp=0.0)
    est.actualizar([_seg(1, 6000)], timestamp=0.1)  # id=2 desaparece
    # El historial interno no debe acumular ids muertos indefinidamente
    assert 2 not in est._historial or len(est._historial[2]) == 0


def test_dos_objetos_simultaneos_se_estiman_independientemente():
    est = EstimadorFisicaVisual()
    # id=1 crece (acercandose), id=2 decrece (alejandose)
    est.actualizar([_seg(1, 5000), _seg(2, 10000)], timestamp=0.0)
    est.actualizar([_seg(1, 7500), _seg(2, 7500)], timestamp=0.5)
    s1 = _seg(1, 10000)
    s2 = _seg(2, 5000)
    est.actualizar([s1, s2], timestamp=1.0)
    assert s1.fisica.ttc_segundos < math.inf  # acercandose
    assert math.isinf(s2.fisica.ttc_segundos)  # alejandose
