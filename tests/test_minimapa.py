import cv2
import numpy as np
import pytest

from src.tipos import ManiobraRuta


ROI = (0.20, 0.15, 0.80, 0.95)


def _frame_con_minimapa(
    puntos: list[tuple[int, int]],
    ramas: list[list[tuple[int, int]]] | None = None,
    w: int = 400,
    h: int = 300,
) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    x1 = int(round(w * ROI[0]))
    y1 = int(round(h * ROI[1]))
    x2 = int(round(w * ROI[2]))
    y2 = int(round(h * ROI[3]))
    roi = np.full((y2 - y1, x2 - x1, 3), 25, dtype=np.uint8)

    cv2.polylines(
        roi,
        [np.array(puntos, dtype=np.int32)],
        isClosed=False,
        color=(0, 0, 255),
        thickness=10,
        lineType=cv2.LINE_AA,
    )
    for rama in ramas or []:
        cv2.polylines(
            roi,
            [np.array(rama, dtype=np.int32)],
            isClosed=False,
            color=(0, 0, 255),
            thickness=10,
            lineType=cv2.LINE_AA,
        )

    camion = np.array(
        [
            (roi.shape[1] // 2, int(roi.shape[0] * 0.72)),
            (roi.shape[1] // 2 - 10, int(roi.shape[0] * 0.88)),
            (roi.shape[1] // 2 + 10, int(roi.shape[0] * 0.88)),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(roi, camion, (0, 255, 0), lineType=cv2.LINE_AA)
    frame[y1:y2, x1:x2] = roi
    return frame


def _estimador():
    from src.percepcion.minimapa import EstimadorMinimapa

    return EstimadorMinimapa(
        roi=ROI,
        referencia_camion=(0.50, 0.80),
        min_confianza=0.25,
        umbral_mantener=0.08,
        umbral_giro_fuerte=0.20,
    )


def test_minimapa_detecta_recta():
    est = _estimador()
    frame = _frame_con_minimapa([(120, 150), (120, 95), (120, 35)])

    estado = est.estimar(frame)

    assert estado.visible is True
    assert estado.maniobra is ManiobraRuta.SEGUIR_RECTO
    assert estado.confianza >= 0.25
    assert estado.ramal_objetivo == "centro"
    assert estado.requiere_cambio_carril is False


def test_minimapa_detecta_salida_derecha():
    est = _estimador()
    frame = _frame_con_minimapa(
        [(120, 150), (120, 98), (120, 46)],
        ramas=[[(120, 104), (150, 84), (182, 58)]],
    )

    estado = est.estimar(frame)

    assert estado.visible is True
    assert estado.maniobra is ManiobraRuta.MANTENER_DER
    assert estado.ramal_objetivo == "der"
    assert estado.requiere_cambio_carril is True
    assert estado.sesgo_lateral_objetivo > 0


def test_minimapa_detecta_salida_izquierda():
    est = _estimador()
    frame = _frame_con_minimapa(
        [(120, 150), (120, 98), (120, 46)],
        ramas=[[(120, 104), (90, 84), (58, 58)]],
    )

    estado = est.estimar(frame)

    assert estado.visible is True
    assert estado.maniobra is ManiobraRuta.MANTENER_IZQ
    assert estado.ramal_objetivo == "izq"
    assert estado.requiere_cambio_carril is True
    assert estado.sesgo_lateral_objetivo < 0


def test_minimapa_detecta_lado_derecho_cuando_el_ramal_ya_es_cercano():
    est = _estimador()
    frame = _frame_con_minimapa(
        [(120, 150), (120, 140), (120, 132)],
        ramas=[[(120, 144), (150, 143), (184, 140)]],
    )

    estado = est.estimar(frame)

    assert estado.visible is True
    assert estado.maniobra in {ManiobraRuta.MANTENER_DER, ManiobraRuta.SALIDA_DER}
    assert estado.ramal_objetivo == "der"
    assert estado.requiere_cambio_carril is True


def test_minimapa_detecta_giro_derecha_fuerte():
    est = _estimador()
    frame = _frame_con_minimapa([(120, 150), (126, 118), (148, 82), (184, 50)])

    estado = est.estimar(frame)

    assert estado.visible is True
    assert estado.maniobra is ManiobraRuta.GIRO_DER
    assert estado.ramal_objetivo == "der"
    assert estado.confianza >= 0.25


def test_minimapa_curva_derecha_no_pide_cambio_de_carril_sin_ramal_real():
    est = _estimador()
    frame = _frame_con_minimapa([(120, 154), (122, 126), (135, 96), (160, 66), (188, 42)])

    estado = est.estimar(frame)

    assert estado.visible is True
    assert estado.maniobra not in {ManiobraRuta.SALIDA_DER, ManiobraRuta.MANTENER_DER}
    assert estado.requiere_cambio_carril is False


def test_minimapa_curva_derecha_con_trazo_secundario_mismo_lado_no_activa_cambio():
    est = _estimador()
    frame = _frame_con_minimapa(
        [(120, 154), (124, 128), (138, 96), (164, 68), (192, 44)],
        ramas=[[(144, 108), (172, 86), (198, 66)]],
    )

    estado = est.estimar(frame)

    assert estado.visible is True
    assert estado.maniobra not in {ManiobraRuta.SALIDA_DER, ManiobraRuta.MANTENER_DER}
    assert estado.requiere_cambio_carril is False


def test_minimapa_devuelve_desconocida_si_no_hay_ruta():
    est = _estimador()
    frame = np.zeros((300, 400, 3), dtype=np.uint8)

    estado = est.estimar(frame)

    assert estado.visible is False
    assert estado.maniobra is ManiobraRuta.DESCONOCIDA
    assert estado.confianza == pytest.approx(0.0)
    assert estado.distancia_normalizada is None
