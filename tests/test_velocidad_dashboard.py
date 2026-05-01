import cv2
import numpy as np

from src.percepcion.velocidad_dashboard import EstimadorVelocidadDashboard, _DIGITOS_ASCII


def _frame_con_velocidad(kmh: int, w: int = 1920, h: int = 1080) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    roi = np.zeros((45, 80), dtype=np.uint8)
    x = 24 if kmh < 10 else 18
    y = 25
    for ch in str(kmh):
        glyph = np.array(
            [[255 if c == "#" else 0 for c in row] for row in _DIGITOS_ASCII[int(ch)]],
            dtype=np.uint8,
        )
        roi[y:y + glyph.shape[0], x:x + glyph.shape[1]] = glyph
        x += glyph.shape[1] + 2

    x1, y1 = int(round(w * 0.023)), int(round(h * 0.844))
    x2, y2 = int(round(w * 0.116)), int(round(h * 0.936))
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    frame[y1:y2, x1:x2] = cv2.resize(roi_bgr, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return frame


def test_lee_velocidad_dashboard_dos_digitos():
    est = EstimadorVelocidadDashboard(max_kmh_norm=100.0)
    lectura = est.leer(_frame_con_velocidad(42))

    assert lectura.valido
    assert lectura.kmh == 42
    assert lectura.norm == 0.42


def test_lee_velocidad_dashboard_cero():
    est = EstimadorVelocidadDashboard(max_kmh_norm=100.0)
    lectura = est.leer(_frame_con_velocidad(0))

    assert lectura.valido
    assert lectura.kmh == 0
    assert lectura.norm == 0.0


def test_retiene_ultima_lectura_si_un_frame_falla():
    est = EstimadorVelocidadDashboard(max_kmh_norm=100.0, retener_frames=2)
    assert est.estimar(_frame_con_velocidad(58)).kmh == 58

    lectura = est.estimar(np.zeros((1080, 1920, 3), dtype=np.uint8))

    assert not lectura.valido
    assert lectura.kmh == 58
    assert lectura.norm == 0.58
