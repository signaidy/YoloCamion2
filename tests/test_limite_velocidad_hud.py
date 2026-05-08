import cv2
import numpy as np
import pytest


ROI = (0.62, 0.58, 0.84, 0.95)


def _frame_con_limite(
    limite_kmh: int | None,
    w: int = 400,
    h: int = 300,
    con_panel_velocidad: bool = False,
) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    x1 = int(round(w * ROI[0]))
    y1 = int(round(h * ROI[1]))
    x2 = int(round(w * ROI[2]))
    y2 = int(round(h * ROI[3]))
    roi = np.full((y2 - y1, x2 - x1, 3), 30, dtype=np.uint8)

    if limite_kmh is not None:
        cx = roi.shape[1] // 2
        cy = roi.shape[0] // 2
        r = min(roi.shape[0], roi.shape[1]) // 2 - 6
        cv2.circle(roi, (cx, cy), r, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(roi, (cx, cy), r - 8, (245, 245, 245), thickness=-1, lineType=cv2.LINE_AA)
        texto = str(limite_kmh)
        escala = 0.95 if len(texto) == 2 else 0.75
        grosor = 2
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
        org = (cx - tw // 2, cy + th // 2)
        cv2.putText(
            roi,
            texto,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            escala,
            (25, 25, 25),
            grosor,
            cv2.LINE_AA,
        )

    if con_panel_velocidad:
        cv2.rectangle(roi, (0, 10), (32, 50), (70, 70, 70), thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(roi, "12", (3, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(roi, "km/h", (30, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2, cv2.LINE_AA)

    frame[y1:y2, x1:x2] = roi
    return frame


def _estimador():
    from src.percepcion.limite_velocidad_hud import EstimadorLimiteVelocidadHUD

    return EstimadorLimiteVelocidadHUD(
        roi=ROI,
        min_confianza=0.40,
        tam_signo=(96, 96),
    )


@pytest.mark.parametrize("limite", [30, 50, 60, 80])
def test_limite_hud_detecta_valores_basicos(limite: int):
    est = _estimador()

    estado = est.estimar(_frame_con_limite(limite))

    assert estado.visible is True
    assert estado.limite_kmh == limite
    assert estado.confianza >= 0.40


def test_limite_hud_devuelve_none_si_no_hay_senal():
    est = _estimador()

    estado = est.estimar(_frame_con_limite(None))

    assert estado.visible is False
    assert estado.limite_kmh is None
    assert estado.confianza == pytest.approx(0.0)


def test_limite_hud_ignora_panel_velocidad_y_conserva_solo_la_senal():
    est = _estimador()

    estado = est.estimar(_frame_con_limite(80, con_panel_velocidad=True))

    assert estado.visible is True
    assert estado.limite_kmh == 80
    assert estado.confianza >= 0.40
