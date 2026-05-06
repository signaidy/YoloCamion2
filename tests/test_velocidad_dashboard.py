import cv2
import numpy as np
import pytest

from src.percepcion.velocidad_dashboard import (
    EstimadorVelocidadDashboard,
    _DIGITOS_ASCII,
    _ROI_DIGITOS,
)

_HUD_1 = (
    "....##....",
    "...###....",
    "..####....",
    "...###....",
    "....##....",
    "....##....",
    "....##....",
    "....##....",
    "....##....",
    "....##....",
    "....##....",
    "....##....",
    "..######..",
    "..######..",
)
_HUD_0 = (
    "..######..",
    ".##....##.",
    "##......##",
    "##......##",
    "##......##",
    "##......##",
    "##......##",
    "##......##",
    "##......##",
    "##......##",
    "##......##",
    "##......##",
    ".##....##.",
    "..######..",
)


def _frame_con_velocidad(
    kmh: int,
    w: int = 1920,
    h: int = 1080,
    clutter: bool = False,
    footer_clutter: bool = False,
    x_start: int | None = None,
    y_start: int = 14,
    spacing: int = 2,
) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    roi = np.zeros((45, 80), dtype=np.uint8)
    x = x_start if x_start is not None else (28 if kmh < 10 else 18)
    y = y_start
    for ch in str(kmh):
        glyph = np.array(
            [[255 if c == "#" else 0 for c in row] for row in _DIGITOS_ASCII[int(ch)]],
            dtype=np.uint8,
        )
        roi[y:y + glyph.shape[0], x:x + glyph.shape[1]] = glyph
        x += glyph.shape[1] + spacing

    if clutter:
        cv2.putText(roi, "10", (52, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.28, 255, 1, cv2.LINE_AA)
        cv2.rectangle(roi, (0, 2), (10, 6), 255, -1)
        cv2.putText(roi, "km/h", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.28, 190, 1, cv2.LINE_AA)
    if footer_clutter:
        for tick_x in (31, 35, 39, 43, 47):
            cv2.rectangle(roi, (tick_x, 31), (tick_x + 1, 37), 220, -1)
        cv2.putText(roi, "km/h", (22, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.24, 180, 1, cv2.LINE_AA)

    x1, y1 = int(round(w * _ROI_DIGITOS[0])), int(round(h * _ROI_DIGITOS[1]))
    x2, y2 = int(round(w * _ROI_DIGITOS[2])), int(round(h * _ROI_DIGITOS[3]))
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    frame[y1:y2, x1:x2] = cv2.resize(roi_bgr, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return frame


def _frame_con_glyphs(
    glyphs: list[tuple[str, ...]],
    w: int = 1920,
    h: int = 1080,
    clutter: bool = False,
    footer_clutter: bool = False,
    x_start: int = 18,
    y_start: int = 16,
) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    roi = np.zeros((45, 80), dtype=np.uint8)
    x = x_start
    for glyph_rows in glyphs:
        glyph = np.array(
            [[255 if c == "#" else 0 for c in row] for row in glyph_rows],
            dtype=np.uint8,
        )
        roi[y_start:y_start + glyph.shape[0], x:x + glyph.shape[1]] = glyph
        x += glyph.shape[1] + 2

    if clutter:
        cv2.putText(roi, "10", (52, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.28, 255, 1, cv2.LINE_AA)
        cv2.rectangle(roi, (0, 2), (10, 6), 255, -1)
        cv2.putText(roi, "km/h", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.28, 190, 1, cv2.LINE_AA)
    if footer_clutter:
        for tick_x in (31, 35, 39, 43, 47):
            cv2.rectangle(roi, (tick_x, 31), (tick_x + 1, 37), 220, -1)
        cv2.putText(roi, "km/h", (22, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.24, 180, 1, cv2.LINE_AA)

    x1, y1 = int(round(w * _ROI_DIGITOS[0])), int(round(h * _ROI_DIGITOS[1]))
    x2, y2 = int(round(w * _ROI_DIGITOS[2])), int(round(h * _ROI_DIGITOS[3]))
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


def test_lee_velocidad_dashboard_cero_hud_con_hueco_no_se_parte_en_cuarenta_y_uno():
    est = EstimadorVelocidadDashboard(max_kmh_norm=100.0)
    lectura = est.leer(_frame_con_glyphs([_HUD_0], clutter=True))

    assert lectura.valido
    assert lectura.kmh == 0
    assert lectura.norm == 0.0


def test_lee_velocidad_dashboard_con_clutter_del_cuadrante():
    est = EstimadorVelocidadDashboard(max_kmh_norm=100.0)
    lectura = est.leer(_frame_con_velocidad(32, clutter=True))

    assert lectura.valido
    assert lectura.kmh == 32
    assert lectura.norm == 0.32


@pytest.mark.parametrize("kmh", [13, 20, 25])
def test_lee_velocidad_dashboard_velocidades_reales_con_clutter(kmh: int):
    est = EstimadorVelocidadDashboard(max_kmh_norm=100.0)
    lectura = est.leer(_frame_con_velocidad(kmh, clutter=True))

    assert lectura.valido
    assert lectura.kmh == kmh
    assert lectura.norm == pytest.approx(kmh / 100.0)


@pytest.mark.parametrize("kmh", [42, 43])
def test_lee_velocidad_dashboard_digitos_cerca_del_borde_derecho(kmh: int):
    est = EstimadorVelocidadDashboard(max_kmh_norm=100.0)
    lectura = est.leer(_frame_con_velocidad(kmh, clutter=True, x_start=58))

    assert lectura.valido
    assert lectura.kmh == kmh
    assert lectura.norm == pytest.approx(kmh / 100.0)


def test_lee_velocidad_dashboard_digitos_bajos_y_cerca_del_borde():
    est = EstimadorVelocidadDashboard(max_kmh_norm=100.0)
    lectura = est.leer(_frame_con_velocidad(27, clutter=True, x_start=58, y_start=20))

    assert lectura.valido
    assert lectura.kmh == 27
    assert lectura.norm == pytest.approx(0.27)


def test_lee_velocidad_dashboard_diez_hud_no_se_confunde_con_noventa():
    est = EstimadorVelocidadDashboard(max_kmh_norm=100.0)
    lectura = est.leer(_frame_con_glyphs([_HUD_1, _HUD_0], clutter=True))

    assert lectura.valido
    assert lectura.kmh == 10
    assert lectura.norm == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("glyphs", "esperado"),
    [
        ([_HUD_1, _DIGITOS_ASCII[3]], 13),
        ([_HUD_1, _HUD_0], 10),
        ([_DIGITOS_ASCII[2], _DIGITOS_ASCII[7]], 27),
        ([_HUD_1, _DIGITOS_ASCII[5]], 15),
        ([_HUD_1, _HUD_1], 11),
    ],
)
def test_lee_velocidad_dashboard_no_confunde_digitos_con_clutter_inferior(glyphs, esperado):
    est = EstimadorVelocidadDashboard(max_kmh_norm=100.0)
    lectura = est.leer(_frame_con_glyphs(glyphs, footer_clutter=True))

    assert lectura.valido
    assert lectura.kmh == esperado
    assert lectura.norm == pytest.approx(esperado / 100.0)


@pytest.mark.parametrize("kmh", [0, 16, 42])
def test_lee_velocidad_dashboard_velocidades_reales_del_run_reciente(kmh: int):
    est = EstimadorVelocidadDashboard(max_kmh_norm=100.0)
    lectura = est.leer(_frame_con_velocidad(kmh, footer_clutter=True))

    assert lectura.valido
    assert lectura.kmh == kmh
    assert lectura.norm == pytest.approx(kmh / 100.0)


@pytest.mark.parametrize("kmh", [16, 42, 48, 54])
def test_lee_velocidad_dashboard_digitos_tocandose_siguen_leyendose(kmh: int):
    est = EstimadorVelocidadDashboard(max_kmh_norm=100.0)
    lectura = est.leer(_frame_con_velocidad(kmh, footer_clutter=True, spacing=0))

    assert lectura.valido
    assert lectura.kmh == kmh
    assert lectura.norm == pytest.approx(kmh / 100.0)


@pytest.mark.parametrize("kmh", [17, 31, 48, 54])
def test_lee_velocidad_dashboard_run_actual_con_digitos_muy_cercanos(kmh: int):
    est = EstimadorVelocidadDashboard(max_kmh_norm=100.0)
    lectura = est.leer(_frame_con_velocidad(kmh, footer_clutter=True, spacing=1))

    assert lectura.valido
    assert lectura.kmh == kmh
    assert lectura.norm == pytest.approx(kmh / 100.0)


def test_extrae_componentes_bajos_detectados_en_run_actual():
    est = EstimadorVelocidadDashboard(max_kmh_norm=100.0)
    mask = np.zeros((45, 80), dtype=np.uint8)
    cv2.rectangle(mask, (10, 12), (12, 22), 255, -1)  # 11x3, como el "1" del run
    cv2.rectangle(mask, (18, 12), (32, 22), 255, -1)  # 11x15, como el dígito ancho del run

    componentes = est._extraer_componentes(mask)

    assert [(w, h) for _x, w, h, _area, _comp in componentes] == [(3, 11), (15, 11)]


def test_retiene_ultima_lectura_si_un_frame_falla():
    est = EstimadorVelocidadDashboard(max_kmh_norm=100.0, retener_frames=2)
    assert est.estimar(_frame_con_velocidad(58)).kmh == 58

    lectura = est.estimar(np.zeros((1080, 1920, 3), dtype=np.uint8))

    assert not lectura.valido
    assert lectura.kmh == 58
    assert lectura.norm == 0.58


def test_retiene_ultima_lectura_si_hay_salto_imposible():
    est = EstimadorVelocidadDashboard(max_kmh_norm=100.0, retener_frames=2)
    assert est.estimar(_frame_con_velocidad(0)).kmh == 0

    lectura = est.estimar(_frame_con_velocidad(90))

    assert not lectura.valido
    assert lectura.kmh == 0
    assert lectura.norm == 0.0
