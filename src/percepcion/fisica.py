"""Estimador de fisica visual: TTC y velocidad relativa por escalado de bbox.

Pure-vision (RNF-07). El TTC se deriva del crecimiento del area del bounding
box a lo largo del tiempo:

    velocidad_aprox = d(area)/dt          (px^2 / s)
    TTC = area / velocidad_aprox          (s, solo si >0)

Cuando el objeto se aleja, mantiene su tamano o salta de identidad, el TTC se
reporta como +inf (no hay riesgo de colision medible). El estimador filtra
muestras con dt excesivo o saltos de area incompatibles con movimiento real.

La capa Optica (Tarea 1.3) puede combinar este estimador con flujo denso para
mayor robustez; aqui solo nos basamos en el area, que es la senal mas barata
y la mas universal (todo Seguimiento la trae poblada).
"""
import math
from collections import defaultdict, deque
from typing import Optional

from src.tipos import FisicaVisual, Seguimiento

_VENTANA_HISTORIAL = 8           # numero de muestras (frames) por id rastreado
_DT_MAX_DEFAULT = 0.5            # s; gap mayor descarta muestra anterior
_RAZON_AREA_MAX = 1.5            # |delta_area| / area_anterior > esto = salto irreal


class EstimadorFisicaVisual:
    """Mantiene historial corto por id y calcula FisicaVisual cada frame.

    Uso:
        estimador = EstimadorFisicaVisual()
        estimador.actualizar(seguimientos, timestamp=time.monotonic())
        for seg in seguimientos:
            print(seg.fisica.ttc_segundos)
    """

    def __init__(
        self,
        ventana: int = _VENTANA_HISTORIAL,
        dt_max: float = _DT_MAX_DEFAULT,
        razon_area_max: float = _RAZON_AREA_MAX,
    ):
        self._ventana = ventana
        self._dt_max = dt_max
        self._razon_area_max = razon_area_max
        # historial[id] = deque[(timestamp, area)]
        self._historial: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self._ventana)
        )

    def actualizar(
        self,
        seguimientos: list[Seguimiento],
        timestamp: float,
    ) -> None:
        """Anota fisica en cada Seguimiento y limpia ids desaparecidos."""
        ids_actuales: set[int] = set()

        for seg in seguimientos:
            tid = seg.id_seguimiento
            ids_actuales.add(tid)
            seg.fisica = self._calcular(tid, seg, timestamp)

        # Limpiar tracks que desaparecieron del frame
        ids_muertos = set(self._historial.keys()) - ids_actuales
        for tid in ids_muertos:
            del self._historial[tid]

    def _calcular(
        self,
        tid: int,
        seg: Seguimiento,
        timestamp: float,
    ) -> FisicaVisual:
        x1, y1, x2, y2 = seg.caja
        centroide = ((x1 + x2) // 2, (y1 + y2) // 2)
        area_actual = seg.area

        hist = self._historial[tid]

        # Sin muestra previa: registrar y devolver fisica neutra
        if not hist:
            hist.append((timestamp, area_actual))
            return FisicaVisual(
                area_px=area_actual,
                area_anterior_px=0,
                centroide=centroide,
            )

        ts_prev, area_prev = hist[-1]
        dt = timestamp - ts_prev

        # Filtro 1: gap temporal excesivo => probable perdida del track
        if dt <= 0 or dt > self._dt_max:
            hist.clear()
            hist.append((timestamp, area_actual))
            return FisicaVisual(
                area_px=area_actual,
                area_anterior_px=area_prev,
                centroide=centroide,
            )

        # Filtro 2: salto de area incompatible con movimiento real
        razon_cambio = abs(area_actual - area_prev) / max(area_prev, 1)
        if razon_cambio > self._razon_area_max:
            hist.clear()
            hist.append((timestamp, area_actual))
            return FisicaVisual(
                area_px=area_actual,
                area_anterior_px=area_prev,
                centroide=centroide,
            )

        # Velocidad de crecimiento del area en px^2/s
        d_area = area_actual - area_prev
        velocidad = d_area / dt

        # TTC = area / d(area)/dt, solo cuando crece
        if velocidad > 0 and area_actual > 0:
            ttc = area_actual / velocidad
        else:
            ttc = math.inf

        hist.append((timestamp, area_actual))

        return FisicaVisual(
            velocidad_relativa_px_s=velocidad,
            ttc_segundos=ttc,
            area_px=area_actual,
            area_anterior_px=area_prev,
            centroide=centroide,
        )

    def reset(self) -> None:
        self._historial.clear()
