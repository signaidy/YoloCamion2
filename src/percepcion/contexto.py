import math
from collections import deque
from pathlib import Path
from typing import Optional
import time

import numpy as np
import yaml

from src.tipos import Clase, EstadoEscena, EstadoSemaforo, Region, Seguimiento
from src.percepcion.fisica import EstimadorFisicaVisual
from src.percepcion.semaforo import clasificar_semaforo

_NOMBRE_A_REGION = {
    "frente_cercano": Region.FRENTE_CERCANO,
    "frente_lejano":  Region.FRENTE_LEJANO,
    "espejo_izq":     Region.ESPEJO_IZQ,
    "espejo_der":     Region.ESPEJO_DER,
    "lateral_izq":    Region.LATERAL_IZQ,
    "lateral_der":    Region.LATERAL_DER,
}


def cargar_rois_yaml(ruta: str | Path) -> dict[Region, tuple[int, int, int, int]]:
    """Carga las ROI calibradas desde un archivo YAML."""
    with open(ruta, encoding="utf-8") as f:
        datos = yaml.safe_load(f)
    rois = {}
    for nombre, coords in datos.items():
        region = _NOMBRE_A_REGION.get(nombre)
        if region is not None:
            rois[region] = tuple(coords)
    return rois


# ROI por defecto para 1920x1080 — ajustables desde regiones_interes.yaml
_ROI_DEFAULT: dict[Region, tuple[int, int, int, int]] = {
    Region.FRENTE_CERCANO: (480, 540, 1440, 1080),
    Region.FRENTE_LEJANO:  (480, 270,  1440,  540),
    Region.ESPEJO_IZQ:     (0,   200,  320,   600),
    Region.ESPEJO_DER:     (1600, 200, 1920,  600),
    Region.LATERAL_IZQ:    (0,   400,  480,   900),
    Region.LATERAL_DER:    (1440, 400, 1920,  900),
}

_AREA_MIN_FRENTE = 5000   # px² para considerar vehículo relevante en frente cercano
_AREA_MIN_ESPEJO = 2000
_AREA_MIN_PEATON = 3000   # peatón muy pequeño = falso positivo (HUD, árbol, poste)


def _intersecta(caja: tuple, roi: tuple) -> bool:
    x1, y1, x2, y2 = caja
    rx1, ry1, rx2, ry2 = roi
    return x1 < rx2 and x2 > rx1 and y1 < ry2 and y2 > ry1


class AnalizadorContexto:
    """Convierte una lista de Seguimiento en EstadoEscena usando ROIs configurables.

    Si se inyecta un EstimadorFisicaVisual, cada Seguimiento queda anotado
    con su FisicaVisual (TTC y velocidad relativa por escalado de bbox), y el
    EstadoEscena reporta el TTC minimo de los vehiculos en frente cercano o
    lejano + el id del vehiculo critico.
    """

    def __init__(
        self,
        rois: Optional[dict[Region, tuple]] = None,
        ventana_confianza: int = 10,
        estimador_fisica: Optional[EstimadorFisicaVisual] = None,
    ):
        self._rois = rois or _ROI_DEFAULT
        self._historial_tracks: deque[int] = deque(maxlen=ventana_confianza)
        self._estimador_fisica = estimador_fisica

    def analizar(
        self,
        seguimientos: list[Seguimiento],
        imagen: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> EstadoEscena:
        ts = timestamp if timestamp is not None else time.monotonic()
        if self._estimador_fisica is not None:
            self._estimador_fisica.actualizar(seguimientos, ts)

        frente_cercano = False
        frente_lejano = False
        peaton_riesgo = False
        espejo_izq = False
        espejo_der = False
        semaforo_estado: Optional[EstadoSemaforo] = None
        senal_alto = False

        roi_fc = self._rois[Region.FRENTE_CERCANO]
        roi_fl = self._rois[Region.FRENTE_LEJANO]
        roi_ei = self._rois[Region.ESPEJO_IZQ]
        roi_ed = self._rois[Region.ESPEJO_DER]
        roi_li = self._rois[Region.LATERAL_IZQ]
        roi_ld = self._rois[Region.LATERAL_DER]

        ttc_min = math.inf
        id_critico: Optional[int] = None

        for seg in seguimientos:
            caja = seg.caja

            if seg.clase == Clase.SEMAFORO and imagen is not None:
                estado = clasificar_semaforo(imagen, caja)
                if estado != EstadoSemaforo.DESCONOCIDO:
                    semaforo_estado = estado

            elif seg.clase == Clase.SENAL_ALTO:
                if _intersecta(caja, roi_fc):
                    senal_alto = True

            elif seg.clase == Clase.PEATON:
                if seg.area >= _AREA_MIN_PEATON and (
                    _intersecta(caja, roi_fc) or _intersecta(caja, roi_li) or _intersecta(caja, roi_ld)
                ):
                    peaton_riesgo = True

            elif seg.clase in (Clase.VEHICULO, Clase.MOTOCICLETA):
                en_frente = _intersecta(caja, roi_fc) or _intersecta(caja, roi_fl)
                if _intersecta(caja, roi_fc) and seg.area >= _AREA_MIN_FRENTE:
                    frente_cercano = True
                if _intersecta(caja, roi_fl):
                    frente_lejano = True
                if _intersecta(caja, roi_ei) and seg.edad >= 3 and seg.area >= _AREA_MIN_ESPEJO:
                    espejo_izq = True
                if _intersecta(caja, roi_ed) and seg.edad >= 3 and seg.area >= _AREA_MIN_ESPEJO:
                    espejo_der = True

                # TTC minimo solo cuenta vehiculos en zona frontal: el riesgo
                # de colision frontal es lo que dispara FRENAR_FUERTE (R3.5).
                if en_frente and seg.fisica is not None:
                    ttc = seg.fisica.ttc_segundos
                    if ttc < ttc_min:
                        ttc_min = ttc
                        id_critico = seg.id_seguimiento

        n_tracks = len(seguimientos)
        self._historial_tracks.append(n_tracks)
        # Confianza = fracción de frames procesados exitosamente por YOLO.
        # Una carretera vacía (0 detecciones) es válida — no penaliza la confianza.
        # Solo cae si el pipeline deja de llamar a analizar() (watchdog lo detecta).
        confianza = min(1.0, len(self._historial_tracks) / self._historial_tracks.maxlen)

        return EstadoEscena(
            frente_cercano_ocupado=frente_cercano,
            frente_lejano_ocupado=frente_lejano,
            peaton_en_riesgo=peaton_riesgo,
            semaforo_visible=semaforo_estado,
            senal_alto_cercana=senal_alto,
            espejo_izq_ocupado=espejo_izq,
            espejo_der_ocupado=espejo_der,
            vehiculos_totales=n_tracks,
            confianza_percepcion=confianza,
            timestamp=ts,
            ttc_minimo_frente_s=ttc_min,
            vehiculo_critico_id=id_critico,
        )
