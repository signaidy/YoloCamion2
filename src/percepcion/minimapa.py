"""Parser determinista del minimapa de ETS2.

La salida de este modulo es observacional en Fase 1: clasifica maniobra y
provee una intencion de ruta, pero no toca el control lateral.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.tipos import EstadoRuta, ManiobraRuta

_DEFAULT_ROI = (0.792, 0.676, 0.995, 0.995)
_DEFAULT_REF_CAMION = (0.50, 0.78)
_DEFAULT_HSV_RUTA = (
    (0, 70, 120, 18, 255, 255),
    (160, 70, 120, 179, 255, 255),
)
_DEFAULT_HSV_CAMION = (
    (35, 40, 80, 95, 255, 255),
)


@dataclass(frozen=True)
class _FilaRuta:
    y: int
    segmentos: tuple[tuple[int, int], ...]


class EstimadorMinimapa:
    def __init__(
        self,
        roi: tuple[float, float, float, float] = _DEFAULT_ROI,
        referencia_camion: tuple[float, float] = _DEFAULT_REF_CAMION,
        hsv_ruta: tuple[tuple[int, int, int, int, int, int], ...] = _DEFAULT_HSV_RUTA,
        hsv_camion: tuple[tuple[int, int, int, int, int, int], ...] = _DEFAULT_HSV_CAMION,
        min_confianza: float = 0.35,
        umbral_mantener: float = 0.10,
        umbral_giro_fuerte: float = 0.22,
        fraccion_inferior_ignorar_ramal: float = 0.18,
        min_filas_ramal: int = 3,
        umbral_distancia_salida: float = 0.18,
    ):
        self.roi = tuple(float(v) for v in roi)
        self.referencia_camion = tuple(float(v) for v in referencia_camion)
        self.hsv_ruta = tuple(tuple(int(v) for v in r) for r in hsv_ruta)
        self.hsv_camion = tuple(tuple(int(v) for v in r) for r in hsv_camion)
        self.min_confianza = float(min_confianza)
        self.umbral_mantener = float(umbral_mantener)
        self.umbral_giro_fuerte = float(umbral_giro_fuerte)
        self.fraccion_inferior_ignorar_ramal = float(np.clip(fraccion_inferior_ignorar_ramal, 0.0, 0.45))
        self.min_filas_ramal = max(2, int(min_filas_ramal))
        self.umbral_distancia_salida = max(0.0, float(umbral_distancia_salida))
        self._debug: dict[str, object] = {}

    def estimar(self, frame_bgr: np.ndarray) -> EstadoRuta:
        roi_bgr, _ = self._recortar_roi(frame_bgr)
        if roi_bgr.size == 0:
            return self._estado_vacio()

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask_ruta = self._mask_hsv(hsv, self.hsv_ruta)
        mask_ruta = cv2.morphologyEx(mask_ruta, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask_ruta = cv2.morphologyEx(mask_ruta, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        mask_camion = self._mask_hsv(hsv, self.hsv_camion)
        ancla = self._estimar_ancla(mask_camion, roi_bgr.shape[1], roi_bgr.shape[0])
        mask_ruta_sel = self._seleccionar_componente(mask_ruta, ancla)
        if int(np.count_nonzero(mask_ruta_sel)) < 45:
            self._guardar_debug(frame_bgr.shape, roi_bgr, mask_ruta, mask_camion, mask_ruta_sel, ancla, None, self._estado_vacio())
            return self._estado_vacio()

        filas = self._extraer_filas(mask_ruta_sel, ancla[1])
        estado = self._clasificar(mask_ruta_sel, filas, ancla)
        self._guardar_debug(frame_bgr.shape, roi_bgr, mask_ruta, mask_camion, mask_ruta_sel, ancla, filas, estado)
        return estado

    def roi_debug(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self._debug or self._debug.get("frame_shape") != frame_bgr.shape:
            self.estimar(frame_bgr)

        roi_bgr = np.array(self._debug.get("roi_bgr"), copy=True)
        if roi_bgr.size == 0:
            return np.zeros((96, 96, 3), dtype=np.uint8)

        mask_ruta = self._debug.get("mask_ruta")
        mask_camion = self._debug.get("mask_camion")
        mask_sel = self._debug.get("mask_ruta_sel")
        ancla = self._debug.get("ancla")
        estado: EstadoRuta = self._debug.get("estado", self._estado_vacio())

        overlay = np.zeros_like(roi_bgr)
        if isinstance(mask_ruta, np.ndarray):
            overlay[mask_ruta > 0] = (0, 0, 180)
        if isinstance(mask_sel, np.ndarray):
            overlay[mask_sel > 0] = (0, 140, 255)
        if isinstance(mask_camion, np.ndarray):
            overlay[mask_camion > 0] = (0, 255, 0)
        dbg = cv2.addWeighted(roi_bgr, 0.70, overlay, 0.30, 0.0)
        if isinstance(ancla, tuple):
            cv2.circle(dbg, ancla, 4, (255, 255, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(dbg, ancla, 4, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        filas: tuple[_FilaRuta, ...] = tuple(self._debug.get("filas") or ())
        for fila in filas[:: max(1, len(filas) // 12)]:
            for x0, x1 in fila.segmentos:
                cv2.line(dbg, (x0, fila.y), (x1, fila.y), (255, 200, 0), 1, lineType=cv2.LINE_AA)

        texto = (
            f"{estado.maniobra.value} conf={estado.confianza:.2f} "
            f"ramal={estado.ramal_objetivo} cambio={int(estado.requiere_cambio_carril)}"
        )
        cv2.putText(dbg, texto, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(dbg, texto, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        return dbg

    def _estado_vacio(self) -> EstadoRuta:
        return EstadoRuta()

    def _recortar_roi(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        h, w = frame_bgr.shape[:2]
        x1 = int(round(w * self.roi[0]))
        y1 = int(round(h * self.roi[1]))
        x2 = int(round(w * self.roi[2]))
        y2 = int(round(h * self.roi[3]))
        x1 = max(0, min(x1, w))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(y1 + 1, min(y2, h))
        return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)

    @staticmethod
    def _mask_hsv(hsv: np.ndarray, rangos: tuple[tuple[int, int, int, int, int, int], ...]) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for h0, s0, v0, h1, s1, v1 in rangos:
            low = np.array([h0, s0, v0], dtype=np.uint8)
            high = np.array([h1, s1, v1], dtype=np.uint8)
            mask |= cv2.inRange(hsv, low, high)
        return mask

    def _estimar_ancla(self, mask_camion: np.ndarray, ancho: int, alto: int) -> tuple[int, int]:
        ys, xs = np.nonzero(mask_camion)
        if xs.size >= 20:
            return int(round(xs.mean())), int(round(ys.mean()))
        return (
            int(round(ancho * self.referencia_camion[0])),
            int(round(alto * self.referencia_camion[1])),
        )

    @staticmethod
    def _seleccionar_componente(mask_ruta: np.ndarray, ancla: tuple[int, int]) -> np.ndarray:
        n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_ruta, connectivity=8)
        if n_labels <= 1:
            return np.zeros_like(mask_ruta)

        mejor = 0
        mejor_score = float("inf")
        ax, ay = ancla
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 30:
                continue
            xs = np.where(labels == label)[1]
            ys = np.where(labels == label)[0]
            if xs.size == 0:
                continue
            dist = np.min((xs - ax) ** 2 + (ys - ay) ** 2)
            bottom_bias = max(0, ay - int(ys.max()))
            score = dist + 4.0 * (bottom_bias ** 2)
            if score < mejor_score:
                mejor_score = score
                mejor = label
        if mejor == 0:
            return np.zeros_like(mask_ruta)
        return np.where(labels == mejor, 255, 0).astype(np.uint8)

    def _extraer_filas(self, mask_ruta: np.ndarray, y_ancla: int) -> tuple[_FilaRuta, ...]:
        filas: list[_FilaRuta] = []
        y0 = max(0, int(round(y_ancla * 0.10)))
        y1 = min(mask_ruta.shape[0] - 1, y_ancla)
        for y in range(y0, y1 + 1):
            xs = np.flatnonzero(mask_ruta[y] > 0)
            if xs.size < 3:
                continue
            segmentos = self._segmentos_x(xs)
            if not segmentos:
                continue
            filas.append(_FilaRuta(y=y, segmentos=tuple(segmentos)))
        return tuple(filas)

    @staticmethod
    def _segmentos_x(xs: np.ndarray, max_gap: int = 6, min_ancho: int = 4) -> list[tuple[int, int]]:
        if xs.size == 0:
            return []
        segmentos: list[tuple[int, int]] = []
        ini = int(xs[0])
        prev = int(xs[0])
        for x in xs[1:]:
            x = int(x)
            if x - prev > max_gap:
                if prev - ini + 1 >= min_ancho:
                    segmentos.append((ini, prev))
                ini = x
            prev = x
        if prev - ini + 1 >= min_ancho:
            segmentos.append((ini, prev))
        return segmentos

    def _clasificar(
        self,
        mask_ruta: np.ndarray,
        filas: tuple[_FilaRuta, ...],
        ancla: tuple[int, int],
    ) -> EstadoRuta:
        if len(filas) < 8:
            return self._estado_vacio()

        width = max(1, mask_ruta.shape[1])
        anchor_x, anchor_y = ancla
        y_limite_superior = min(fila.y for fila in filas)
        y_limite_inferior = max(fila.y for fila in filas)
        altura_util = max(1, y_limite_inferior - y_limite_superior)
        filas_bottom = [f for f in filas if f.y >= y_limite_inferior - max(8, int(altura_util * 0.18))]
        filas_top = [f for f in filas if f.y <= y_limite_superior + max(12, int(altura_util * 0.22))]
        if not filas_bottom or not filas_top:
            return self._estado_vacio()

        bottom_center = np.mean([self._segmento_mas_cercano(f.segmentos, anchor_x) for f in filas_bottom])
        top_center = np.mean([self._segmento_mas_cercano(f.segmentos, bottom_center) for f in filas_top])
        offset_norm = float((top_center - bottom_center) / width)

        margen_inferior_ramal = max(8, int(round(altura_util * self.fraccion_inferior_ignorar_ramal)))
        filas_branch = [f for f in filas if f.y <= anchor_y - margen_inferior_ramal]
        branch_side: str | None = None
        branch_rows = 0
        branch_y: int | None = None
        for fila in filas_branch:
            if len(fila.segmentos) < 2:
                continue
            main_center = self._segmento_mas_cercano(fila.segmentos, bottom_center)
            side = self._lado_secundario(fila.segmentos, main_center, width)
            if side is None:
                continue
            branch_side = side if branch_side is None else branch_side
            if side == branch_side:
                branch_rows += 1
                if branch_y is None or fila.y < branch_y:
                    branch_y = fila.y

        maniobra = ManiobraRuta.SEGUIR_RECTO
        ramal = "centro"
        sesgo = 0.0
        requiere_cambio = False

        distancia_ramal = None
        if branch_y is not None:
            distancia_ramal = float(np.clip((anchor_y - branch_y) / max(1, mask_ruta.shape[0]), 0.0, 1.0))

        umbral_curva_mismo_lado = max(0.04, self.umbral_mantener * 0.5)
        curva_mismo_lado = (
            (branch_side == "der" and offset_norm >= umbral_curva_mismo_lado)
            or (branch_side == "izq" and offset_norm <= -umbral_curva_mismo_lado)
        )
        if branch_rows >= self.min_filas_ramal and branch_side == "der":
            if curva_mismo_lado:
                maniobra = ManiobraRuta.GIRO_DER
                ramal = "der"
                sesgo = 0.12
            else:
                if distancia_ramal is not None and distancia_ramal <= self.umbral_distancia_salida:
                    maniobra = ManiobraRuta.SALIDA_DER
                    sesgo = 0.35
                else:
                    maniobra = ManiobraRuta.MANTENER_DER
                    sesgo = 0.18
                ramal = "der"
                requiere_cambio = True
        elif branch_rows >= self.min_filas_ramal and branch_side == "izq":
            if curva_mismo_lado:
                maniobra = ManiobraRuta.GIRO_IZQ
                ramal = "izq"
                sesgo = -0.12
            else:
                if distancia_ramal is not None and distancia_ramal <= self.umbral_distancia_salida:
                    maniobra = ManiobraRuta.SALIDA_IZQ
                    sesgo = -0.35
                else:
                    maniobra = ManiobraRuta.MANTENER_IZQ
                    sesgo = -0.18
                ramal = "izq"
                requiere_cambio = True
        elif offset_norm >= self.umbral_giro_fuerte:
            maniobra = ManiobraRuta.GIRO_DER
            ramal = "der"
            sesgo = 0.30
        elif offset_norm <= -self.umbral_giro_fuerte:
            maniobra = ManiobraRuta.GIRO_IZQ
            ramal = "izq"
            sesgo = -0.30
        elif offset_norm >= self.umbral_mantener:
            maniobra = ManiobraRuta.GIRO_DER
            ramal = "der"
            sesgo = 0.12
        elif offset_norm <= -self.umbral_mantener:
            maniobra = ManiobraRuta.GIRO_IZQ
            ramal = "izq"
            sesgo = -0.12

        row_factor = min(1.0, len(filas) / max(10.0, anchor_y * 0.28))
        px_factor = min(1.0, float(np.count_nonzero(mask_ruta)) / (mask_ruta.size * 0.045))
        geom_factor = 0.40 + min(0.60, abs(offset_norm) * 2.0 + branch_rows * 0.10)
        confianza = float(np.clip(0.40 * row_factor + 0.35 * px_factor + 0.25 * geom_factor, 0.0, 1.0))
        if maniobra is ManiobraRuta.SEGUIR_RECTO:
            confianza = max(confianza, 0.55)

        if confianza < self.min_confianza:
            return self._estado_vacio()

        distancia = None
        if branch_y is not None:
            distancia = distancia_ramal
        elif maniobra is not ManiobraRuta.SEGUIR_RECTO:
            distancia = float(np.clip((anchor_y - y_limite_superior) / max(1, mask_ruta.shape[0]), 0.0, 1.0))

        return EstadoRuta(
            visible=True,
            confianza=confianza,
            maniobra=maniobra,
            distancia_normalizada=distancia,
            sesgo_lateral_objetivo=sesgo,
            ramal_objetivo=ramal,
            requiere_cambio_carril=requiere_cambio,
        )

    @staticmethod
    def _segmento_mas_cercano(segmentos: tuple[tuple[int, int], ...], x_ref: float) -> float:
        centros = [((x0 + x1) * 0.5) for x0, x1 in segmentos]
        idx = int(np.argmin([abs(c - x_ref) for c in centros]))
        return float(centros[idx])

    @staticmethod
    def _lado_secundario(segmentos: tuple[tuple[int, int], ...], main_center: float, width: int) -> str | None:
        secundarios = []
        for x0, x1 in segmentos:
            centro = (x0 + x1) * 0.5
            delta = centro - main_center
            if abs(delta) < width * 0.10:
                continue
            secundarios.append(delta)
        if not secundarios:
            return None
        delta = max(secundarios, key=abs)
        return "der" if delta > 0 else "izq"

    def _guardar_debug(
        self,
        frame_shape: tuple[int, ...],
        roi_bgr: np.ndarray,
        mask_ruta: np.ndarray,
        mask_camion: np.ndarray,
        mask_ruta_sel: np.ndarray,
        ancla: tuple[int, int],
        filas: tuple[_FilaRuta, ...] | None,
        estado: EstadoRuta,
    ) -> None:
        self._debug = {
            "frame_shape": frame_shape,
            "roi_bgr": roi_bgr,
            "mask_ruta": mask_ruta,
            "mask_camion": mask_camion,
            "mask_ruta_sel": mask_ruta_sel,
            "ancla": ancla,
            "filas": filas,
            "estado": estado,
        }
