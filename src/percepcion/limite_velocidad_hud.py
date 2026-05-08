"""Lectura del limite de velocidad mostrado en el HUD de ETS2."""

from __future__ import annotations

import cv2
import numpy as np

from src.tipos import EstadoLimiteVelocidadHUD

_DEFAULT_ROI = (0.742, 0.730, 0.842, 0.948)
_DEFAULT_SIZE = (96, 96)
_DEFAULT_HSV_RED = (
    (0, 90, 120, 15, 255, 255),
    (160, 90, 120, 179, 255, 255),
)
_DEFAULT_HSV_WHITE = (
    (0, 0, 145, 179, 80, 255),
)
_SIZE_DIGITO = (16, 24)
_MIN_AREA_SIGNO = 180
_MIN_AREA_DIGITO = 28
_MIN_CONF_DIGITO = 0.38


class EstimadorLimiteVelocidadHUD:
    def __init__(
        self,
        roi: tuple[float, float, float, float] = _DEFAULT_ROI,
        hsv_borde_rojo: tuple[tuple[int, int, int, int, int, int], ...] = _DEFAULT_HSV_RED,
        hsv_fondo_blanco: tuple[tuple[int, int, int, int, int, int], ...] = _DEFAULT_HSV_WHITE,
        min_confianza: float = 0.48,
        tam_signo: tuple[int, int] = _DEFAULT_SIZE,
    ):
        self.roi = tuple(float(v) for v in roi)
        self.hsv_borde_rojo = tuple(tuple(int(v) for v in r) for r in hsv_borde_rojo)
        self.hsv_fondo_blanco = tuple(tuple(int(v) for v in r) for r in hsv_fondo_blanco)
        self.min_confianza = float(min_confianza)
        self.tam_signo = tuple(int(v) for v in tam_signo)
        self._plantillas = {d: self._crear_plantilla(d) for d in range(10)}
        self._plantillas_limite = {
            limite: self._crear_plantilla_limite(limite)
            for limite in (30, 40, 50, 60, 70, 80, 90, 100, 110, 120)
        }
        self._debug: dict[str, object] = {}

    def estimar(self, frame_bgr: np.ndarray) -> EstadoLimiteVelocidadHUD:
        roi_bgr, _ = self._recortar_roi(frame_bgr)
        if roi_bgr.size == 0:
            return self._estado_vacio()

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask_red = self._mask_hsv(hsv, self.hsv_borde_rojo)
        mask_white = self._mask_hsv(hsv, self.hsv_fondo_blanco)
        recorte = self._recortar_signo(roi_bgr, mask_red, mask_white)
        if recorte is None:
            self._guardar_debug(frame_bgr.shape, roi_bgr, mask_red, mask_white, None, None, None, self._estado_vacio())
            return self._estado_vacio()

        signo_bgr = cv2.resize(recorte, self.tam_signo, interpolation=cv2.INTER_LINEAR)
        hsv_signo = cv2.cvtColor(signo_bgr, cv2.COLOR_BGR2HSV)
        mask_white_signo = self._mask_hsv(hsv_signo, self.hsv_fondo_blanco)
        mask_white_signo = cv2.morphologyEx(mask_white_signo, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if int(np.count_nonzero(mask_white_signo)) < _MIN_AREA_SIGNO:
            estado = EstadoLimiteVelocidadHUD(visible=True, confianza=0.0, limite_kmh=None)
            self._guardar_debug(frame_bgr.shape, roi_bgr, mask_red, mask_white, signo_bgr, mask_white_signo, None, estado)
            return estado

        digit_mask = self._extraer_digitos(signo_bgr, mask_white_signo)
        componentes = self._extraer_componentes(digit_mask)
        if not componentes:
            limite, conf = self._clasificar_signo_completo(digit_mask)
            estado = EstadoLimiteVelocidadHUD(
                visible=True,
                confianza=conf,
                limite_kmh=limite if conf >= self.min_confianza else None,
            )
            self._guardar_debug(frame_bgr.shape, roi_bgr, mask_red, mask_white, signo_bgr, mask_white_signo, digit_mask, estado)
            return estado

        componentes_validos: list[np.ndarray] = []
        for x, _y, w, _h, comp in componentes:
            digito, conf = self._clasificar_digito(comp)
            if conf < _MIN_CONF_DIGITO:
                continue
            componentes_validos.append(comp)

        if not componentes_validos:
            limite, conf = self._clasificar_signo_completo(digit_mask)
            estado = EstadoLimiteVelocidadHUD(
                visible=True,
                confianza=conf,
                limite_kmh=limite if conf >= self.min_confianza else None,
            )
            self._guardar_debug(frame_bgr.shape, roi_bgr, mask_red, mask_white, signo_bgr, mask_white_signo, digit_mask, estado, componentes)
            return estado

        limite, confianza = self._clasificar_limite_por_componentes(componentes_validos)
        limite_tpl, conf_tpl = self._clasificar_signo_completo(digit_mask)
        if limite is None and limite_tpl is not None:
            limite = limite_tpl
            confianza = conf_tpl
        elif limite_tpl is not None and confianza < self.min_confianza and conf_tpl >= confianza + 0.03:
            limite = limite_tpl
            confianza = conf_tpl
        elif confianza < self.min_confianza and conf_tpl > confianza:
            limite = limite_tpl
            confianza = conf_tpl
        estado = EstadoLimiteVelocidadHUD(
            visible=True,
            confianza=confianza,
            limite_kmh=limite if confianza >= self.min_confianza else None,
        )
        self._guardar_debug(frame_bgr.shape, roi_bgr, mask_red, mask_white, signo_bgr, mask_white_signo, digit_mask, estado, componentes)
        return estado

    def roi_debug(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self._debug or self._debug.get("frame_shape") != frame_bgr.shape:
            self.estimar(frame_bgr)

        roi_bgr = np.array(self._debug.get("roi_bgr"), copy=True)
        if roi_bgr.size == 0:
            return np.zeros((96, 96, 3), dtype=np.uint8)

        dbg = roi_bgr
        signo_bgr = self._debug.get("signo_bgr")
        if isinstance(signo_bgr, np.ndarray):
            signo = np.array(signo_bgr, copy=True)
            componentes = self._debug.get("componentes") or []
            for x, y, w, h, _comp in componentes:
                cv2.rectangle(signo, (x, y), (x + w - 1, y + h - 1), (255, 120, 0), 1, cv2.LINE_AA)
            estado: EstadoLimiteVelocidadHUD = self._debug.get("estado", self._estado_vacio())
            texto = f"lim={estado.limite_kmh if estado.limite_kmh is not None else '-'} conf={estado.confianza:.2f}"
            cv2.putText(signo, texto, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(signo, texto, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
            signo = cv2.resize(signo, (dbg.shape[1], dbg.shape[0]), interpolation=cv2.INTER_NEAREST)
            dbg = np.hstack([dbg, signo])
        return dbg

    def _estado_vacio(self) -> EstadoLimiteVelocidadHUD:
        return EstadoLimiteVelocidadHUD()

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

    @staticmethod
    def _recortar_signo(
        roi_bgr: np.ndarray,
        mask_red: np.ndarray,
        mask_white: np.ndarray,
    ) -> np.ndarray | None:
        n_red, labels_red, stats_red, _ = cv2.connectedComponentsWithStats(mask_red, connectivity=8)
        mejor_rojo = 0
        mejor_area_roja = 0
        for label in range(1, n_red):
            area = int(stats_red[label, cv2.CC_STAT_AREA])
            if area > mejor_area_roja:
                mejor_area_roja = area
                mejor_rojo = label
        if mejor_rojo != 0 and mejor_area_roja >= 40:
            x = int(stats_red[mejor_rojo, cv2.CC_STAT_LEFT])
            y = int(stats_red[mejor_rojo, cv2.CC_STAT_TOP])
            w = int(stats_red[mejor_rojo, cv2.CC_STAT_WIDTH])
            h = int(stats_red[mejor_rojo, cv2.CC_STAT_HEIGHT])
            cx = x + w * 0.5
            cy = y + h * 0.5
            half = max(w, h) * 0.5 + 8.0
            x0 = max(0, int(round(cx - half)))
            y0 = max(0, int(round(cy - half)))
            x1 = min(roi_bgr.shape[1], int(round(cx + half)))
            y1 = min(roi_bgr.shape[0], int(round(cy + half)))
            return roi_bgr[y0:y1, x0:x1]

        mask_signo = cv2.morphologyEx(mask_red | mask_white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_signo, connectivity=8)
        if n_labels <= 1:
            return None
        mejor = 0
        mejor_area = 0
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area > mejor_area:
                mejor_area = area
                mejor = label
        if mejor == 0 or mejor_area < _MIN_AREA_SIGNO:
            return None
        x = int(stats[mejor, cv2.CC_STAT_LEFT])
        y = int(stats[mejor, cv2.CC_STAT_TOP])
        w = int(stats[mejor, cv2.CC_STAT_WIDTH])
        h = int(stats[mejor, cv2.CC_STAT_HEIGHT])
        pad = 4
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(roi_bgr.shape[1], x + w + pad)
        y1 = min(roi_bgr.shape[0], y + h + pad)
        return roi_bgr[y0:y1, x0:x1]

    @staticmethod
    def _extraer_digitos(signo_bgr: np.ndarray, mask_white: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(signo_bgr, cv2.COLOR_BGR2GRAY)
        interior = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        interior = cv2.erode(interior, np.ones((5, 5), np.uint8), iterations=1)
        alto, ancho = interior.shape
        circle = np.zeros_like(interior)
        radio = max(8, int(round(min(alto, ancho) * 0.34)))
        cv2.circle(circle, (ancho // 2, alto // 2), radio, 255, thickness=-1, lineType=cv2.LINE_AA)
        interior = cv2.bitwise_and(interior, circle)
        interior_vals = gray[interior > 0]
        if interior_vals.size == 0:
            return np.zeros_like(gray)
        thr = int(np.clip(np.mean(interior_vals) - 55, 45, 170))
        mask = np.where((gray <= thr) & (interior > 0), 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        return mask

    def _extraer_componentes(self, mask: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
        perfil = np.count_nonzero(mask > 0, axis=0)
        xs = np.flatnonzero(perfil > 0)
        if xs.size == 0:
            return []

        rangos: list[tuple[int, int]] = []
        ini = int(xs[0])
        prev = int(xs[0])
        for x in xs[1:]:
            x = int(x)
            if x - prev > 2:
                rangos.append((ini, prev + 1))
                ini = x
            prev = x
        rangos.append((ini, prev + 1))

        comps: list[tuple[int, int, int, int, np.ndarray]] = []
        for x0, x1 in rangos:
            banda = mask[:, x0:x1]
            ys, xs_band = np.nonzero(banda)
            if xs_band.size == 0:
                continue
            y0 = int(ys.min())
            y1 = int(ys.max()) + 1
            crop = banda[y0:y1, :]
            area = int(np.count_nonzero(crop))
            if area < _MIN_AREA_DIGITO or crop.shape[1] < 4 or crop.shape[0] < 10:
                continue
            comps.append((x0, y0, x1 - x0, y1 - y0, crop))

        if len(comps) == 1 and comps[0][2] >= int(mask.shape[1] * 0.46):
            split = self._split_component(comps[0][4], comps[0][0], comps[0][1])
            if split is not None:
                comps = split
        return comps

    def _split_component(
        self,
        comp: np.ndarray,
        x_off: int,
        y_off: int,
    ) -> list[tuple[int, int, int, int, np.ndarray]] | None:
        perfil = np.count_nonzero(comp > 0, axis=0)
        if perfil.size < 12:
            return None
        ini = max(3, perfil.size // 4)
        fin = min(perfil.size - 3, (perfil.size * 3) // 4)
        if fin <= ini:
            return None
        i_split = int(np.argmin(perfil[ini:fin])) + ini
        if perfil[i_split] > max(1, int(perfil.max() * 0.30)):
            return None
        izquierda = comp[:, :i_split]
        derecha = comp[:, i_split:]
        partes = []
        for dx, parte in ((0, izquierda), (i_split, derecha)):
            ys, xs = np.nonzero(parte)
            if xs.size == 0:
                continue
            x0 = int(xs.min())
            y0 = int(ys.min())
            x1 = int(xs.max()) + 1
            y1 = int(ys.max()) + 1
            crop = parte[y0:y1, x0:x1]
            if crop.size == 0 or int(np.count_nonzero(crop)) < _MIN_AREA_DIGITO:
                continue
            partes.append((x_off + dx + x0, y_off + y0, x1 - x0, y1 - y0, crop))
        return partes if len(partes) >= 2 else None

    def _clasificar_digito(self, comp: np.ndarray) -> tuple[int, float]:
        mejor_digito = 0
        mejor_score = -1.0
        for digito in self._plantillas:
            score = self._score_digito(comp, digito)
            if score > mejor_score:
                mejor_score = score
                mejor_digito = digito
        return mejor_digito, float(np.clip(mejor_score, 0.0, 1.0))

    def _clasificar_limite_por_componentes(self, componentes: list[np.ndarray]) -> tuple[int | None, float]:
        if not componentes:
            return None, 0.0
        candidatos = [
            limite
            for limite in self._plantillas_limite
            if len(str(limite)) == len(componentes)
        ]
        if not candidatos:
            return None, 0.0

        mejor_limite = None
        mejor_score = -1.0
        for limite in candidatos:
            scores = [
                self._score_digito(comp, int(char))
                for comp, char in zip(componentes, str(limite))
            ]
            score = float(np.mean(scores))
            if score > mejor_score:
                mejor_score = score
                mejor_limite = limite
        return mejor_limite, float(np.clip(mejor_score, 0.0, 1.0))

    def _score_digito(self, comp: np.ndarray, digito: int) -> float:
        norm = cv2.resize(comp, _SIZE_DIGITO, interpolation=cv2.INTER_NEAREST)
        norm = np.where(norm > 0, 255, 0).astype(np.uint8)
        plantilla = self._plantillas[digito]
        pixel = 1.0 - float(np.mean(np.abs(norm.astype(np.float32) - plantilla.astype(np.float32))) / 255.0)
        inter = np.count_nonzero((norm > 0) & (plantilla > 0))
        union = np.count_nonzero((norm > 0) | (plantilla > 0))
        iou = float(inter / union) if union else 0.0
        score = 0.55 * pixel + 0.45 * iou
        huecos = self._contar_huecos(norm)
        esperado = self._huecos_esperados(digito)
        if huecos == esperado:
            score += 0.08
        else:
            score -= 0.10 * abs(huecos - esperado)
        return float(score)

    @staticmethod
    def _huecos_esperados(digito: int) -> int:
        if digito == 8:
            return 2
        if digito in {0, 6, 9}:
            return 1
        return 0

    @staticmethod
    def _contar_huecos(mask: np.ndarray) -> int:
        if mask.size == 0:
            return 0
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        inv = np.where(padded == 0, 255, 0).astype(np.uint8)
        n_labels, labels, _stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        huecos = 0
        for label in range(1, n_labels):
            ys, xs = np.nonzero(labels == label)
            if xs.size == 0:
                continue
            if xs.min() == 0 or ys.min() == 0 or xs.max() == inv.shape[1] - 1 or ys.max() == inv.shape[0] - 1:
                continue
            huecos += 1
        return huecos

    @staticmethod
    def _crear_plantilla(digito: int) -> np.ndarray:
        canvas = np.full((40, 28), 245, dtype=np.uint8)
        txt = str(digito)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.95
        thickness = 2
        (tw, th), _ = cv2.getTextSize(txt, font, scale, thickness)
        org = ((canvas.shape[1] - tw) // 2, (canvas.shape[0] + th) // 2 - 3)
        cv2.putText(canvas, txt, org, font, scale, 20, thickness, cv2.LINE_AA)
        mask = np.where(canvas <= 150, 255, 0).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return np.zeros((_SIZE_DIGITO[1], _SIZE_DIGITO[0]), dtype=np.uint8)
        x0 = int(xs.min())
        y0 = int(ys.min())
        x1 = int(xs.max()) + 1
        y1 = int(ys.max()) + 1
        crop = mask[y0:y1, x0:x1]
        return cv2.resize(crop, _SIZE_DIGITO, interpolation=cv2.INTER_NEAREST)

    def _clasificar_signo_completo(self, digit_mask: np.ndarray) -> tuple[int | None, float]:
        digit_mask = np.where(digit_mask > 0, 255, 0).astype(np.uint8)
        best_limit = None
        best_score = -1.0
        for limite, tpl in self._plantillas_limite.items():
            pixel = 1.0 - float(np.mean(np.abs(digit_mask.astype(np.float32) - tpl.astype(np.float32))) / 255.0)
            inter = np.count_nonzero((digit_mask > 0) & (tpl > 0))
            union = np.count_nonzero((digit_mask > 0) | (tpl > 0))
            iou = float(inter / union) if union else 0.0
            score = 0.45 * pixel + 0.55 * iou
            if score > best_score:
                best_score = score
                best_limit = limite
        return best_limit, float(np.clip(best_score, 0.0, 1.0))

    def _crear_plantilla_limite(self, limite: int) -> np.ndarray:
        w, h = self.tam_signo
        canvas = np.full((h, w, 3), 30, dtype=np.uint8)
        cx = w // 2
        cy = h // 2
        r = min(h, w) // 2 - 6
        cv2.circle(canvas, (cx, cy), r, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), r - 8, (245, 245, 245), thickness=-1, lineType=cv2.LINE_AA)
        texto = str(limite)
        escala = 0.95 if len(texto) == 2 else 0.75
        grosor = 2
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
        org = (cx - tw // 2, cy + th // 2)
        cv2.putText(
            canvas,
            texto,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            escala,
            (25, 25, 25),
            grosor,
            cv2.LINE_AA,
        )
        mask_white = self._mask_hsv(cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV), self.hsv_fondo_blanco)
        return self._extraer_digitos(canvas, mask_white)

    def _guardar_debug(
        self,
        frame_shape: tuple[int, ...],
        roi_bgr: np.ndarray,
        mask_red: np.ndarray,
        mask_white: np.ndarray,
        signo_bgr: np.ndarray | None,
        mask_white_signo: np.ndarray | None,
        digit_mask: np.ndarray | None,
        estado: EstadoLimiteVelocidadHUD,
        componentes: list[tuple[int, int, int, int, np.ndarray]] | None = None,
    ) -> None:
        self._debug = {
            "frame_shape": frame_shape,
            "roi_bgr": roi_bgr,
            "mask_red": mask_red,
            "mask_white": mask_white,
            "signo_bgr": signo_bgr,
            "mask_white_signo": mask_white_signo,
            "digit_mask": digit_mask,
            "estado": estado,
            "componentes": componentes or [],
        }
