"""Análisis de carriles sobre la máscara de YOLOP.

Clasifica cada línea detectada como:
  - ego_izq / ego_der  → bordes del carril del propio camión
  - contrario          → carriles del sentido opuesto (a la izquierda del ego)
  - mismo_sentido      → otros carriles del mismo sentido (a la derecha del ego)

Calcula offset del centro del carril respecto al centro de imagen y radio de
curvatura aproximado, e incluye un rastreador EMA para suavizar los polinomios
del ego entre frames (necesario en video).

Compatible con Python 3.9+ — usa `from __future__ import annotations` para que
las firmas con `int | None` no fallen al ejecutar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ────────────────────────────────────────────────────────────
# Datos
# ────────────────────────────────────────────────────────────
@dataclass
class Linea:
    coefs: np.ndarray  # polinomio cuadrático x = f(y)
    y_min: int
    y_max: int
    x_eval: float      # x evaluado cerca del coche (para ordenar/clasificar)
    n_pixeles: int = 0


@dataclass
class CarrilesClasificados:
    ego_izq: Optional[Linea] = None
    ego_der: Optional[Linea] = None
    contrario: List[Linea] = field(default_factory=list)
    mismo_sentido: List[Linea] = field(default_factory=list)
    offset_px: Optional[float] = None
    curvatura_px: Optional[float] = None

    @property
    def estado(self) -> str:
        if self.ego_izq is not None and self.ego_der is not None:
            return "OK"
        if self.ego_izq is not None or self.ego_der is not None:
            return "PARCIAL"
        return "NO_DETECTADO"


# ────────────────────────────────────────────────────────────
# Rastreador EMA (para video)
# ────────────────────────────────────────────────────────────
class RastreadorEgo:
    """EMA sobre coeficientes polinómicos del ego para reducir parpadeo."""

    def __init__(self, alpha: float = 0.4, max_frames_perdidos: int = 6):
        self.alpha = alpha
        self.max_frames_perdidos = max_frames_perdidos
        self.izq: Optional[Linea] = None
        self.der: Optional[Linea] = None
        self._perdidos_izq = 0
        self._perdidos_der = 0

    def _suavizar(self, viejo: Optional[Linea], nuevo: Optional[Linea]) -> Optional[Linea]:
        if nuevo is None:
            return viejo
        if viejo is None:
            return nuevo
        coefs = self.alpha * nuevo.coefs + (1.0 - self.alpha) * viejo.coefs
        return Linea(
            coefs=coefs,
            y_min=nuevo.y_min,
            y_max=nuevo.y_max,
            x_eval=nuevo.x_eval,
            n_pixeles=nuevo.n_pixeles,
        )

    def actualizar(self, izq: Optional[Linea], der: Optional[Linea]) -> Tuple[Optional[Linea], Optional[Linea]]:
        if izq is None:
            self._perdidos_izq += 1
            if self._perdidos_izq > self.max_frames_perdidos:
                self.izq = None
        else:
            self._perdidos_izq = 0
            self.izq = self._suavizar(self.izq, izq)
        if der is None:
            self._perdidos_der += 1
            if self._perdidos_der > self.max_frames_perdidos:
                self.der = None
        else:
            self._perdidos_der = 0
            self.der = self._suavizar(self.der, der)
        return self.izq, self.der

    def reset(self) -> None:
        self.izq = None
        self.der = None
        self._perdidos_izq = 0
        self._perdidos_der = 0


# ────────────────────────────────────────────────────────────
# Analizador
# ────────────────────────────────────────────────────────────
class AnalizadorCarriles:
    """Convierte la máscara de YOLOP en clasificación ego/contrario/mismo."""

    def __init__(
        self,
        roi_top_frac: float = 0.42,
        min_pixeles_linea: int = 60,
        min_rango_y: int = 25,
        merge_dx: int = 55,
        max_lineas: int = 6,
        dilatar_area: int = 25,
        usar_suavizado: bool = True,
        ema_alpha: float = 0.4,
        max_frames_perdidos: int = 6,
    ):
        self.roi_top_frac = roi_top_frac
        self.min_pixeles_linea = min_pixeles_linea
        self.min_rango_y = min_rango_y
        self.merge_dx = merge_dx
        self.max_lineas = max_lineas
        self.dilatar_area = dilatar_area
        self._rastreador: Optional[RastreadorEgo] = (
            RastreadorEgo(ema_alpha, max_frames_perdidos) if usar_suavizado else None
        )

    def reset(self) -> None:
        if self._rastreador is not None:
            self._rastreador.reset()

    # ── Pipeline ────────────────────────────────────────────
    def _filtrar(self, lineas_mask: np.ndarray, area_mask: Optional[np.ndarray]) -> np.ndarray:
        h = lineas_mask.shape[0]
        roi = np.zeros_like(lineas_mask)
        roi[int(h * self.roi_top_frac):, :] = 1
        out = lineas_mask * roi
        if area_mask is not None and self.dilatar_area > 0 and area_mask.any():
            kernel = np.ones((self.dilatar_area, self.dilatar_area), np.uint8)
            area_dil = cv2.dilate(area_mask, kernel)
            out = out * area_dil
        return out

    def _ajustar(self, ys: np.ndarray, xs: np.ndarray, h: int) -> Optional[Linea]:
        if len(ys) < self.min_pixeles_linea:
            return None
        if int(ys.max()) - int(ys.min()) < self.min_rango_y:
            return None
        try:
            coefs = np.polyfit(ys, xs, deg=2)
        except (np.linalg.LinAlgError, ValueError):
            return None
        y_eval = int(h * 0.9)
        return Linea(
            coefs=coefs,
            y_min=int(ys.min()),
            y_max=int(ys.max()),
            x_eval=float(np.polyval(coefs, y_eval)),
            n_pixeles=int(len(ys)),
        )

    def _detectar(self, mascara: np.ndarray, h: int) -> List[Tuple[Linea, np.ndarray, np.ndarray]]:
        num, labels = cv2.connectedComponents(mascara.astype(np.uint8))
        salida = []
        for lbl in range(1, num):
            ys, xs = np.where(labels == lbl)
            ln = self._ajustar(ys, xs, h)
            if ln is not None:
                salida.append((ln, ys, xs))
        return salida

    def _fusionar(self, lineas_pts: List[Tuple[Linea, np.ndarray, np.ndarray]],
                  h: int) -> List[Linea]:
        """Merge componentes con x_eval cercano (líneas discontinuas)."""
        if not lineas_pts:
            return []
        lineas_pts.sort(key=lambda lp: lp[0].x_eval)
        grupos: List[List[Tuple[Linea, np.ndarray, np.ndarray]]] = [[lineas_pts[0]]]
        for lp in lineas_pts[1:]:
            if abs(lp[0].x_eval - grupos[-1][-1][0].x_eval) < self.merge_dx:
                grupos[-1].append(lp)
            else:
                grupos.append([lp])
        fusionadas: List[Linea] = []
        for g in grupos:
            if len(g) == 1:
                fusionadas.append(g[0][0])
                continue
            ys = np.concatenate([gg[1] for gg in g])
            xs = np.concatenate([gg[2] for gg in g])
            ln = self._ajustar(ys, xs, h)
            if ln is not None:
                fusionadas.append(ln)
        return fusionadas[:self.max_lineas]

    @staticmethod
    def _clasificar(lineas: List[Linea], w: int) -> Tuple[
        Optional[Linea], Optional[Linea], List[Linea], List[Linea]
    ]:
        centro = w / 2
        izq = [l for l in lineas if l.x_eval < centro]
        der = [l for l in lineas if l.x_eval >= centro]
        ego_izq = izq[-1] if izq else None
        ego_der = der[0] if der else None
        contrario = izq[:-1] if len(izq) > 1 else []
        mismo = der[1:] if len(der) > 1 else []
        return ego_izq, ego_der, contrario, mismo

    @staticmethod
    def _offset(ego_izq: Optional[Linea], ego_der: Optional[Linea],
                h: int, w: int) -> Optional[float]:
        if ego_izq is None or ego_der is None:
            return None
        y = h - 1
        xi = float(np.polyval(ego_izq.coefs, y))
        xd = float(np.polyval(ego_der.coefs, y))
        return ((xi + xd) / 2.0) - (w / 2.0)

    @staticmethod
    def _curvatura(linea: Optional[Linea], h: int) -> Optional[float]:
        if linea is None:
            return None
        a, b, _ = linea.coefs
        if abs(a) < 1e-6:
            return None
        y = h - 1
        return ((1.0 + (2.0 * a * y + b) ** 2) ** 1.5) / abs(2.0 * a)

    def analizar(
        self,
        lineas_mask: np.ndarray,
        area_mask: Optional[np.ndarray] = None,
    ) -> CarrilesClasificados:
        h, w = lineas_mask.shape
        mask = self._filtrar(lineas_mask, area_mask)
        lineas_pts = self._detectar(mask, h)
        lineas = self._fusionar(lineas_pts, h)
        ego_izq, ego_der, contrario, mismo = self._clasificar(lineas, w)

        if self._rastreador is not None:
            ego_izq, ego_der = self._rastreador.actualizar(ego_izq, ego_der)

        return CarrilesClasificados(
            ego_izq=ego_izq,
            ego_der=ego_der,
            contrario=contrario,
            mismo_sentido=mismo,
            offset_px=self._offset(ego_izq, ego_der, h, w),
            curvatura_px=self._curvatura(ego_izq or ego_der, h),
        )


# ────────────────────────────────────────────────────────────
# Render
# ────────────────────────────────────────────────────────────
COLOR_EGO = (0, 220, 0)
COLOR_CONTRARIO = (0, 0, 255)
COLOR_MISMO = (0, 165, 255)
COLOR_BORDE_EGO = (255, 255, 0)
COLOR_HUD_BG = (28, 28, 28)
COLOR_HUD_TXT = (240, 240, 240)


def _polilinea(linea: Linea, y_max: Optional[int] = None) -> np.ndarray:
    y0 = linea.y_min
    y1 = linea.y_max if y_max is None else y_max
    ys = np.arange(y0, y1 + 1)
    xs = np.polyval(linea.coefs, ys).astype(np.int32)
    return np.stack([xs, ys], axis=1)


def _dibujar_linea(img: np.ndarray, linea: Linea, color, grosor: int = 5,
                   extender_a: Optional[int] = None) -> None:
    pts = _polilinea(linea, y_max=extender_a).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=grosor)


def _rellenar_carril(img: np.ndarray, izq: Optional[Linea], der: Optional[Linea],
                     color, alpha: float = 0.30,
                     extender_a: Optional[int] = None) -> None:
    if izq is None or der is None:
        return
    y0 = max(izq.y_min, der.y_min)
    y1_real = min(izq.y_max, der.y_max)
    y1 = extender_a if extender_a is not None else y1_real
    if y1 <= y0:
        return
    ys = np.arange(y0, y1 + 1)
    xs_i = np.polyval(izq.coefs, ys).astype(np.int32)
    xs_d = np.polyval(der.coefs, ys).astype(np.int32)
    poly = np.concatenate([
        np.stack([xs_i, ys], axis=1),
        np.stack([xs_d, ys], axis=1)[::-1],
    ]).reshape(-1, 1, 2)
    overlay = img.copy()
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)


def _etiqueta(img: np.ndarray, x: int, y: int, texto: str, color) -> None:
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    x = int(np.clip(x - tw // 2, 5, img.shape[1] - tw - 5))
    cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 6, y + 6), (0, 0, 0), -1)
    cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def _hud(img: np.ndarray, info: dict) -> None:
    lineas = []
    estado = info.get("estado", "?")
    color_estado = (0, 220, 0) if estado == "OK" else \
                   (0, 165, 255) if estado == "PARCIAL" else (0, 0, 255)
    lineas.append(("Carril ego: " + estado, color_estado))
    if info.get("offset") is not None:
        off = info["offset"]
        lado = "DER" if off > 0 else "IZQ"
        lineas.append((f"Offset: {abs(off):4.0f}px {lado}", COLOR_HUD_TXT))
    if info.get("curvatura") is not None:
        lineas.append((f"Curvatura: ~{info['curvatura']:.0f}px", COLOR_HUD_TXT))
    if info.get("fps") is not None:
        lineas.append((f"FPS: {info['fps']:.1f}", COLOR_HUD_TXT))
    if info.get("frame") is not None:
        lineas.append((f"Frame: {info['frame']}", COLOR_HUD_TXT))

    pad = 10
    line_h = 26
    box_h = pad * 2 + line_h * len(lineas)
    box_w = 280
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.70, img, 0.30, 0, dst=img)
    for i, (txt, color) in enumerate(lineas):
        cv2.putText(img, txt, (20, 32 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _indicador_offset(img: np.ndarray, offset_px: float, w: int) -> None:
    bar_w = 240
    bar_h = 8
    x0 = w // 2 - bar_w // 2
    y0 = 14
    cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), -1)
    cv2.line(img, (x0 + bar_w // 2, y0 - 3), (x0 + bar_w // 2, y0 + bar_h + 3),
             (200, 200, 200), 1)
    pos = int(np.clip(bar_w // 2 + offset_px / 3, 4, bar_w - 4))
    color = (0, 220, 0) if abs(offset_px) < 60 else \
            (0, 165, 255) if abs(offset_px) < 120 else (0, 0, 255)
    cv2.circle(img, (x0 + pos, y0 + bar_h // 2), 7, color, -1)


def superponer_carriles(
    frame: np.ndarray,
    carriles: CarrilesClasificados,
    area_mask: Optional[np.ndarray] = None,
    mostrar_hud: bool = True,
    fps: Optional[float] = None,
    frame_idx: Optional[int] = None,
) -> np.ndarray:
    """Devuelve una copia del frame con la clasificación pintada encima."""
    h, w = frame.shape[:2]
    out = frame.copy()

    if area_mask is not None and area_mask.any():
        overlay = out.copy()
        overlay[area_mask > 0] = (60, 200, 60)
        cv2.addWeighted(overlay, 0.15, out, 0.85, 0, dst=out)

    extender = h - 1

    _rellenar_carril(out, carriles.ego_izq, carriles.ego_der, COLOR_EGO, 0.32, extender)
    if carriles.contrario and carriles.ego_izq is not None:
        _rellenar_carril(out, carriles.contrario[-1], carriles.ego_izq,
                         COLOR_CONTRARIO, 0.28, extender)
    for i in range(len(carriles.contrario) - 1):
        _rellenar_carril(out, carriles.contrario[i], carriles.contrario[i + 1],
                         COLOR_CONTRARIO, 0.28, extender)
    if carriles.ego_der is not None and carriles.mismo_sentido:
        _rellenar_carril(out, carriles.ego_der, carriles.mismo_sentido[0],
                         COLOR_MISMO, 0.28, extender)
    for i in range(len(carriles.mismo_sentido) - 1):
        _rellenar_carril(out, carriles.mismo_sentido[i], carriles.mismo_sentido[i + 1],
                         COLOR_MISMO, 0.28, extender)

    for l in carriles.contrario:
        _dibujar_linea(out, l, COLOR_CONTRARIO, 4, extender)
    for l in carriles.mismo_sentido:
        _dibujar_linea(out, l, COLOR_MISMO, 4, extender)
    if carriles.ego_izq is not None:
        _dibujar_linea(out, carriles.ego_izq, COLOR_BORDE_EGO, 6, extender)
    if carriles.ego_der is not None:
        _dibujar_linea(out, carriles.ego_der, COLOR_BORDE_EGO, 6, extender)

    y_label = h - 25
    if carriles.ego_izq is not None and carriles.ego_der is not None:
        cx = int((np.polyval(carriles.ego_izq.coefs, extender) +
                  np.polyval(carriles.ego_der.coefs, extender)) / 2)
        _etiqueta(out, cx, y_label, "TU CARRIL", COLOR_EGO)
    if carriles.contrario:
        cx = int(np.mean([np.polyval(l.coefs, extender) for l in carriles.contrario]))
        _etiqueta(out, cx, y_label, "CONTRARIO", COLOR_CONTRARIO)
    if carriles.mismo_sentido:
        cx = int(np.mean([np.polyval(l.coefs, extender) for l in carriles.mismo_sentido]))
        _etiqueta(out, cx, y_label, "MISMO SENTIDO", COLOR_MISMO)

    if mostrar_hud:
        info = {
            "estado": carriles.estado,
            "offset": carriles.offset_px,
            "curvatura": carriles.curvatura_px,
            "fps": fps,
            "frame": frame_idx,
        }
        _hud(out, info)
        if carriles.offset_px is not None:
            _indicador_offset(out, carriles.offset_px, w)

    return out
