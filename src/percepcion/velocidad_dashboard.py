"""Lectura de velocidad desde el HUD de ETS2.

Evita usar flujo optico para decidir si el camion esta parado: en ETS2, mantener
LT en 0 km/h engrana reversa. El HUD tiene la velocidad numerica en una posicion
estable, asi que leemos ese valor con OpenCV sin dependencias OCR externas.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LecturaVelocidadDashboard:
    kmh: int | None
    norm: float
    confianza: float
    valido: bool


@dataclass(frozen=True)
class _ProtoDigito:
    vec: np.ndarray
    zonas: np.ndarray
    perfil_x: np.ndarray
    perfil_y: np.ndarray
    aspecto: float
    relleno: float
    huecos: int
    fuente: str


_ROI_DIGITOS = (0.025, 0.874, 0.091, 0.995)  # deja margen extra a la derecha: los 2 dígitos altos quedan muy pegados al borde
_SIZE_ROI = (80, 45)  # ancho, alto de referencia
_SIZE_DIGITO = (10, 14)
_BANDA_DIGITOS = (0.12, 0.28, 1.00, 0.84)  # más baja: evita recortar la mitad inferior de los dígitos reales del HUD
_UMBRAL_DIGITOS = 140          # fallback si Otsu devuelve valor degenerado
_MIN_CONF_DIGITO = 0.42
_MIN_CONF_LECTURA = 0.48
_MIN_CONF_DIGITO_RELAX = 0.34
_MIN_CONF_LECTURA_RELAX = 0.40
_MIN_PIXELES_COLUMNA = 3
_MIN_PIXELES_FILA = 5
_MAX_GAP_FILAS = 2
_ANCHO_MIN_SPLIT = 15
_ANCHO_MIN_SPLIT_CON_HUECO = 20
_MAX_PIXELES_COLUMNA_SPLIT = 2
_RATIO_VALLE_SPLIT = 0.45
_MIN_ANCHO_BLOB = 2
_MIN_ALTO_BLOB = 10
_MIN_AREA_BLOB = 10
_MIN_ANCHO_DIGITO = 2
_MIN_ALTO_DIGITO = 10
_MIN_AREA_DIGITO = 10
_MAX_ANCHO_DIGITO = 24
_MAX_ALTO_DIGITO = 28
_PESO_SCORE_PIXELES = 0.34
_PESO_SCORE_ZONAS = 0.18
_PESO_SCORE_PERFIL_X = 0.12
_PESO_SCORE_PERFIL_Y = 0.12
_PESO_SCORE_ASPECTO = 0.08
_PESO_SCORE_RELLENO = 0.06
_PESO_SCORE_HUECOS = 0.10
_MAX_SALTO_KMH_BASE = 12
_MAX_SALTO_KMH_POR_FRAME_PERDIDO = 2
_GRID_ZONAS = (5, 4)
_MARGEN_CONFIANZA_MAX = 0.12


_DIGITOS_ASCII: dict[int, tuple[str, ...]] = {
    0: (
        "..######..",
        "..######..",
        ".########.",
        ".###...###",
        ".##....###",
        ".##....###",
        "###....###",
        "###....###",
        ".##....###",
        ".##....###",
        ".###...###",
        ".########.",
        "..#######.",
        "....###...",
    ),
    1: (
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
    ),
    2: (
        "...######.",
        "...######.",
        "..########",
        "####...###",
        "####...###",
        ".......###",
        "......####",
        ".....####.",
        ".....####.",
        "....####..",
        "...####...",
        "..####....",
        "##########",
        "##########",
    ),
    3: (
        ".########.",
        ".########.",
        ".########.",
        "......###.",
        ".....###..",
        "....###...",
        "...#####..",
        "...######.",
        ".......###",
        ".......###",
        "###....###",
        ".########.",
        "..######..",
        "....##....",
    ),
    4: (
        "......###.",
        "......###.",
        ".....####.",
        "....#####.",
        "...######.",
        "...##.###.",
        "..##..###.",
        ".###..###.",
        ".###..###.",
        ".#########",
        "##########",
        ".#########",
        "......###.",
        "......###.",
    ),
    5: (
        ".########.",
        ".########.",
        ".########.",
        ".###......",
        ".###......",
        ".#######..",
        ".########.",
        ".###..####",
        ".......###",
        "###....###",
        "###....###",
        ".########.",
        ".#######..",
        "....##....",
    ),
    6: (
        "..#######.",
        "..#######.",
        ".########.",
        ".###......",
        ".##.......",
        ".######...",
        ".########.",
        ".###..####",
        ".......###",
        ".##....###",
        "###....###",
        ".########.",
        "..######..",
        "....##....",
    ),
    7: (
        "##########",
        "##########",
        "##########",
        ".......###",
        "......###.",
        "......###.",
        ".....###..",
        ".....###..",
        "....###...",
        "....###...",
        "...###....",
        "...###....",
        "..###.....",
        "..###.....",
    ),
    8: (
        "..#######.",
        "..#######.",
        ".########.",
        ".###...###",
        ".##....###",
        ".###..####",
        "..#######.",
        ".#########",
        ".##....###",
        "###.....##",
        "###....###",
        ".#########",
        "..#######.",
        "....###...",
    ),
    9: (
        "..######..",
        "..######..",
        ".########.",
        "###...###.",
        "###....###",
        "###....###",
        "###...####",
        ".########.",
        ".########.",
        "..######..",
        ".....###..",
        "....###...",
        "...###....",
        "..###.....",
    ),
}


def _plantilla(bits: tuple[str, ...]) -> np.ndarray:
    arr = np.array([[255 if c == "#" else 0 for c in row] for row in bits], dtype=np.uint8)
    return arr.reshape(-1).astype(np.float32) / 255.0


_PLANTILLAS = {digito: _plantilla(bits) for digito, bits in _DIGITOS_ASCII.items()}


def _matriz_plantilla(bits: tuple[str, ...]) -> np.ndarray:
    return np.array([[255 if c == "#" else 0 for c in row] for row in bits], dtype=np.uint8)


def _zonas(arr: np.ndarray, filas: int, cols: int) -> np.ndarray:
    h, w = arr.shape[:2]
    ys = np.linspace(0, h, filas + 1, dtype=np.int32)
    xs = np.linspace(0, w, cols + 1, dtype=np.int32)
    feats: list[float] = []
    for iy in range(filas):
        for ix in range(cols):
            zona = arr[ys[iy]:ys[iy + 1], xs[ix]:xs[ix + 1]]
            feats.append(float(zona.mean()) / 255.0 if zona.size else 0.0)
    return np.array(feats, dtype=np.float32)


def _contar_huecos(arr: np.ndarray) -> int:
    inv = (arr == 0).astype(np.uint8)
    n_labels, _ = cv2.connectedComponents(inv, connectivity=4)
    return max(0, int(n_labels) - 2)


def _perfil_x(arr: np.ndarray) -> np.ndarray:
    return (np.count_nonzero(arr > 0, axis=0).astype(np.float32) / max(1, arr.shape[0]))


def _perfil_y(arr: np.ndarray) -> np.ndarray:
    return (np.count_nonzero(arr > 0, axis=1).astype(np.float32) / max(1, arr.shape[1]))


def _crear_proto(mask: np.ndarray, fuente: str) -> _ProtoDigito:
    norm = cv2.resize(mask, _SIZE_DIGITO, interpolation=cv2.INTER_NEAREST)
    vec = norm.reshape(-1).astype(np.float32) / 255.0
    h, w = mask.shape[:2]
    area = int(np.count_nonzero(mask > 0))
    return _ProtoDigito(
        vec=vec,
        zonas=_zonas(norm, *_GRID_ZONAS),
        perfil_x=_perfil_x(norm),
        perfil_y=_perfil_y(norm),
        aspecto=float(w / max(1, h)),
        relleno=float(area / max(1, w * h)),
        huecos=_contar_huecos(norm),
        fuente=fuente,
    )


_ASPECTO_ESPERADO = {
    0: 0.70,
    1: 0.45,
    2: 0.68,
    3: 0.68,
    4: 0.62,
    5: 0.68,
    6: 0.70,
    7: 0.58,
    8: 0.72,
    9: 0.70,
}
_RELLENO_ESPERADO = {
    0: 0.52,
    1: 0.31,
    2: 0.44,
    3: 0.45,
    4: 0.40,
    5: 0.47,
    6: 0.50,
    7: 0.34,
    8: 0.56,
    9: 0.48,
}
_ZONAS_ESPERADAS = {
    digito: _zonas(_matriz_plantilla(bits), *_GRID_ZONAS)
    for digito, bits in _DIGITOS_ASCII.items()
}
_HUECOS_ESPERADOS = {
    digito: _contar_huecos(_matriz_plantilla(bits))
    for digito, bits in _DIGITOS_ASCII.items()
}
_PROTOS_SINTETICOS = {
    digito: [_crear_proto(_matriz_plantilla(bits), "synthetic")]
    for digito, bits in _DIGITOS_ASCII.items()
}


class EstimadorVelocidadDashboard:
    """Lee km/h del HUD inferior izquierdo y lo normaliza a [0, 1]."""

    def __init__(
        self,
        max_kmh_norm: float = 90.0,
        retener_frames: int = 15,
        prototypes_path: str | None = None,
    ) -> None:
        self._max_kmh_norm = max(1.0, float(max_kmh_norm))
        self._retener_frames = max(0, int(retener_frames))
        self._ultimo_kmh: int | None = None
        self._frames_sin_lectura = 0
        self._prototipos_por_digito = self._cargar_prototipos(prototypes_path)

    @staticmethod
    def _cargar_prototipos(prototypes_path: str | None) -> dict[int, list[_ProtoDigito]]:
        protos = {digito: list(items) for digito, items in _PROTOS_SINTETICOS.items()}
        if not prototypes_path:
            return protos

        ruta = Path(prototypes_path)
        if not ruta.exists():
            _log.warning("Banco de prototipos no encontrado: %s", ruta)
            return protos

        try:
            data = json.loads(ruta.read_text(encoding="utf-8"))
        except Exception:
            _log.exception("No se pudo leer banco de prototipos: %s", ruta)
            return protos

        if not isinstance(data, dict):
            _log.warning("Banco de prototipos inválido: %s", ruta)
            return protos

        for key, muestras in data.items():
            try:
                digito = int(key)
            except (TypeError, ValueError):
                continue
            if digito < 0 or digito > 9:
                continue
            if not isinstance(muestras, list):
                continue
            protos_reales: list[_ProtoDigito] = []
            for muestra in muestras:
                rows = muestra.get("rows") if isinstance(muestra, dict) else muestra
                if not isinstance(rows, list) or not rows:
                    continue
                try:
                    mask = np.array(
                        [[255 if c == "#" else 0 for c in str(row)] for row in rows],
                        dtype=np.uint8,
                    )
                except Exception:
                    continue
                if mask.ndim != 2 or mask.size == 0:
                    continue
                protos_reales.append(_crear_proto(mask, "real"))
            if protos_reales:
                protos[digito] = protos_reales + protos[digito]
        return protos

    def roi_debug(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Devuelve el recorte ROI con la banda, máscara y componentes superpuestos.

        Código de color:
          Verde  — píxeles binarizados
          Cyan   — ventana de búsqueda (_BANDA_DIGITOS)
          Rojo   — blob descartado por filtro grueso
          Naranja — blob aceptado en el filtro grueso pero descartado por tamaño final
          Azul   — componente que llega a clasificación
        """
        roi = self._recortar_roi(frame_bgr)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mask = self._binarizar_digitos(gray)
        debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        debug[mask > 0] = (0, 200, 60)
        bx1, by1, bx2, by2 = self._ventana_digitos(debug.shape[1], debug.shape[0])
        cv2.rectangle(debug, (bx1, by1), (bx2 - 1, by2 - 1), (0, 220, 255), 1)

        # Anotar componentes encontrados (antes de filtros de tamaño) en rojo.
        n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for lbl in range(1, n_labels):
            x0, y0, w, h, area = (int(v) for v in stats[lbl, :5])
            pasa_grueso = self._cumple_filtro_blob(w, h, area)
            color = (0, 0, 220)
            texto_h, texto_w = h, w
            if pasa_grueso:
                sub = mask[y0:y0 + h, x0:x0 + w]
                _px, _py, bw, bh, barea = self._bbox_activa(sub)
                texto_h, texto_w = bh, bw
                color = (0, 100, 255)
                if self._cumple_filtro_digito(bw, bh, barea):
                    color = (220, 80, 0)
            cv2.rectangle(debug, (x0, y0), (x0 + w - 1, y0 + h - 1), color, 1)
            cv2.putText(debug, f"{texto_h}x{texto_w}", (x0, max(0, y0 - 1)),
                        cv2.FONT_HERSHEY_PLAIN, 0.55, color, 1)
        return debug

    def guardar_componentes_debug(self, frame_bgr: np.ndarray, out_dir: str | Path, frame_idx: int) -> None:
        ruta_dir = Path(out_dir)
        ruta_dir.mkdir(parents=True, exist_ok=True)
        roi = self._recortar_roi(frame_bgr)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mask = self._binarizar_digitos(gray)
        componentes = self._extraer_componentes(mask)
        manifest = ruta_dir / "manifest.jsonl"
        for idx, (_x, w, h, area, comp) in enumerate(componentes):
            digito, conf = self._clasificar(comp, w, h, area)
            nombre = f"vel_comp_{frame_idx:06d}_{idx}.png"
            cv2.imwrite(str(ruta_dir / nombre), comp)
            with manifest.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "frame": int(frame_idx),
                    "index": int(idx),
                    "file": nombre,
                    "pred": int(digito),
                    "conf": float(conf),
                    "w": int(w),
                    "h": int(h),
                    "area": int(area),
                }, ensure_ascii=True) + "\n")

    def estimar(self, frame_bgr: np.ndarray) -> LecturaVelocidadDashboard:
        lectura = self.leer(frame_bgr)
        if lectura.valido and lectura.kmh is not None:
            if not self._salto_lectura_aceptable(lectura.kmh):
                lectura = LecturaVelocidadDashboard(None, 0.0, 0.0, False)
            else:
                self._ultimo_kmh = lectura.kmh
                self._frames_sin_lectura = 0
                return lectura

        self._frames_sin_lectura += 1
        if self._ultimo_kmh is not None and self._frames_sin_lectura <= self._retener_frames:
            norm = min(1.0, max(0.0, self._ultimo_kmh / self._max_kmh_norm))
            return LecturaVelocidadDashboard(self._ultimo_kmh, norm, 0.0, False)
        return LecturaVelocidadDashboard(None, 0.0, 0.0, False)

    def leer(self, frame_bgr: np.ndarray) -> LecturaVelocidadDashboard:
        roi = self._recortar_roi(frame_bgr)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mask = self._binarizar_digitos(gray)

        componentes = self._extraer_componentes(mask)
        if not componentes:
            if _log.isEnabledFor(logging.DEBUG):
                n, _l, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                blobs = [(int(stats[i, 2]), int(stats[i, 3]), int(stats[i, 4]))
                         for i in range(1, n)]  # (w, h, area) per blob
                px = int(np.count_nonzero(mask))
                _log.debug("vel sin componentes: px=%d blobs=%s", px, blobs)
            return LecturaVelocidadDashboard(None, 0.0, 0.0, False)

        lectura = self._construir_lectura(componentes, _MIN_CONF_DIGITO, _MIN_CONF_LECTURA)
        if lectura.valido:
            return lectura
        lectura_relajada = self._construir_lectura(
            componentes,
            _MIN_CONF_DIGITO_RELAX,
            _MIN_CONF_LECTURA_RELAX,
        )
        if lectura_relajada.kmh is not None:
            return lectura_relajada
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("vel conf baja: comps=%s lectura=%s", componentes, lectura)
        return lectura

    def _recortar_roi(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        x1f, y1f, x2f, y2f = _ROI_DIGITOS
        x1 = max(0, min(w - 1, int(round(w * x1f))))
        y1 = max(0, min(h - 1, int(round(h * y1f))))
        x2 = max(x1 + 1, min(w, int(round(w * x2f))))
        y2 = max(y1 + 1, min(h, int(round(h * y2f))))
        roi = frame_bgr[y1:y2, x1:x2]
        return cv2.resize(roi, _SIZE_ROI, interpolation=cv2.INTER_AREA)

    def _binarizar_digitos(self, gray: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self._ventana_digitos(gray.shape[1], gray.shape[0])
        banda = gray[y1:y2, x1:x2]
        # Otsu adapta el umbral al contraste real del frame; si la imagen es
        # demasiado uniforme (umbral degenerado) se cae al valor fijo.
        otsu_val, mask = cv2.threshold(banda, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if otsu_val < 40 or otsu_val > 220:
            _, mask = cv2.threshold(banda, _UMBRAL_DIGITOS, 255, cv2.THRESH_BINARY)
        # Primera pasada: aisla el grupo principal de filas (descarta elementos
        # del cuadrante por encima/debajo de los dígitos reales).
        mask = self._seleccionar_banda_filas(mask)
        # Une cortes horizontales de trazo sin enganchar texto cercano.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 2), np.uint8))
        # Segunda pasada: después del cierre morfológico, reafirma el grupo.
        mask = self._seleccionar_banda_filas(mask)
        salida = np.zeros_like(gray)
        salida[y1:y2, x1:x2] = mask
        return salida

    @staticmethod
    def _ventana_digitos(ancho: int, alto: int) -> tuple[int, int, int, int]:
        x1f, y1f, x2f, y2f = _BANDA_DIGITOS
        x1 = max(0, min(ancho - 1, int(round(ancho * x1f))))
        y1 = max(0, min(alto - 1, int(round(alto * y1f))))
        x2 = max(x1 + 1, min(ancho, int(round(ancho * x2f))))
        y2 = max(y1 + 1, min(alto, int(round(alto * y2f))))
        return x1, y1, x2, y2

    def _extraer_componentes(self, mask: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
        n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:
            return []

        componentes: list[tuple[int, int, int, int, np.ndarray]] = []
        for label in range(1, n_labels):
            x0, y0, w, h, area = (int(v) for v in stats[label, :5])
            if not self._cumple_filtro_blob(w, h, area):
                continue
            sub = mask[y0:y0 + h, x0:x0 + w]
            componentes.extend(self._componentes_de_grupo(sub, x0))

        componentes.sort(key=lambda item: item[3], reverse=True)
        principales = [(x, w, h, area, comp) for x, w, h, area, comp in componentes[:3]]
        principales.sort(key=lambda item: item[0])
        return principales

    def _componentes_de_grupo(
        self,
        sub: np.ndarray,
        x_offset: int,
    ) -> list[tuple[int, int, int, int, np.ndarray]]:
        x_local, y_local, w, h, area = self._bbox_activa(sub)
        if area <= 0:
            return []

        bbox = sub[y_local:y_local + h, x_local:x_local + w]
        split_col = self._buscar_split_columna(bbox)
        if split_col is not None:
            componentes_split: list[tuple[int, int, int, int, np.ndarray]] = []
            for part_x, part in ((0, bbox[:, :split_col]), (split_col, bbox[:, split_col:])):
                px, py, pw, ph, parea = self._bbox_activa(part)
                if not self._cumple_filtro_digito(pw, ph, parea):
                    continue
                comp = part[py:py + ph, px:px + pw]
                comp = cv2.copyMakeBorder(comp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                comp = cv2.resize(comp, _SIZE_DIGITO, interpolation=cv2.INTER_NEAREST)
                componentes_split.append((x_offset + x_local + part_x + px, pw, ph, parea, comp))
            if len(componentes_split) >= 2:
                return componentes_split

        if not self._cumple_filtro_digito(w, h, area):
            return []

        comp = bbox
        comp = cv2.copyMakeBorder(comp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        comp = cv2.resize(comp, _SIZE_DIGITO, interpolation=cv2.INTER_NEAREST)
        return [(x_offset + x_local, w, h, area, comp)]

    @staticmethod
    def _seleccionar_banda_filas(mask: np.ndarray) -> np.ndarray:
        rows = np.count_nonzero(mask > 0, axis=1)
        activas = np.flatnonzero(rows >= _MIN_PIXELES_FILA)
        if activas.size == 0:
            return mask

        cortes = np.where(np.diff(activas) > _MAX_GAP_FILAS)[0] + 1
        grupos = np.split(activas, cortes)
        mejor = max(grupos, key=lambda grupo: (int(rows[grupo].sum()), int(grupo.size)))
        y0 = max(0, int(mejor[0]) - 1)
        y1 = min(mask.shape[0], int(mejor[-1]) + 2)
        salida = np.zeros_like(mask)
        salida[y0:y1, :] = mask[y0:y1, :]
        return salida

    @staticmethod
    def _bbox_activa(mask: np.ndarray) -> tuple[int, int, int, int, int]:
        xs = np.flatnonzero(np.count_nonzero(mask > 0, axis=0) > 0)
        ys = np.flatnonzero(np.count_nonzero(mask > 0, axis=1) > 0)
        if xs.size == 0 or ys.size == 0:
            return 0, 0, 0, 0, 0

        x = int(xs[0])
        y = int(ys[0])
        w = int(xs[-1] - xs[0] + 1)
        h = int(ys[-1] - ys[0] + 1)
        area = int(np.count_nonzero(mask[y:y + h, x:x + w]))
        return x, y, w, h, area

    @staticmethod
    def _cumple_filtro_blob(ancho: int, alto: int, area: int) -> bool:
        return area >= _MIN_AREA_BLOB and alto >= _MIN_ALTO_BLOB and ancho >= _MIN_ANCHO_BLOB

    @staticmethod
    def _cumple_filtro_digito(ancho: int, alto: int, area: int) -> bool:
        return (
            _MIN_ALTO_DIGITO <= alto <= _MAX_ALTO_DIGITO
            and _MIN_ANCHO_DIGITO <= ancho <= _MAX_ANCHO_DIGITO
            and area >= _MIN_AREA_DIGITO
        )

    @staticmethod
    def _buscar_split_columna(mask: np.ndarray) -> int | None:
        h, w = mask.shape[:2]
        if w < _ANCHO_MIN_SPLIT:
            return None
        if _contar_huecos(mask) > 0 and w < _ANCHO_MIN_SPLIT_CON_HUECO:
            return None
        counts = np.count_nonzero(mask > 0, axis=0)
        i0 = max(2, int(round(w * 0.28)))
        i1 = min(w - 2, int(round(w * 0.72)))
        if i1 <= i0:
            return None
        central = counts[i0:i1]
        idx_local = int(np.argmin(central))
        min_count = int(central[idx_local])
        left_peak = int(central[:idx_local].max()) if idx_local > 0 else 0
        right_peak = int(central[idx_local + 1:].max()) if idx_local + 1 < central.size else 0
        peak = max(left_peak, right_peak)
        if peak <= 0:
            return None
        max_valley = max(_MAX_PIXELES_COLUMNA_SPLIT, int(round(peak * _RATIO_VALLE_SPLIT)))
        if min_count > max_valley:
            return None
        split_col = i0 + idx_local
        if split_col < 3 or (w - split_col) < 3:
            return None
        return split_col

    def _construir_lectura(
        self,
        componentes: list[tuple[int, int, int, int, np.ndarray]],
        min_conf_digito: float,
        min_conf_lectura: float,
    ) -> LecturaVelocidadDashboard:
        digitos: list[int] = []
        confs: list[float] = []
        for _x, w, h, area, comp in componentes:
            digito, conf = self._clasificar(comp, w, h, area)
            if conf < min_conf_digito:
                continue
            digitos.append(digito)
            confs.append(conf)

        if not digitos:
            return LecturaVelocidadDashboard(None, 0.0, 0.0, False)

        kmh = int("".join(str(d) for d in digitos))
        if kmh > 140:
            return LecturaVelocidadDashboard(None, 0.0, 0.0, False)

        confianza = float(np.mean(confs))
        norm = min(1.0, max(0.0, kmh / self._max_kmh_norm))
        return LecturaVelocidadDashboard(kmh, norm, confianza, confianza >= min_conf_lectura)

    def _clasificar(
        self,
        comp: np.ndarray,
        ancho_orig: int,
        alto_orig: int,
        area_orig: int,
    ) -> tuple[int, float]:
        candidato = _ProtoDigito(
            vec=comp.reshape(-1).astype(np.float32) / 255.0,
            zonas=_zonas(comp, *_GRID_ZONAS),
            perfil_x=_perfil_x(comp),
            perfil_y=_perfil_y(comp),
            aspecto=float(ancho_orig / max(1, alto_orig)),
            relleno=float(area_orig / max(1, ancho_orig * alto_orig)),
            huecos=_contar_huecos(comp),
            fuente="live",
        )
        scores: list[tuple[int, float]] = []
        for digito, protos in self._prototipos_por_digito.items():
            if not protos:
                continue
            score = max(self._score_proto(candidato, proto) for proto in protos)
            scores.append((digito, score))
        if not scores:
            return 0, 0.0
        scores.sort(key=lambda item: item[1], reverse=True)
        mejor_digito, mejor_score = scores[0]
        segundo_score = scores[1][1] if len(scores) > 1 else 0.0
        return mejor_digito, float(min(1.0, max(0.0, mejor_score)))

    def _salto_lectura_aceptable(self, kmh: int) -> bool:
        if self._ultimo_kmh is None:
            return True
        max_delta = _MAX_SALTO_KMH_BASE + self._frames_sin_lectura * _MAX_SALTO_KMH_POR_FRAME_PERDIDO
        return abs(int(kmh) - int(self._ultimo_kmh)) <= max_delta

    @staticmethod
    def _score_metrica(valor: float, esperado: float, tolerancia: float) -> float:
        if tolerancia <= 0.0:
            return 0.0
        return max(0.0, 1.0 - abs(valor - esperado) / tolerancia)

    @staticmethod
    def _score_zonas(actual: np.ndarray, esperado: np.ndarray) -> float:
        if actual.shape != esperado.shape or actual.size == 0:
            return 0.0
        return max(0.0, 1.0 - float(np.mean(np.abs(actual - esperado))))

    @staticmethod
    def _score_huecos(actual: int, esperado: int) -> float:
        if actual == esperado:
            return 1.0
        return max(0.0, 1.0 - 0.6 * abs(int(actual) - int(esperado)))

    def _score_proto(self, candidato: _ProtoDigito, proto: _ProtoDigito) -> float:
        return (
            self._score_binario(candidato.vec, proto.vec) * _PESO_SCORE_PIXELES
            + self._score_zonas(candidato.zonas, proto.zonas) * _PESO_SCORE_ZONAS
            + self._score_zonas(candidato.perfil_x, proto.perfil_x) * _PESO_SCORE_PERFIL_X
            + self._score_zonas(candidato.perfil_y, proto.perfil_y) * _PESO_SCORE_PERFIL_Y
            + self._score_metrica(candidato.aspecto, proto.aspecto, 0.35) * _PESO_SCORE_ASPECTO
            + self._score_metrica(candidato.relleno, proto.relleno, 0.25) * _PESO_SCORE_RELLENO
            + self._score_huecos(candidato.huecos, proto.huecos) * _PESO_SCORE_HUECOS
        )

    @staticmethod
    def _score_binario(actual: np.ndarray, esperado: np.ndarray) -> float:
        if actual.shape != esperado.shape or actual.size == 0:
            return 0.0
        return max(0.0, 1.0 - float(np.mean(np.abs(actual - esperado))))

    @staticmethod
    def _score(a: np.ndarray, b: np.ndarray) -> float:
        if float(a.std()) == 0.0 or float(b.std()) == 0.0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
