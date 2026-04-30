# src/control/pure_pursuit.py
import numpy as np


class PurePursuitVisual:
    """
    Controlador Pure Pursuit visual con look-ahead dinámico y detección por ll_mask.

    Fuentes de señal (en orden de prioridad):
    - Nivel 1: ll_mask (líneas pintadas). Calcula el centro exacto del carril
      actual como (borde_izq + borde_der) / 2. Funciona para vías de 1-3 carriles.
    - Nivel 2: centroide de da_mask con barrido adaptativo (fallback).
    - Nivel 3: último error × 0.85 (memoria con decaimiento) cuando ambas fallan.

    Look-ahead dinámico: la fila de anticipación se acorta en curvas y se alarga
    en rectas, medido por la diferencia de centroides entre fila 72% y 85%.
    Suavizado multi-fila: promedia 5 filas con pesos gaussianos.
    """

    _DECAY = 0.85          # factor de decaimiento por frame cuando carril perdido
    _BIAS_FRAC = 0.00      # sin sesgo: centroide del área verde completa
    _FILA_LEJOS = 0.72     # recta: zona media donde carretera lejana está clara sin espejos
    _FILA_CERCA = 0.85     # curva: más cerca pero no al ras del asfalto
    _CURVATURA_SCALE = 6.0 # factor de amplificación de la curvatura cruda
    _ESCALA_ERROR = 0.40   # normalización intermedia (look-ahead a distancia media)
    _MIN_LL_PIXELES = 15   # píxeles mínimos por lado en ll_mask para activar nivel 1

    def __init__(self) -> None:
        self._ultimo_error: float = 0.0
        self._ultimo_punto: tuple[int, int] | None = None

    # ── API pública ────────────────────────────────────────────────────────────

    @property
    def ultimo_punto_debug(self) -> tuple[int, int] | None:
        """Último look-ahead point calculado; None si el carril estaba perdido."""
        return self._ultimo_punto

    # Barrido adaptativo: si la fila primaria no tiene verde, baja en pasos de 40px
    # hasta un máximo de _FILA_MAX. Permite funcionar tanto en autopista (72%)
    # como en ciudad/tráfico (fallback hasta 90%).
    _FILA_MAX = 0.90
    _SWEEP_PX = 40

    def calcular_giro(self, mascara_camino: np.ndarray, ll_mask: np.ndarray | None = None) -> tuple[float, bool]:
        """
        Calcula el error de dirección a partir de la máscara del área manejable.

        Returns:
            (error_norm, carril_perdido)
            error_norm  ∈ [-1, 1]: positivo → girar izquierda, negativo → girar derecha
            carril_perdido: True cuando no hay píxeles de carril visibles
        """
        alto, ancho = mascara_camino.shape
        x_camion = ancho // 2

        curvatura = self._estimar_curvatura(mascara_camino, alto, ancho)
        fila_base = int(alto * (self._FILA_LEJOS + curvatura * (self._FILA_CERCA - self._FILA_LEJOS)))

        offsets = [-20, -10,  0, 10, 20]
        pesos   = [ 0.10, 0.20, 0.40, 0.20, 0.10]
        fila_max = int(alto * self._FILA_MAX)

        # Nivel 1: ll_mask — bordes pintados del carril actual
        if ll_mask is not None:
            filas_ll = [max(0, min(fila_base + off, alto - 1)) for off in offsets]
            centro_ll = self._centro_desde_ll(ll_mask, filas_ll, x_camion)
            if centro_ll is not None:
                self._ultimo_punto = (int(round(centro_ll)), fila_base)
                dx = x_camion - centro_ll
                error = float(np.clip(dx / (ancho * self._ESCALA_ERROR), -1.0, 1.0))
                self._ultimo_error = error
                return error, False

        # Nivel 2: centroide da_mask con barrido adaptativo
        x_sum, w_sum, fila_usada = 0.0, 0.0, fila_base
        fila_try = fila_base
        while fila_try <= fila_max:
            xs, ws = 0.0, 0.0
            for off, peso in zip(offsets, pesos):
                y = max(0, min(fila_try + off, alto - 1))
                x = self._centroide_con_bias(mascara_camino, y, ancho)
                if x is not None:
                    xs += x * peso
                    ws += peso
            if ws > 0.0:
                x_sum, w_sum, fila_usada = xs, ws, fila_try
                break
            fila_try += self._SWEEP_PX

        if w_sum == 0.0:
            self._ultimo_punto = None
            self._ultimo_error *= self._DECAY
            return self._ultimo_error, True

        x_obj = x_sum / w_sum
        self._ultimo_punto = (int(round(x_obj)), fila_usada)

        dx = x_camion - x_obj
        error = float(np.clip(dx / (ancho * self._ESCALA_ERROR), -1.0, 1.0))
        self._ultimo_error = error
        return error, False

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _centroide_con_bias(self, mascara: np.ndarray, fila_y: int, ancho: int) -> int | None:
        """
        Centroide del área manejable en la fila dada (_BIAS_FRAC=0.00 → centroide completo).
        Retorna None si no hay píxeles.
        """
        fila = mascara[fila_y, :]
        indices = np.where(fila > 0)[0]
        if len(indices) == 0:
            return None

        x_min = int(indices[0])
        x_max = int(indices[-1])
        ancho_area = x_max - x_min

        if ancho_area == 0:
            return x_min

        x_bias = x_min + int(ancho_area * self._BIAS_FRAC)
        indices_der = indices[indices >= x_bias]
        if len(indices_der) == 0:
            return int(np.mean(indices))
        return int(np.mean(indices_der))

    def _estimar_curvatura(self, mascara: np.ndarray, alto: int, ancho: int) -> float:
        """
        Curvatura ∈ [0, 1] basada en la diferencia horizontal entre el
        centroide cercano (fila 85%) y el lejano (fila 72%).
        """
        y_cerca = int(alto * 0.85)
        y_lejos = int(alto * 0.72)

        x_cerca = self._centroide_con_bias(mascara, y_cerca, ancho)
        x_lejos = self._centroide_con_bias(mascara, y_lejos, ancho)

        if x_cerca is None or x_lejos is None:
            return 0.0

        return float(np.clip(abs(x_cerca - x_lejos) / ancho * self._CURVATURA_SCALE, 0.0, 1.0))

    def _centro_desde_ll(
        self,
        ll_mask: np.ndarray,
        filas: list[int],
        x_camion: int,
    ) -> float | None:
        """
        Centro geométrico del carril actual desde ll_mask.

        Acumula píxeles de las filas dadas, los separa en izquierda/derecha
        respecto a x_camion (= frame center) y retorna (max_izq + min_der) / 2.
        En vías de N carriles, max_izq y min_der son siempre los bordes del
        carril actual (los más cercanos al camión), no los de los carriles vecinos.
        Retorna None si algún lado tiene menos de _MIN_LL_PIXELES píxeles.
        """
        pixeles_izq: list[int] = []
        pixeles_der: list[int] = []
        alto = ll_mask.shape[0]
        for fila_y in filas:
            if fila_y < 0 or fila_y >= alto:
                continue
            indices = np.where(ll_mask[fila_y, :] > 0)[0]
            for x in indices:
                if x < x_camion:
                    pixeles_izq.append(int(x))
                else:
                    pixeles_der.append(int(x))
        if len(pixeles_izq) < self._MIN_LL_PIXELES or len(pixeles_der) < self._MIN_LL_PIXELES:
            return None
        borde_izq = int(np.max(pixeles_izq))
        borde_der = int(np.min(pixeles_der))
        return (borde_izq + borde_der) / 2.0
