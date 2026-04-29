# src/control/pure_pursuit.py
import numpy as np


class PurePursuitVisual:
    """
    Controlador Pure Pursuit visual con bias de carril derecho y look-ahead dinámico.

    Mejoras respecto a la versión anterior:
    - Bias derecho: solo considera el 70% derecho del área manejable para no
      apuntar a la línea central en autopistas de dos carriles.
    - Look-ahead dinámico: fila de anticipación se acorta en curvas y se alarga
      en rectas, medido por la curvatura del propio área manejable.
    - Suavizado multi-fila: promedia 5 filas con pesos gaussianos para reducir
      el ruido puntual de la máscara.
    - Recuperación con memoria: cuando se pierde el carril devuelve el último
      error multiplicado por 0.85 (decaimiento) en vez de devolver 0.0.
    """

    _DECAY = 0.85          # factor de decaimiento por frame cuando carril perdido
    _BIAS_FRAC = 0.20      # fracción izquierda del área verde que se descarta
    _FILA_LEJOS = 0.65     # fila relativa para look-ahead en recta (por debajo de máscara 55%)
    _FILA_CERCA = 0.75     # fila relativa para look-ahead en curva (más cerca)
    _CURVATURA_SCALE = 6.0 # factor de amplificación de la curvatura cruda
    _ESCALA_ERROR = 0.35   # divisor de normalización (fracción del ancho)

    def __init__(self) -> None:
        self._ultimo_error: float = 0.0
        self._ultimo_punto: tuple[int, int] | None = None

    # ── API pública ────────────────────────────────────────────────────────────

    @property
    def ultimo_punto_debug(self) -> tuple[int, int] | None:
        """Último look-ahead point calculado; None si el carril estaba perdido."""
        return self._ultimo_punto

    def calcular_giro(self, mascara_camino: np.ndarray) -> tuple[float, bool]:
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

        # Suavizado: 5 filas con pesos gaussianos, separadas 10 px
        offsets = [-20, -10,  0, 10, 20]
        pesos   = [ 0.10, 0.20, 0.40, 0.20, 0.10]

        x_sum = 0.0
        w_sum = 0.0
        for off, peso in zip(offsets, pesos):
            y = max(0, min(fila_base + off, alto - 1))
            x = self._centroide_con_bias(mascara_camino, y, ancho)
            if x is not None:
                x_sum += x * peso
                w_sum += peso

        if w_sum == 0.0:
            self._ultimo_punto = None
            self._ultimo_error *= self._DECAY
            return self._ultimo_error, True

        x_obj = x_sum / w_sum
        self._ultimo_punto = (int(round(x_obj)), fila_base)

        dx = x_camion - x_obj
        error = float(np.clip(dx / (ancho * self._ESCALA_ERROR), -1.0, 1.0))
        self._ultimo_error = error
        return error, False

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _centroide_con_bias(self, mascara: np.ndarray, fila_y: int, ancho: int) -> int | None:
        """
        Centroide del 70% derecho del área manejable en la fila dada.
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
        centroide cercano (fila 80%) y el lejano (fila 65%).
        """
        y_cerca = int(alto * 0.80)
        y_lejos = int(alto * 0.65)

        x_cerca = self._centroide_con_bias(mascara, y_cerca, ancho)
        x_lejos = self._centroide_con_bias(mascara, y_lejos, ancho)

        if x_cerca is None or x_lejos is None:
            return 0.0

        return float(np.clip(abs(x_cerca - x_lejos) / ancho * self._CURVATURA_SCALE, 0.0, 1.0))
