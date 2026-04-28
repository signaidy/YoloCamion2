"""EstimadorVelocidadPropia: deriva la velocidad del ego del flujo optico.

Pure-vision (RNF-07): la velocidad NO viene de telemetria del juego.
Un humano juzga su velocidad mirando el asfalto pasar; aqui hacemos lo
mismo: medimos el modulo medio de la componente vertical del flujo
optico en una franja inferior-central de la imagen (asfalto del propio
carril, fuera del capo) y lo escalamos por un factor de calibracion.

La salida queda normalizada en [0, 1], donde 1.0 corresponde a la
velocidad maxima fijada por la calibracion (p. ej. 90 km/h).

Compatible con flujo denso (Farneback) y disperso (Lucas-Kanade): se
ignoran celdas exactamente en cero al promediar para no diluir la senal
del flujo disperso, donde la mayoria del mapa es cero.
"""
from typing import Optional

import numpy as np


_FRANJA_Y_INI = 0.78
_FRANJA_Y_FIN = 0.97
_FRANJA_X_INI = 0.30
_FRANJA_X_FIN = 0.70


class EstimadorVelocidadPropia:
    """Convierte un mapa de flujo optico (H, W, 2) en velocidad propia [0, 1].

    Args:
        factor_calibracion: px/s -> velocidad normalizada. Calibrar con
            dos pasadas conocidas en ETS2 (p. ej. 40 y 80 km/h en recta)
            y guardar el valor en config/default.yaml.
        alpha_ema: factor del filtro exponencial (0, 1]. 1.0 = sin
            suavizar (default). Valores menores amortiguan picos.
    """

    def __init__(
        self,
        factor_calibracion: float = 0.01,
        alpha_ema: float = 1.0,
    ):
        if not (0.0 < alpha_ema <= 1.0):
            raise ValueError("alpha_ema debe estar en (0, 1]")
        self._factor = float(factor_calibracion)
        self._alpha = float(alpha_ema)
        self._estado: Optional[float] = None

    def estimar(self, flujo: np.ndarray) -> float:
        h, w = flujo.shape[:2]
        y1 = int(round(_FRANJA_Y_INI * h))
        y2 = int(round(_FRANJA_Y_FIN * h))
        x1 = int(round(_FRANJA_X_INI * w))
        x2 = int(round(_FRANJA_X_FIN * w))
        franja = flujo[y1:y2, x1:x2]

        v_comp = franja[..., 1]
        mascara = v_comp != 0
        if mascara.any():
            magnitud_media = float(np.abs(v_comp[mascara]).mean())
        else:
            magnitud_media = 0.0

        crudo = magnitud_media * self._factor
        if crudo < 0.0:
            crudo = 0.0
        elif crudo > 1.0:
            crudo = 1.0

        if self._estado is None:
            self._estado = crudo
        else:
            self._estado = self._alpha * crudo + (1.0 - self._alpha) * self._estado
        return self._estado

    def reset(self) -> None:
        self._estado = None
