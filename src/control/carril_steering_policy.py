"""Politica de direccion segun la confianza del seguimiento de carril."""

from __future__ import annotations


_ZONA_MUERTA_CARRIL_PP = 0.005
_GANANCIA_CARRIL_PP = 1.65
_LIMITE_COMANDO_DA = 0.18
_LIMITE_COMANDO_DA_BAJA_VEL = 0.12
_LIMITE_COMANDO_LL = 0.50
_UMBRAL_BOOST_LL = 0.08
_BOOST_LL = 1.15
_KMH_FULL_AUTORIDAD = 18
_FACTOR_MIN_BAJA_VEL = 0.70


def _clip(valor: float, minimo: float, maximo: float) -> float:
    return max(minimo, min(maximo, valor))


def comando_direccion_por_carril(
    desviacion_ema: float,
    fuente_carril: str,
    velocidad_kmh: int | None = None,
) -> float:
    """
    Convierte la desviacion lateral estimada en comando de stick.

    Pure Pursuit usa signo opuesto al gamepad:
      - desviacion positiva -> objetivo a la izquierda
      - stick negativo -> girar a la izquierda

    Cuando `ll` es confiable y la desviacion ya es apreciable, se aplica un
    pequeño boost para evitar que el camión sostenga offsets grandes varios
    ciclos seguidos antes de cerrar el giro.
    """
    if abs(desviacion_ema) < _ZONA_MUERTA_CARRIL_PP:
        return 0.0

    ganancia = _GANANCIA_CARRIL_PP
    if fuente_carril == "ll" and abs(desviacion_ema) >= _UMBRAL_BOOST_LL:
        ganancia *= _BOOST_LL

    salida = _clip(desviacion_ema * ganancia, -1.0, 1.0)
    if velocidad_kmh is not None and velocidad_kmh < _KMH_FULL_AUTORIDAD:
        kmh = max(0.0, min(float(_KMH_FULL_AUTORIDAD), float(velocidad_kmh)))
        factor = _FACTOR_MIN_BAJA_VEL + (1.0 - _FACTOR_MIN_BAJA_VEL) * (kmh / _KMH_FULL_AUTORIDAD)
        salida *= factor
    if fuente_carril == "da":
        limite_da = _LIMITE_COMANDO_DA
        if velocidad_kmh is not None and velocidad_kmh < _KMH_FULL_AUTORIDAD:
            kmh = max(0.0, min(float(_KMH_FULL_AUTORIDAD), float(velocidad_kmh)))
            t = kmh / float(_KMH_FULL_AUTORIDAD)
            limite_da = _LIMITE_COMANDO_DA_BAJA_VEL + (_LIMITE_COMANDO_DA - _LIMITE_COMANDO_DA_BAJA_VEL) * t
        salida = _clip(salida, -limite_da, limite_da)
    elif fuente_carril == "ll":
        salida = _clip(salida, -_LIMITE_COMANDO_LL, _LIMITE_COMANDO_LL)

    return -float(salida)
