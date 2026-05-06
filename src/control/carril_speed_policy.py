"""Politica de velocidad segun la confianza de seguimiento de carril.

Se usa para evitar dos fallos opuestos observados en pista:
  - Arranque muy debil cuando YOLOP todavia esta en `decay`.
  - Frenadas espurias a ~8-10 km/h cuando Pure Pursuit cae a `da`.

La politica devuelve dos limites puros y testeables:
  - factor multiplicativo sobre velocidad_objetivo_norm
  - freno minimo preventivo
"""

from __future__ import annotations

_VEL_BAJA_ASISTIDA = 0.12  # ~11 km/h con max_kmh_norm=90
_VEL_FRENO_FALLBACK = 0.17  # ~15 km/h
_FACTOR_DECAY_BAJA = 0.70
_FACTOR_DECAY = 0.50
_FACTOR_DA_BAJA = 0.78
_FACTOR_DA = 0.60
_FRENO_DECAY = 0.10
_FRENO_DA = 0.0


def limites_velocidad_por_carril(
    *,
    fuente_carril: str,
    carril_perdido: bool,
    velocidad_actual_norm: float,
    estado_con_carril: bool,
) -> tuple[float, float]:
    """Devuelve (factor_velocidad, freno_minimo) para el setpoint actual."""
    if not estado_con_carril:
        return 1.0, 0.0

    vel = max(0.0, float(velocidad_actual_norm))
    vel_baja = vel < _VEL_BAJA_ASISTIDA

    if carril_perdido:
        factor = _FACTOR_DECAY_BAJA if vel_baja else _FACTOR_DECAY
        freno = _FRENO_DECAY if vel >= _VEL_FRENO_FALLBACK else 0.0
        return factor, freno

    if fuente_carril == "da":
        factor = _FACTOR_DA_BAJA if vel_baja else _FACTOR_DA
        freno = _FRENO_DA
        return factor, freno

    return 1.0, 0.0
