"""Guardas cuando la velocidad del HUD no se puede leer por varios frames."""

from __future__ import annotations

_FRAMES_CAP_VELOCIDAD_DESCONOCIDA = 12
_FRAMES_FRENO_VELOCIDAD_DESCONOCIDA = 45
_CAP_SUAVE_VELOCIDAD_DESCONOCIDA = 0.16
_CAP_FUERTE_VELOCIDAD_DESCONOCIDA = 0.10
_CURVA_FRENO_VELOCIDAD_DESCONOCIDA = 0.08
_FRENO_VELOCIDAD_DESCONOCIDA = 0.06


def limites_por_velocidad_desconocida(
    *,
    frames_sin_lectura: int,
    fuente_carril: str,
    curva: float,
    estado_con_carril: bool,
) -> tuple[float | None, float]:
    """Devuelve (cap_velocidad_objetivo_norm, freno_minimo)."""
    if not estado_con_carril or frames_sin_lectura < _FRAMES_CAP_VELOCIDAD_DESCONOCIDA:
        return None, 0.0

    if frames_sin_lectura < _FRAMES_FRENO_VELOCIDAD_DESCONOCIDA:
        return _CAP_SUAVE_VELOCIDAD_DESCONOCIDA, 0.0

    cap = _CAP_FUERTE_VELOCIDAD_DESCONOCIDA
    if fuente_carril != "ll" or float(curva) >= _CURVA_FRENO_VELOCIDAD_DESCONOCIDA:
        return cap, _FRENO_VELOCIDAD_DESCONOCIDA
    return cap, 0.0
