"""Politica para la velocidad realimentada al controlador."""

from __future__ import annotations

from src.decision.estado import EstadoFSM

_ESTADOS_DETENIDOS = frozenset({
    EstadoFSM.DETENIDO_ALTO,
    EstadoFSM.DETENIDO_SEMAFORO,
})
_KMH_CREEP_DETENIDO = 4


def velocidad_feedback_para_control(
    velocidad_norm: float,
    velocidad_kmh: int | None,
    lectura_valida: bool,
    estado_fsm: EstadoFSM,
) -> float:
    """
    Calcula la velocidad que el PID debe considerar como velocidad actual.

    En estados detenidos, una lectura inválida o un valor de creep muy bajo no
    deben mantener LT aplicado: en ETS2 eso termina engranando reversa cuando el
    camión ya está prácticamente parado.
    """
    if estado_fsm in _ESTADOS_DETENIDOS:
        if not lectura_valida:
            return 0.0
        if velocidad_kmh is not None and velocidad_kmh <= _KMH_CREEP_DETENIDO:
            return 0.0
    return max(0.0, min(1.0, float(velocidad_norm)))
