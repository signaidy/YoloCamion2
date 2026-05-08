"""Sesgo pasivo para preferir el carril derecho cuando no hay maniobra activa.

Esta politica no sustituye la navegacion. Solo evita que el seguidor de carril
normalice quedarse en carriles izquierdos cuando existe un carril del mismo
sentido a la derecha, y reduce el riesgo de invadir carriles contrarios en
tramos urbanos o carreteras sin separador central.
"""

from __future__ import annotations

from src.percepcion.analisis_carriles import CarrilesClasificados
from src.tipos import EstadoEscena, EstadoRuta, ManiobraRuta

_BIAS_MISMO_SENTIDO_DER = 0.06
_BIAS_ALEJAR_CONTRARIO = 0.05
_MIN_CONF_RUTA_IZQ = 0.55
_REL_LANE_MIN = 0.60
_REL_LANE_MAX = 1.55


def sesgo_preferir_carril_derecho(
    carriles: CarrilesClasificados,
    escena: EstadoEscena | None,
    estado_ruta: EstadoRuta,
    *,
    ruta_activa: bool = False,
    ruta_en_cooldown: bool = False,
) -> tuple[float, str]:
    """
    Devuelve un sesgo lateral pasivo hacia la derecha.

    Reglas:
    - Nunca luchar contra una maniobra de ruta ya activa o en cooldown.
    - Nunca empujar a la derecha si el lateral/espejo derecho está ocupado.
    - Si existe un carril del mismo sentido a la derecha, preferirlo.
    - Si no existe, pero sí hay carriles contrarios a la izquierda, mantener
      un sesgo suave alejándose de la línea central.
    - Si el minimapa ya confirma una maniobra a la izquierda, no interferir.
    """
    if carriles.estado != "OK":
        return 0.0, ""
    if ruta_activa or ruta_en_cooldown:
        return 0.0, ""
    if _ruta_izquierda_confirmada(estado_ruta):
        return 0.0, ""
    if escena is not None and (escena.espejo_der_ocupado or escena.lateral_der_ocupado):
        return 0.0, ""
    if carriles.contrario and hay_carril_mismo_sentido_derecho_valido(carriles):
        return _BIAS_MISMO_SENTIDO_DER, "ms"
    if carriles.contrario:
        return _BIAS_ALEJAR_CONTRARIO, "ct"
    return 0.0, ""


def _ruta_izquierda_confirmada(estado_ruta: EstadoRuta) -> bool:
    if not estado_ruta.visible or estado_ruta.confianza < _MIN_CONF_RUTA_IZQ:
        return False
    return estado_ruta.maniobra in {
        ManiobraRuta.MANTENER_IZQ,
        ManiobraRuta.SALIDA_IZQ,
        ManiobraRuta.GIRO_IZQ,
    }


def hay_carril_mismo_sentido_derecho_valido(carriles: CarrilesClasificados) -> bool:
    if carriles.estado != "OK":
        return False
    if carriles.ego_izq is None or carriles.ego_der is None or not carriles.mismo_sentido:
        return False
    ancho_ego = float(carriles.ego_der.x_eval - carriles.ego_izq.x_eval)
    ancho_candidato = float(carriles.mismo_sentido[0].x_eval - carriles.ego_der.x_eval)
    if ancho_ego <= 0.0 or ancho_candidato <= 0.0:
        return False
    return (_REL_LANE_MIN * ancho_ego) <= ancho_candidato <= (_REL_LANE_MAX * ancho_ego)
