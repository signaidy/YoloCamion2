import pytest

from src.control.ruta_bias_policy import GobernadorRutaLateral
from src.tipos import EstadoEscena, EstadoRuta, ManiobraRuta


def _ruta(
    maniobra: ManiobraRuta,
    sesgo: float,
    ramal: str,
    conf: float = 0.80,
    requiere_cambio: bool = True,
    visible: bool = True,
    distancia: float | None = None,
) -> EstadoRuta:
    return EstadoRuta(
        visible=visible,
        confianza=conf,
        maniobra=maniobra,
        distancia_normalizada=distancia,
        sesgo_lateral_objetivo=sesgo,
        ramal_objetivo=ramal,
        requiere_cambio_carril=requiere_cambio,
    )


def _escena(
    *,
    espejo_izq: bool = False,
    espejo_der: bool = False,
    lateral_izq: bool = False,
    lateral_der: bool = False,
) -> EstadoEscena:
    return EstadoEscena(
        frente_cercano_ocupado=False,
        frente_lejano_ocupado=False,
        peaton_en_riesgo=False,
        semaforo_visible=None,
        senal_alto_cercana=False,
        espejo_izq_ocupado=espejo_izq,
        espejo_der_ocupado=espejo_der,
        vehiculos_totales=0,
        confianza_percepcion=1.0,
        timestamp=0.0,
        lateral_izq_ocupado=lateral_izq,
        lateral_der_ocupado=lateral_der,
    )


def _activar(gov: GobernadorRutaLateral, estado: EstadoRuta, repeticiones: int = 2):
    resultado = None
    for _ in range(repeticiones):
        resultado = gov.actualizar_lectura(estado)
    return resultado


def test_ruta_exige_dos_lecturas_consecutivas_antes_de_activar():
    gov = GobernadorRutaLateral(
        min_confianza_aplicar=0.60,
        activar_tras_lecturas=2,
        factor_sesgo_lateral=0.55,
    )

    primera = gov.actualizar_lectura(_ruta(ManiobraRuta.MANTENER_DER, 0.18, "der", conf=0.78))
    segunda = gov.actualizar_lectura(_ruta(ManiobraRuta.MANTENER_DER, 0.18, "der", conf=0.79))

    assert primera.activo is False
    assert segunda.activo is True
    assert segunda.maniobra is ManiobraRuta.MANTENER_DER
    assert segunda.sesgo_lateral_aplicado == pytest.approx(0.18 * 0.55)


def test_salida_fuerte_y_cercana_activa_en_una_sola_lectura():
    gov = GobernadorRutaLateral(
        min_confianza_aplicar=0.60,
        activar_tras_lecturas=2,
        activar_salida_fuerte_confianza=0.95,
        activar_salida_fuerte_distancia=0.08,
        factor_sesgo_lateral=0.55,
    )

    estado = gov.actualizar_lectura(
        _ruta(ManiobraRuta.SALIDA_IZQ, -0.35, "izq", conf=0.99, distancia=0.03)
    )

    assert estado.activo is True
    assert estado.retenido is False
    assert estado.ramal_objetivo == "izq"
    assert estado.sesgo_lateral_aplicado < 0.0


def test_ruta_retiene_salida_derecha_ante_flicker_a_recta():
    gov = GobernadorRutaLateral(
        min_confianza_aplicar=0.60,
        activar_tras_lecturas=2,
        retener_recto_lecturas=2,
        factor_sesgo_lateral=0.55,
    )

    _activar(gov, _ruta(ManiobraRuta.MANTENER_DER, 0.18, "der", conf=0.80))
    estado = gov.actualizar_lectura(
        _ruta(ManiobraRuta.SEGUIR_RECTO, 0.0, "centro", conf=0.72, requiere_cambio=False)
    )

    assert estado.activo is True
    assert estado.retenido is True
    assert estado.ramal_objetivo == "der"
    assert estado.sesgo_lateral_aplicado > 0.0


def test_ruta_no_cambia_a_lado_opuesto_por_una_sola_lectura():
    gov = GobernadorRutaLateral(
        min_confianza_aplicar=0.60,
        activar_tras_lecturas=2,
        cambiar_opuesto_tras_lecturas=2,
        factor_sesgo_lateral=0.55,
    )

    _activar(gov, _ruta(ManiobraRuta.MANTENER_DER, 0.18, "der", conf=0.79))
    estado = gov.actualizar_lectura(_ruta(ManiobraRuta.SALIDA_IZQ, -0.35, "izq", conf=0.84))

    assert estado.activo is True
    assert estado.ramal_objetivo == "der"
    assert estado.sesgo_lateral_aplicado > 0.0


def test_ruta_cambia_a_lado_opuesto_tras_dos_lecturas_consecutivas():
    gov = GobernadorRutaLateral(
        min_confianza_aplicar=0.60,
        activar_tras_lecturas=2,
        cambiar_opuesto_tras_lecturas=2,
        factor_sesgo_lateral=0.55,
    )

    _activar(gov, _ruta(ManiobraRuta.MANTENER_DER, 0.18, "der", conf=0.79))
    gov.actualizar_lectura(_ruta(ManiobraRuta.SALIDA_IZQ, -0.35, "izq", conf=0.84))
    estado = gov.actualizar_lectura(_ruta(ManiobraRuta.SALIDA_IZQ, -0.35, "izq", conf=0.82))

    assert estado.activo is True
    assert estado.ramal_objetivo == "izq"
    assert estado.sesgo_lateral_aplicado < 0.0


def test_ruta_bloquea_sesgo_si_espejo_objetivo_esta_ocupado():
    gov = GobernadorRutaLateral(
        min_confianza_aplicar=0.60,
        activar_tras_lecturas=2,
        factor_sesgo_lateral=0.55,
    )

    _activar(gov, _ruta(ManiobraRuta.MANTENER_DER, 0.18, "der", conf=0.78))
    estado = gov.estado_actual(_escena(espejo_der=True))

    assert estado.activo is True
    assert estado.bloqueado_por_espejo is True
    assert estado.sesgo_lateral_aplicado == 0.0


def test_ruta_bloquea_sesgo_si_lateral_objetivo_esta_ocupado():
    gov = GobernadorRutaLateral(
        min_confianza_aplicar=0.60,
        activar_tras_lecturas=2,
        factor_sesgo_lateral=0.55,
    )

    _activar(gov, _ruta(ManiobraRuta.MANTENER_IZQ, -0.18, "izq", conf=0.78))
    estado = gov.estado_actual(_escena())
    assert estado.sesgo_lateral_aplicado < 0.0

    estado = gov.estado_actual(_escena(lateral_izq=True))

    assert estado.activo is True
    assert estado.bloqueado_por_lateral is True
    assert estado.sesgo_lateral_aplicado == 0.0


def test_una_sola_lectura_sin_carril_no_da_por_cumplida_la_maniobra():
    gov = GobernadorRutaLateral(
        min_confianza_aplicar=0.60,
        activar_tras_lecturas=2,
        satisfacer_tras_lecturas_sin_carril=2,
        factor_sesgo_lateral=0.55,
    )

    _activar(gov, _ruta(ManiobraRuta.MANTENER_IZQ, -0.18, "izq", conf=0.90))
    estado = gov.estado_actual(_escena(), hay_carril_objetivo=False)

    assert estado.activo is True
    assert estado.sin_carril_objetivo is True
    assert estado.bloqueado_por_satisfaccion is False
    assert estado.sesgo_lateral_aplicado == 0.0


def test_salida_izq_sigue_activa_sin_carril_si_minimapa_confirma_mismo_lado():
    gov = GobernadorRutaLateral(
        min_confianza_aplicar=0.60,
        activar_tras_lecturas=2,
        satisfacer_tras_lecturas_sin_carril=2,
        factor_sesgo_lateral=0.55,
    )
    salida_izq = _ruta(ManiobraRuta.SALIDA_IZQ, -0.35, "izq", conf=0.84)

    _activar(gov, salida_izq)
    primera = gov.estado_actual(_escena(), hay_carril_objetivo=False)
    gov.actualizar_lectura(_ruta(ManiobraRuta.SALIDA_IZQ, -0.35, "izq", conf=0.62))
    segunda = gov.estado_actual(_escena(), hay_carril_objetivo=False)

    assert primera.activo is True
    assert primera.sin_carril_objetivo is True
    assert primera.bloqueado_por_satisfaccion is False
    assert primera.sesgo_lateral_aplicado == 0.0
    assert segunda.activo is True
    assert segunda.sin_carril_objetivo is True
    assert segunda.bloqueado_por_satisfaccion is False
    assert segunda.sesgo_lateral_aplicado == 0.0


def test_ruta_entra_en_cooldown_tras_sin_carril_sostenido_y_no_rearma_mismo_lado():
    gov = GobernadorRutaLateral(
        min_confianza_aplicar=0.60,
        activar_tras_lecturas=2,
        satisfacer_tras_lecturas_sin_carril=2,
        factor_sesgo_satisfecha=0.45,
        reset_satisfecha_tras_neutrales=3,
        factor_sesgo_lateral=0.55,
    )
    ruta_izq = _ruta(ManiobraRuta.MANTENER_IZQ, -0.18, "izq", conf=0.90)

    _activar(gov, ruta_izq)
    gov.estado_actual(_escena(), hay_carril_objetivo=False)
    estado = gov.estado_actual(_escena(), hay_carril_objetivo=False)

    assert estado.activo is False
    assert estado.sin_carril_objetivo is True
    assert estado.bloqueado_por_satisfaccion is True
    assert estado.sesgo_lateral_aplicado == 0.0

    cooldown = gov.estado_actual(_escena())
    assert cooldown.activo is False
    assert cooldown.bloqueado_por_satisfaccion is True
    assert cooldown.sesgo_lateral_aplicado == pytest.approx(-0.18 * 0.55 * 0.45)

    _activar(gov, ruta_izq)
    estado = gov.estado_actual(_escena())
    assert estado.activo is False
    assert estado.bloqueado_por_satisfaccion is True
    assert estado.sesgo_lateral_aplicado == pytest.approx(-0.18 * 0.55 * 0.45)


def test_ruta_en_cooldown_ignora_lado_opuesto_hasta_neutral_sostenida():
    gov = GobernadorRutaLateral(
        min_confianza_aplicar=0.60,
        activar_tras_lecturas=2,
        satisfacer_tras_lecturas_sin_carril=2,
        reset_satisfecha_tras_neutrales=3,
        factor_sesgo_lateral=0.55,
    )
    ruta_izq = _ruta(ManiobraRuta.MANTENER_IZQ, -0.18, "izq", conf=0.90)
    ruta_der = _ruta(ManiobraRuta.MANTENER_DER, 0.18, "der", conf=0.88)

    _activar(gov, ruta_izq)
    gov.estado_actual(_escena(), hay_carril_objetivo=False)
    gov.estado_actual(_escena(), hay_carril_objetivo=False)

    _activar(gov, ruta_der)
    estado = gov.estado_actual(_escena())
    assert estado.activo is False
    assert estado.bloqueado_por_satisfaccion is True

    neutral = _ruta(ManiobraRuta.SEGUIR_RECTO, 0.0, "centro", conf=0.90, requiere_cambio=False)
    gov.actualizar_lectura(neutral)
    gov.actualizar_lectura(neutral)
    gov.actualizar_lectura(neutral)

    reactivado = _activar(gov, ruta_der)
    assert reactivado.activo is True
    assert reactivado.ramal_objetivo == "der"


def test_cooldown_suprime_sesgo_si_lateral_objetivo_esta_ocupado():
    gov = GobernadorRutaLateral(
        min_confianza_aplicar=0.60,
        activar_tras_lecturas=2,
        satisfacer_tras_lecturas_sin_carril=2,
        factor_sesgo_satisfecha=0.45,
        reset_satisfecha_tras_neutrales=3,
        factor_sesgo_lateral=0.55,
    )
    ruta_der = _ruta(ManiobraRuta.MANTENER_DER, 0.18, "der", conf=0.90)

    _activar(gov, ruta_der)
    gov.estado_actual(_escena(), hay_carril_objetivo=False)
    gov.estado_actual(_escena(), hay_carril_objetivo=False)

    libre = gov.estado_actual(_escena())
    bloqueado = gov.estado_actual(_escena(lateral_der=True))

    assert libre.bloqueado_por_satisfaccion is True
    assert libre.sesgo_lateral_aplicado == pytest.approx(0.18 * 0.55 * 0.45)
    assert bloqueado.bloqueado_por_satisfaccion is True
    assert bloqueado.sesgo_lateral_aplicado == 0.0
