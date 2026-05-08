import numpy as np

from src.control.preferencia_carril_policy import sesgo_preferir_carril_derecho
from src.percepcion.analisis_carriles import CarrilesClasificados, Linea
from src.tipos import EstadoEscena, EstadoRuta, ManiobraRuta


def _linea(x_eval: float) -> Linea:
    return Linea(
        coefs=np.array([0.0, 0.0, float(x_eval)]),
        y_min=200,
        y_max=400,
        x_eval=float(x_eval),
        n_pixeles=200,
    )


def _carriles(*, mismo: bool = False, contrario: bool = False) -> CarrilesClasificados:
    return CarrilesClasificados(
        ego_izq=_linea(780.0),
        ego_der=_linea(1080.0),
        mismo_sentido=[_linea(1360.0)] if mismo else [],
        contrario=[_linea(520.0)] if contrario else [],
    )


def _escena(*, espejo_der: bool = False, lateral_der: bool = False) -> EstadoEscena:
    return EstadoEscena(
        frente_cercano_ocupado=False,
        frente_lejano_ocupado=False,
        peaton_en_riesgo=False,
        semaforo_visible=None,
        senal_alto_cercana=False,
        espejo_izq_ocupado=False,
        espejo_der_ocupado=espejo_der,
        vehiculos_totales=0,
        confianza_percepcion=1.0,
        timestamp=0.0,
        lateral_izq_ocupado=False,
        lateral_der_ocupado=lateral_der,
    )


def _ruta(
    maniobra: ManiobraRuta = ManiobraRuta.SEGUIR_RECTO,
    conf: float = 0.0,
    visible: bool = False,
) -> EstadoRuta:
    return EstadoRuta(
        visible=visible,
        confianza=conf,
        maniobra=maniobra,
        distancia_normalizada=None,
        sesgo_lateral_objetivo=0.0,
        ramal_objetivo="centro",
        requiere_cambio_carril=False,
    )


def test_prefiere_carril_derecho_si_hay_otro_del_mismo_sentido():
    sesgo, motivo = sesgo_preferir_carril_derecho(
        _carriles(mismo=True, contrario=True),
        _escena(),
        _ruta(),
    )

    assert sesgo == 0.06
    assert motivo == "ms"


def test_se_aleja_del_carril_contrario_si_no_hay_otro_a_la_derecha():
    sesgo, motivo = sesgo_preferir_carril_derecho(
        _carriles(contrario=True),
        _escena(),
        _ruta(),
    )

    assert sesgo == 0.05
    assert motivo == "ct"


def test_no_aplica_si_hay_ruta_activa_o_en_cooldown():
    sesgo_activo, _ = sesgo_preferir_carril_derecho(
        _carriles(mismo=True),
        _escena(),
        _ruta(),
        ruta_activa=True,
    )
    sesgo_cooldown, _ = sesgo_preferir_carril_derecho(
        _carriles(mismo=True),
        _escena(),
        _ruta(),
        ruta_en_cooldown=True,
    )

    assert sesgo_activo == 0.0
    assert sesgo_cooldown == 0.0


def test_no_aplica_si_el_lateral_derecho_esta_ocupado():
    sesgo, motivo = sesgo_preferir_carril_derecho(
        _carriles(mismo=True, contrario=True),
        _escena(espejo_der=True),
        _ruta(),
    )

    assert sesgo == 0.0
    assert motivo == ""


def test_no_lucha_con_una_ruta_confirmada_hacia_la_izquierda():
    sesgo, motivo = sesgo_preferir_carril_derecho(
        _carriles(mismo=True, contrario=True),
        _escena(),
        _ruta(ManiobraRuta.GIRO_IZQ, conf=0.80, visible=True),
    )

    assert sesgo == 0.0
    assert motivo == ""


def test_no_prefiere_mismo_sentido_si_no_hay_referencia_de_contrario():
    sesgo, motivo = sesgo_preferir_carril_derecho(
        _carriles(mismo=True, contrario=False),
        _escena(),
        _ruta(),
    )

    assert sesgo == 0.0
    assert motivo == ""
