from src.control.velocidad_feedback_policy import velocidad_feedback_para_control
from src.decision.estado import EstadoFSM


def test_estado_detenido_con_lectura_invalida_corta_feedback():
    feedback = velocidad_feedback_para_control(
        velocidad_norm=0.45,
        velocidad_kmh=41,
        lectura_valida=False,
        estado_fsm=EstadoFSM.DETENIDO_SEMAFORO,
    )
    assert feedback == 0.0


def test_estado_detenido_con_creep_bajo_evitar_lt():
    feedback = velocidad_feedback_para_control(
        velocidad_norm=0.05,
        velocidad_kmh=4,
        lectura_valida=True,
        estado_fsm=EstadoFSM.DETENIDO_ALTO,
    )
    assert feedback == 0.0


def test_estado_detenido_con_velocidad_aun_alta_mantiene_feedback():
    feedback = velocidad_feedback_para_control(
        velocidad_norm=0.12,
        velocidad_kmh=9,
        lectura_valida=True,
        estado_fsm=EstadoFSM.DETENIDO_ALTO,
    )
    assert feedback == 0.12


def test_fuera_de_estado_detenido_no_interfiere():
    feedback = velocidad_feedback_para_control(
        velocidad_norm=0.45,
        velocidad_kmh=41,
        lectura_valida=False,
        estado_fsm=EstadoFSM.SIGUIENDO_VEHICULO,
    )
    assert feedback == 0.45
