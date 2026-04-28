import math
import time
import pytest

from src.tipos import Accion, EstadoEscena, EstadoSemaforo
from src.decision.estado import EstadoFSM
from src.decision.fsm import (
    FSMDecision,
    _N_FRAMES_OCUPADO,
    _N_FRAMES_LIBRE,
    _T_ESPERA_ALTO,
    _TTC_FRENO_FUERTE,
    _TTC_FRENO_SUAVE,
    _TTC_REBASE_OK,
)


def _escena(**kwargs) -> EstadoEscena:
    defaults = dict(
        frente_cercano_ocupado=False,
        frente_lejano_ocupado=False,
        peaton_en_riesgo=False,
        semaforo_visible=None,
        senal_alto_cercana=False,
        espejo_izq_ocupado=False,
        espejo_der_ocupado=False,
        vehiculos_totales=0,
        confianza_percepcion=1.0,
        timestamp=time.monotonic(),
    )
    defaults.update(kwargs)
    return EstadoEscena(**defaults)


def _decidir_n(fsm: FSMDecision, escena: EstadoEscena, n: int):
    """Envía la misma escena n veces para superar la histéresis."""
    resultado = None
    for _ in range(n):
        resultado = fsm.decidir(escena)
    return resultado


# ── Regla 1: paro manual ─────────────────────────────────────────────────────

def test_regla1_paro_manual():
    fsm = FSMDecision()
    fsm.activar_paro_manual()
    r = fsm.decidir(_escena())
    assert r.regla == 1
    assert r.accion == Accion.ALTO_TOTAL
    assert r.estado_nuevo == EstadoFSM.PARO_EMERGENCIA


# ── Regla 2: confianza baja ───────────────────────────────────────────────────

def test_regla2_confianza_baja():
    fsm = FSMDecision()
    r = _decidir_n(fsm, _escena(confianza_percepcion=0.1), _N_FRAMES_OCUPADO)
    assert r.regla == 2
    assert r.accion == Accion.FRENAR_SUAVE
    assert r.estado_nuevo == EstadoFSM.RECUPERACION


# ── Regla 3: peatón en riesgo ────────────────────────────────────────────────

def test_regla3_peaton_en_riesgo():
    fsm = FSMDecision()
    r = _decidir_n(fsm, _escena(peaton_en_riesgo=True), _N_FRAMES_OCUPADO)
    assert r.regla == 3
    assert r.accion == Accion.FRENAR_FUERTE
    assert r.estado_nuevo == EstadoFSM.FRENANDO_PREVENTIVO


# ── Regla 4: semáforo rojo ───────────────────────────────────────────────────

def test_regla4_semaforo_rojo():
    fsm = FSMDecision()
    r = _decidir_n(fsm, _escena(semaforo_visible=EstadoSemaforo.ROJO), _N_FRAMES_OCUPADO)
    assert r.regla == 4
    assert r.accion == Accion.ALTO_TOTAL
    assert r.estado_nuevo == EstadoFSM.DETENIDO_SEMAFORO


# ── Regla 5: semáforo amarillo ───────────────────────────────────────────────

def test_regla5_semaforo_amarillo():
    fsm = FSMDecision()
    r = fsm.decidir(_escena(semaforo_visible=EstadoSemaforo.AMARILLO))
    assert r.regla == 5
    assert r.accion == Accion.FRENAR_SUAVE
    assert r.estado_nuevo == EstadoFSM.APROXIMANDO_SEMAFORO


# ── Regla 6: señal de alto ───────────────────────────────────────────────────

def test_regla6_senal_alto_detiene():
    fsm = FSMDecision()
    r = _decidir_n(fsm, _escena(senal_alto_cercana=True), _N_FRAMES_OCUPADO)
    assert r.regla == 6
    assert r.accion == Accion.ALTO_TOTAL
    assert r.estado_nuevo == EstadoFSM.DETENIDO_ALTO


# ── Regla 7: cruzar tras esperar en alto ────────────────────────────────────

def test_regla7_cruza_tras_espera(monkeypatch):
    fsm = FSMDecision()
    # Llevar al estado DETENIDO_ALTO
    _decidir_n(fsm, _escena(senal_alto_cercana=True), _N_FRAMES_OCUPADO)
    assert fsm.estado_actual == EstadoFSM.DETENIDO_ALTO

    # Simular que ya pasó el tiempo de espera
    t_pasado = time.monotonic() - (_T_ESPERA_ALTO + 0.1)
    fsm._t_inicio_alto = t_pasado

    # Pre-sembrar ambos contadores de espejos como libres
    fsm._c_espejo_izq.n_negativos = _N_FRAMES_LIBRE
    fsm._c_espejo_izq.n_positivos = 0
    fsm._c_espejo_der.n_negativos = _N_FRAMES_LIBRE
    fsm._c_espejo_der.n_positivos = 0

    escena_libre = _escena(
        senal_alto_cercana=True,  # aún visible para no resetear timer
        espejo_izq_ocupado=False,
        espejo_der_ocupado=False,
    )
    r = fsm.decidir(escena_libre)

    assert r.regla == 7
    assert r.accion == Accion.ACELERAR
    assert r.estado_nuevo == EstadoFSM.CRUZANDO


# ── Regla 8: seguir vehículo (con TTC en zona de freno) ─────────────────────

def test_regla8_sigue_vehiculo():
    """R8 con TTC bajo (en zona de freno) -> FRENAR_SUAVE.

    Tras la refinacion R8b: la accion depende del TTC. Sin TTC poblado
    (default inf) el FSM solo MANTIENE distancia. Para frenar suave hace
    falta evidencia visual de aproximacion (TTC < _TTC_FRENO_SUAVE).
    """
    fsm = FSMDecision()
    escena = _escena(frente_cercano_ocupado=True, ttc_minimo_frente_s=2.0)
    r = _decidir_n(fsm, escena, _N_FRAMES_OCUPADO)
    assert r.regla == 8
    assert r.accion == Accion.FRENAR_SUAVE
    assert r.estado_nuevo == EstadoFSM.SIGUIENDO_VEHICULO


# ── Regla 9: iniciar rebase (requiere TTC bajo, R9b) ────────────────────────

def test_regla9_rebase_cuando_condiciones_cumplidas():
    """R9 ahora requiere ademas TTC frontal < _TTC_REBASE_OK (R9b)."""
    fsm = FSMDecision()
    # Establecer estado SIGUIENDO_VEHICULO con TTC en zona de freno
    _decidir_n(fsm, _escena(frente_cercano_ocupado=True, ttc_minimo_frente_s=2.5),
                _N_FRAMES_OCUPADO)
    assert fsm.estado_actual == EstadoFSM.SIGUIENDO_VEHICULO

    # Simular que ya llevamos suficiente tiempo siguiendo (> _T_MIN_SIGUIENDO)
    fsm._t_siguiendo_desde = time.monotonic() - 9.0
    # Espejo izquierdo libre por suficientes frames
    fsm._c_espejo_izq.n_negativos = _N_FRAMES_LIBRE + 1

    r = fsm.decidir(_escena(
        frente_cercano_ocupado=True,
        espejo_izq_ocupado=False,
        ttc_minimo_frente_s=2.5,   # < _TTC_REBASE_OK -> rebasar conviene
    ))
    assert r.regla == 9
    assert r.accion == Accion.REBASAR_IZQ
    assert r.estado_nuevo == EstadoFSM.REBASANDO


# ── Regla 10: abortar rebase ─────────────────────────────────────────────────

def test_regla10_aborta_rebase_por_conflicto():
    fsm = FSMDecision()
    # R10 es inmediata (sin histéresis) por seguridad — 1 frame basta
    fsm._estado = EstadoFSM.REBASANDO
    r = fsm.decidir(_escena(espejo_izq_ocupado=True))
    assert r.regla == 10
    assert r.accion == Accion.FRENAR_SUAVE
    assert r.estado_nuevo == EstadoFSM.SIGUIENDO_VEHICULO


# ── Regla 11: semáforo verde ─────────────────────────────────────────────────

def test_regla11_semaforo_verde_frente_libre():
    fsm = FSMDecision()
    # Frente libre (suficientes frames negativos)
    for _ in range(_N_FRAMES_LIBRE):
        fsm._c_frente_cercano.n_negativos = _N_FRAMES_LIBRE
    r = _decidir_n(fsm, _escena(semaforo_visible=EstadoSemaforo.VERDE), _N_FRAMES_OCUPADO)
    assert r.regla == 11
    assert r.accion == Accion.ACELERAR
    assert r.estado_nuevo == EstadoFSM.CONDUCIENDO_NORMAL


# ── Regla 12: default ────────────────────────────────────────────────────────

def test_regla12_default_mantener():
    fsm = FSMDecision()
    r = fsm.decidir(_escena())
    assert r.regla == 12
    assert r.accion == Accion.MANTENER
    assert r.estado_nuevo == EstadoFSM.CONDUCIENDO_NORMAL


# ── Prioridad: paro manual sobre todo ────────────────────────────────────────

def test_paro_manual_tiene_maxima_prioridad():
    fsm = FSMDecision()
    fsm.activar_paro_manual()
    r = fsm.decidir(_escena(
        semaforo_visible=EstadoSemaforo.VERDE,
        frente_cercano_ocupado=False,
        peaton_en_riesgo=True,
    ))
    assert r.regla == 1
    assert r.estado_nuevo == EstadoFSM.PARO_EMERGENCIA


# ── Reglas TTC (Fase 2.1) ────────────────────────────────────────────────────


def test_regla35_ttc_critico_dispara_freno_fuerte():
    """TTC < _TTC_FRENO_FUERTE en frente -> R3.5 FRENAR_FUERTE inmediato (sin histeresis)."""
    fsm = FSMDecision()
    escena = _escena(
        frente_cercano_ocupado=True,
        ttc_minimo_frente_s=_TTC_FRENO_FUERTE - 0.2,   # 1.3s p.ej.
        vehiculo_critico_id=99,
    )
    r = fsm.decidir(escena)
    assert r.regla == 35
    assert r.accion == Accion.FRENAR_FUERTE
    assert r.estado_nuevo == EstadoFSM.FRENANDO_PREVENTIVO
    assert "ttc" in r.razon.lower() or "TTC" in r.razon


def test_regla35_tiene_prioridad_sobre_seguir_vehiculo():
    """Aunque frente este ocupado por varios frames (R8 candidata), R3.5 gana si TTC critico."""
    fsm = FSMDecision()
    escena = _escena(
        frente_cercano_ocupado=True,
        ttc_minimo_frente_s=0.8,   # critico
    )
    r = _decidir_n(fsm, escena, _N_FRAMES_OCUPADO + 1)
    assert r.regla == 35


def test_regla35_no_se_dispara_si_ttc_es_alto():
    """TTC infinito = no critico, R3.5 no debe disparar."""
    fsm = FSMDecision()
    escena = _escena(frente_cercano_ocupado=False, ttc_minimo_frente_s=math.inf)
    r = fsm.decidir(escena)
    assert r.regla != 35


def test_regla35_no_se_dispara_si_ttc_entre_freno_fuerte_y_freno_suave():
    """TTC en zona intermedia debe ir a R8b, no a R3.5."""
    fsm = FSMDecision()
    escena = _escena(
        frente_cercano_ocupado=True,
        ttc_minimo_frente_s=(_TTC_FRENO_FUERTE + _TTC_FRENO_SUAVE) / 2,  # ~2.25s
    )
    r = _decidir_n(fsm, escena, _N_FRAMES_OCUPADO)
    assert r.regla != 35


def test_regla8b_frente_ocupado_y_ttc_intermedio_frena_suave():
    """R8 con TTC entre _TTC_FRENO_FUERTE y _TTC_FRENO_SUAVE -> FRENAR_SUAVE."""
    fsm = FSMDecision()
    escena = _escena(
        frente_cercano_ocupado=True,
        ttc_minimo_frente_s=2.5,   # entre 1.5 y 3.0
    )
    r = _decidir_n(fsm, escena, _N_FRAMES_OCUPADO)
    assert r.regla == 8
    assert r.accion == Accion.FRENAR_SUAVE
    assert r.estado_nuevo == EstadoFSM.SIGUIENDO_VEHICULO


def test_regla8b_frente_ocupado_pero_ttc_alto_solo_mantiene():
    """Si frente ocupado pero TTC > _TTC_FRENO_SUAVE: vamos a velocidad similar, MANTENER."""
    fsm = FSMDecision()
    escena = _escena(
        frente_cercano_ocupado=True,
        ttc_minimo_frente_s=10.0,   # muy alto
    )
    r = _decidir_n(fsm, escena, _N_FRAMES_OCUPADO)
    assert r.accion == Accion.MANTENER
    assert r.estado_nuevo == EstadoFSM.SIGUIENDO_VEHICULO


def test_regla9b_no_rebasa_si_ttc_alto():
    """Si TTC frontal > _TTC_REBASE_OK, el de adelante avanza igual o mas: NO rebasar."""
    fsm = FSMDecision()
    _decidir_n(fsm, _escena(frente_cercano_ocupado=True), _N_FRAMES_OCUPADO)
    fsm._t_siguiendo_desde = time.monotonic() - 9.0
    fsm._c_espejo_izq.n_negativos = _N_FRAMES_LIBRE + 1

    escena = _escena(
        frente_cercano_ocupado=True,
        espejo_izq_ocupado=False,
        ttc_minimo_frente_s=_TTC_REBASE_OK + 1.0,   # frente avanza, no hay urgencia
    )
    r = fsm.decidir(escena)
    assert r.regla != 9
    assert r.accion != Accion.REBASAR_IZQ


def test_regla9b_si_rebasa_cuando_ttc_es_bajo_y_condiciones_ok():
    """Si TTC frontal < _TTC_REBASE_OK Y siguiendo > _T_MIN_SIGUIENDO Y espejo izq libre: rebasar."""
    fsm = FSMDecision()
    _decidir_n(fsm, _escena(frente_cercano_ocupado=True), _N_FRAMES_OCUPADO)
    fsm._t_siguiendo_desde = time.monotonic() - 9.0
    fsm._c_espejo_izq.n_negativos = _N_FRAMES_LIBRE + 1

    escena = _escena(
        frente_cercano_ocupado=True,
        espejo_izq_ocupado=False,
        ttc_minimo_frente_s=_TTC_REBASE_OK - 1.0,   # frente lento -> conviene rebasar
    )
    r = fsm.decidir(escena)
    assert r.regla == 9
    assert r.accion == Accion.REBASAR_IZQ


# ── ResultadoDecision lleva SetpointControl (Fase 2.2) ─────────────────────


def test_resultado_decision_incluye_setpoint_para_pid():
    """ResultadoDecision debe traer un SetpointControl listo para los PIDs."""
    fsm = FSMDecision()
    r = fsm.decidir(_escena())   # default -> R12 MANTENER
    assert r.setpoint is not None
    assert 0.0 <= r.setpoint.velocidad_objetivo_norm <= 1.0
    assert 0.0 <= r.setpoint.freno_objetivo <= 1.0
    assert -1.0 <= r.setpoint.desviacion_volante <= 1.0


def test_setpoint_acelerar_es_velocidad_alta_sin_freno():
    fsm = FSMDecision()
    fsm._c_frente_cercano.n_negativos = _N_FRAMES_LIBRE
    fsm._c_semaforo_verde.n_positivos = _N_FRAMES_OCUPADO
    r = fsm.decidir(_escena(semaforo_visible=EstadoSemaforo.VERDE))
    assert r.accion == Accion.ACELERAR
    assert r.setpoint.velocidad_objetivo_norm >= 0.5
    assert r.setpoint.freno_objetivo == 0.0


def test_setpoint_alto_total_es_freno_completo_sin_velocidad():
    fsm = FSMDecision()
    fsm.activar_paro_manual()
    r = fsm.decidir(_escena())
    assert r.accion == Accion.ALTO_TOTAL
    assert r.setpoint.freno_objetivo >= 0.9
    assert r.setpoint.velocidad_objetivo_norm == 0.0


def test_setpoint_frenar_fuerte_aplica_freno_alto():
    fsm = FSMDecision()
    escena = _escena(
        frente_cercano_ocupado=True,
        ttc_minimo_frente_s=0.8,
    )
    r = fsm.decidir(escena)
    assert r.accion == Accion.FRENAR_FUERTE
    assert r.setpoint.freno_objetivo >= 0.7
    assert r.setpoint.velocidad_objetivo_norm == 0.0


def test_setpoint_frenar_suave_aplica_freno_moderado():
    fsm = FSMDecision()
    escena = _escena(
        frente_cercano_ocupado=True,
        ttc_minimo_frente_s=2.0,
    )
    r = _decidir_n(fsm, escena, _N_FRAMES_OCUPADO)
    assert r.accion == Accion.FRENAR_SUAVE
    assert 0.2 <= r.setpoint.freno_objetivo < 0.7
    assert r.setpoint.velocidad_objetivo_norm == 0.0


def test_setpoint_rebase_lleva_desviacion_de_volante_a_la_izquierda():
    fsm = FSMDecision()
    _decidir_n(fsm, _escena(frente_cercano_ocupado=True, ttc_minimo_frente_s=2.5),
                _N_FRAMES_OCUPADO)
    fsm._t_siguiendo_desde = time.monotonic() - 9.0
    fsm._c_espejo_izq.n_negativos = _N_FRAMES_LIBRE + 1
    r = fsm.decidir(_escena(
        frente_cercano_ocupado=True,
        espejo_izq_ocupado=False,
        ttc_minimo_frente_s=2.5,
    ))
    assert r.accion == Accion.REBASAR_IZQ
    assert r.setpoint.desviacion_volante < 0   # negativo = izquierda
