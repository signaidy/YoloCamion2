import time
import pytest

from src.tipos import Accion, EstadoEscena, EstadoSemaforo
from src.decision.estado import EstadoFSM
from src.decision.fsm import FSMDecision, _N_FRAMES_OCUPADO, _N_FRAMES_LIBRE, _T_ESPERA_ALTO


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


# ── Regla 8: seguir vehículo ─────────────────────────────────────────────────

def test_regla8_sigue_vehiculo():
    fsm = FSMDecision()
    r = _decidir_n(fsm, _escena(frente_cercano_ocupado=True), _N_FRAMES_OCUPADO)
    assert r.regla == 8
    assert r.accion == Accion.FRENAR_SUAVE
    assert r.estado_nuevo == EstadoFSM.SIGUIENDO_VEHICULO


# ── Regla 9: iniciar rebase ──────────────────────────────────────────────────

def test_regla9_rebase_cuando_condiciones_cumplidas():
    fsm = FSMDecision()
    # Establecer estado SIGUIENDO_VEHICULO
    _decidir_n(fsm, _escena(frente_cercano_ocupado=True), _N_FRAMES_OCUPADO)
    assert fsm.estado_actual == EstadoFSM.SIGUIENDO_VEHICULO

    # Simular que ya llevamos >3s siguiendo
    fsm._t_siguiendo_desde = time.monotonic() - 4.0
    # Espejo izquierdo libre por suficientes frames
    fsm._c_espejo_izq.n_negativos = _N_FRAMES_LIBRE + 1

    r = fsm.decidir(_escena(frente_cercano_ocupado=True, espejo_izq_ocupado=False))
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
