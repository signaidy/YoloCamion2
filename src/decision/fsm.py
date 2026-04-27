"""Máquina de estados de decisión con 12 reglas priorizadas.

Las reglas se evalúan en orden; la primera coincidencia gana.
Toda transición emite la razón y el número de regla en el log.
"""
import time
from dataclasses import dataclass, field
from typing import Optional

from src.tipos import Accion, EstadoEscena, EstadoSemaforo
from src.decision.estado import EstadoFSM

_N_FRAMES_OCUPADO = 4   # frames consecutivos para declarar "ocupado" (mss ~5-10 FPS → ~0.5-0.8s)
_N_FRAMES_LIBRE   = 10  # frames consecutivos para declarar "libre"  (mss → ~1-2s sin detección)
_T_ESPERA_ALTO    = 2.0  # segundos de parada completa antes de cruzar
_T_MIN_SIGUIENDO  = 8.0  # segundos mínimos siguiendo antes de evaluar rebase


@dataclass
class _Contador:
    """Histéresis de N frames consecutivos positivos / negativos."""
    n_positivos: int = 0
    n_negativos: int = 0

    def actualizar(self, condicion: bool) -> None:
        if condicion:
            self.n_positivos += 1
            self.n_negativos = 0
        else:
            self.n_negativos += 1
            self.n_positivos = 0

    def esta_activo(self, umbral: int = _N_FRAMES_OCUPADO) -> bool:
        return self.n_positivos >= umbral

    def esta_inactivo(self, umbral: int = _N_FRAMES_LIBRE) -> bool:
        return self.n_negativos >= umbral


@dataclass
class ResultadoDecision:
    accion: Accion
    estado_nuevo: EstadoFSM
    regla: int
    razon: str


class FSMDecision:
    """Evalúa el EstadoEscena y devuelve la acción a ejecutar."""

    def __init__(self):
        self._estado = EstadoFSM.INICIALIZANDO
        self._paro_manual = False

        # Contadores de histéresis por región
        self._c_frente_cercano = _Contador()
        self._c_frente_lejano  = _Contador()
        self._c_peaton         = _Contador()
        self._c_espejo_izq     = _Contador()
        self._c_espejo_der     = _Contador()
        self._c_semaforo_rojo  = _Contador()
        self._c_semaforo_verde = _Contador()
        self._c_senal_alto     = _Contador()
        self._c_confianza_baja = _Contador()

        # Timers
        self._t_inicio_alto: Optional[float] = None
        self._t_siguiendo_desde: Optional[float] = None

    def activar_paro_manual(self) -> None:
        self._paro_manual = True

    def decidir(self, escena: EstadoEscena) -> ResultadoDecision:
        self._actualizar_contadores(escena)
        resultado = self._evaluar_reglas(escena)
        self._estado = resultado.estado_nuevo
        return resultado

    @property
    def estado_actual(self) -> EstadoFSM:
        return self._estado

    def _actualizar_contadores(self, escena: EstadoEscena) -> None:
        self._c_frente_cercano.actualizar(escena.frente_cercano_ocupado)
        self._c_frente_lejano.actualizar(escena.frente_lejano_ocupado)
        self._c_peaton.actualizar(escena.peaton_en_riesgo)
        self._c_espejo_izq.actualizar(escena.espejo_izq_ocupado)
        self._c_espejo_der.actualizar(escena.espejo_der_ocupado)
        self._c_semaforo_rojo.actualizar(escena.semaforo_visible == EstadoSemaforo.ROJO)
        self._c_semaforo_verde.actualizar(escena.semaforo_visible == EstadoSemaforo.VERDE)
        self._c_senal_alto.actualizar(escena.senal_alto_cercana)
        self._c_confianza_baja.actualizar(escena.confianza_percepcion < 0.3)

    def _evaluar_reglas(self, escena: EstadoEscena) -> ResultadoDecision:
        # Regla 1 — Paro manual o watchdog (máxima prioridad, RF-12)
        if self._paro_manual:
            return ResultadoDecision(
                Accion.ALTO_TOTAL, EstadoFSM.PARO_EMERGENCIA,
                1, "paro manual activado"
            )

        # Regla 2 — Confianza de percepción muy baja → recuperación
        if self._c_confianza_baja.esta_activo():
            return ResultadoDecision(
                Accion.FRENAR_SUAVE, EstadoFSM.RECUPERACION,
                2, f"confianza_percepcion={escena.confianza_percepcion:.2f} < 0.3 por {self._c_confianza_baja.n_positivos} frames"
            )

        # Regla 3 — Peatón en riesgo (RF prioridad humana)
        if self._c_peaton.esta_activo():
            return ResultadoDecision(
                Accion.FRENAR_FUERTE, EstadoFSM.FRENANDO_PREVENTIVO,
                3, "peatón detectado en zona de riesgo"
            )

        # Regla 4 — Semáforo rojo → detenerse (RF-07)
        if self._c_semaforo_rojo.esta_activo():
            return ResultadoDecision(
                Accion.ALTO_TOTAL, EstadoFSM.DETENIDO_SEMAFORO,
                4, "semáforo en ROJO"
            )

        # Regla 5 — Semáforo amarillo → frenar suave si no estamos detenidos
        if (escena.semaforo_visible == EstadoSemaforo.AMARILLO
                and self._estado != EstadoFSM.DETENIDO_SEMAFORO):
            return ResultadoDecision(
                Accion.FRENAR_SUAVE, EstadoFSM.APROXIMANDO_SEMAFORO,
                5, "semáforo en AMARILLO"
            )

        # Regla 6 — Señal de alto cercana y no hemos esperado suficiente (RF-07)
        if self._c_senal_alto.esta_activo():
            if self._t_inicio_alto is None:
                self._t_inicio_alto = time.monotonic()
            t_detenido = time.monotonic() - self._t_inicio_alto
            if t_detenido < _T_ESPERA_ALTO:
                return ResultadoDecision(
                    Accion.ALTO_TOTAL, EstadoFSM.DETENIDO_ALTO,
                    6, f"señal de ALTO — esperando {t_detenido:.1f}/{_T_ESPERA_ALTO}s"
                )
        else:
            self._t_inicio_alto = None

        # Regla 7 — Saliendo de alto: laterales libres → cruzar (RF-08)
        if (self._estado == EstadoFSM.DETENIDO_ALTO
                and self._t_inicio_alto is not None
                and (time.monotonic() - self._t_inicio_alto) >= _T_ESPERA_ALTO
                and self._c_espejo_izq.esta_inactivo()
                and self._c_espejo_der.esta_inactivo()):
            return ResultadoDecision(
                Accion.ACELERAR, EstadoFSM.CRUZANDO,
                7, "alto completado, laterales libres"
            )

        # Resetear timer si venimos de REBASANDO (rebase terminado o abortado)
        # Sin esto, R9 dispara de nuevo en el siguiente frame porque el timer
        # ya tenía >= _T_MIN_SIGUIENDO acumulados.
        if self._estado == EstadoFSM.REBASANDO:
            self._t_siguiendo_desde = None

        # Actualizar timer de "siguiendo" antes de evaluar R8/R9
        if escena.frente_cercano_ocupado:
            if self._t_siguiendo_desde is None:
                self._t_siguiendo_desde = time.monotonic()
        else:
            self._t_siguiendo_desde = None

        # Regla 10 — Conflicto lateral durante rebase → abortar inmediatamente (RF seguridad)
        if (self._estado == EstadoFSM.REBASANDO
                and (escena.espejo_izq_ocupado or escena.espejo_der_ocupado)):
            self._t_siguiendo_desde = None
            return ResultadoDecision(
                Accion.FRENAR_SUAVE, EstadoFSM.SIGUIENDO_VEHICULO,
                10, "conflicto lateral durante rebase — abortando"
            )

        # Regla 9 — Siguiendo >_T_MIN_SIGUIENDO s + espejo izq libre → iniciar rebase (RF-09)
        # Evaluado ANTES de R8 para que pueda disparar cuando las condiciones son correctas
        if (self._estado == EstadoFSM.SIGUIENDO_VEHICULO
                and self._t_siguiendo_desde is not None
                and (time.monotonic() - self._t_siguiendo_desde) >= _T_MIN_SIGUIENDO
                and self._c_espejo_izq.esta_inactivo()
                and not escena.espejo_izq_ocupado):
            return ResultadoDecision(
                Accion.REBASAR_IZQ, EstadoFSM.REBASANDO,
                9, "condiciones de rebase cumplidas"
            )

        # Regla 8 — Frente cercano ocupado → seguir vehículo
        if self._c_frente_cercano.esta_activo() and self._estado != EstadoFSM.REBASANDO:
            return ResultadoDecision(
                Accion.FRENAR_SUAVE, EstadoFSM.SIGUIENDO_VEHICULO,
                8, "vehículo en frente cercano"
            )

        # Regla 11 — Semáforo verde y frente libre → avanzar (RF-06)
        if self._c_semaforo_verde.esta_activo() and self._c_frente_cercano.esta_inactivo():
            return ResultadoDecision(
                Accion.ACELERAR, EstadoFSM.CONDUCIENDO_NORMAL,
                11, "semáforo VERDE, frente libre"
            )

        # Regla 12 — Default: conducir normal
        return ResultadoDecision(
            Accion.MANTENER, EstadoFSM.CONDUCIENDO_NORMAL,
            12, "sin condiciones especiales"
        )
