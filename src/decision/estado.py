from enum import Enum


class EstadoFSM(Enum):
    INICIALIZANDO = "inicializando"
    CONDUCIENDO_NORMAL = "conduciendo_normal"
    SIGUIENDO_VEHICULO = "siguiendo_vehiculo"
    FRENANDO_PREVENTIVO = "frenando_preventivo"
    APROXIMANDO_ALTO = "aproximando_alto"
    DETENIDO_ALTO = "detenido_alto"
    APROXIMANDO_SEMAFORO = "aproximando_semaforo"
    DETENIDO_SEMAFORO = "detenido_semaforo"
    CRUZANDO = "cruzando"
    EVALUANDO_REBASE = "evaluando_rebase"
    REBASANDO = "rebasando"
    RECUPERACION = "recuperacion"
    PARO_EMERGENCIA = "paro_emergencia"
