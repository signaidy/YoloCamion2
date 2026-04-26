from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


@dataclass
class Cuadro:
    imagen: np.ndarray
    timestamp: float
    indice: int
    fps_instantaneo: float


class Clase(Enum):
    VEHICULO = "vehiculo"
    MOTOCICLETA = "motocicleta"
    PEATON = "peaton"
    SEMAFORO = "semaforo"
    SENAL_ALTO = "senal_alto"
    DESCONOCIDO = "desconocido"


@dataclass
class Deteccion:
    clase: Clase
    caja: tuple[int, int, int, int]
    confianza: float
    area: int


@dataclass
class Seguimiento(Deteccion):
    id_seguimiento: int
    edad: int


class EstadoSemaforo(Enum):
    ROJO = "rojo"
    AMARILLO = "amarillo"
    VERDE = "verde"
    DESCONOCIDO = "desconocido"


class Region(Enum):
    FRENTE_CERCANO = "frente_cercano"
    FRENTE_LEJANO = "frente_lejano"
    ESPEJO_IZQ = "espejo_izq"
    ESPEJO_DER = "espejo_der"
    LATERAL_IZQ = "lateral_izq"
    LATERAL_DER = "lateral_der"


@dataclass
class EstadoEscena:
    frente_cercano_ocupado: bool
    frente_lejano_ocupado: bool
    peaton_en_riesgo: bool
    semaforo_visible: Optional[EstadoSemaforo]
    senal_alto_cercana: bool
    espejo_izq_ocupado: bool
    espejo_der_ocupado: bool
    vehiculos_totales: int
    confianza_percepcion: float
    timestamp: float


class Accion(Enum):
    MANTENER = "mantener"
    ACELERAR = "acelerar"
    FRENAR_SUAVE = "frenar_suave"
    FRENAR_FUERTE = "frenar_fuerte"
    ALTO_TOTAL = "alto_total"
    GIRAR_IZQ = "girar_izq"
    GIRAR_DER = "girar_der"
    REBASAR_IZQ = "rebasar_izq"
    REBASAR_DER = "rebasar_der"
    ESPERAR = "esperar"


@dataclass
class ComandoControl:
    acelerador: float
    freno: float
    volante: float
    timestamp: float
