import numpy as np
import pytest

from src.tipos import (
    Accion,
    Clase,
    ComandoControl,
    Cuadro,
    Deteccion,
    EstadoEscena,
    EstadoSemaforo,
    Region,
    Seguimiento,
)


def test_cuadro_se_crea_correctamente():
    imagen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cuadro = Cuadro(imagen=imagen, timestamp=100.0, indice=0, fps_instantaneo=30.0)
    assert cuadro.imagen.shape == (1080, 1920, 3)
    assert cuadro.timestamp == 100.0


def test_deteccion_se_crea_correctamente():
    det = Deteccion(
        clase=Clase.VEHICULO,
        caja=(100, 200, 300, 400),
        confianza=0.85,
        area=(300 - 100) * (400 - 200),
    )
    assert det.clase == Clase.VEHICULO
    assert det.caja == (100, 200, 300, 400)


def test_seguimiento_extiende_deteccion():
    seg = Seguimiento(
        clase=Clase.PEATON,
        caja=(0, 0, 50, 100),
        confianza=0.9,
        area=5000,
        id_seguimiento=42,
        edad=5,
    )
    assert seg.id_seguimiento == 42
    assert seg.edad == 5


def test_estado_escena_campos_obligatorios():
    estado = EstadoEscena(
        frente_cercano_ocupado=False,
        frente_lejano_ocupado=False,
        peaton_en_riesgo=False,
        semaforo_visible=None,
        senal_alto_cercana=False,
        espejo_izq_ocupado=False,
        espejo_der_ocupado=False,
        vehiculos_totales=0,
        confianza_percepcion=1.0,
        timestamp=0.0,
    )
    assert estado.confianza_percepcion == 1.0


def test_comando_control_en_rangos():
    cmd = ComandoControl(acelerador=0.5, freno=0.0, volante=-0.3, timestamp=0.0)
    assert 0.0 <= cmd.acelerador <= 1.0
    assert 0.0 <= cmd.freno <= 1.0
    assert -1.0 <= cmd.volante <= 1.0


def test_enums_tienen_valores_esperados():
    assert Clase.VEHICULO.value == "vehiculo"
    assert EstadoSemaforo.ROJO.value == "rojo"
    assert Region.FRENTE_CERCANO.value == "frente_cercano"
    assert Accion.ALTO_TOTAL.value == "alto_total"


def test_clase_cubre_todos_los_tipos_relevantes():
    clases = {c.value for c in Clase}
    assert "vehiculo" in clases
    assert "peaton" in clases
    assert "semaforo" in clases
    assert "senal_alto" in clases


def test_accion_cubre_maniobras_del_fsm():
    acciones = {a.value for a in Accion}
    for esperada in ("mantener", "acelerar", "frenar_suave", "frenar_fuerte",
                     "alto_total", "girar_izq", "girar_der", "rebasar_izq",
                     "rebasar_der", "esperar"):
        assert esperada in acciones
