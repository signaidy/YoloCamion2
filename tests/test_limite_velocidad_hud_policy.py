from src.control.limite_velocidad_hud_policy import GobernadorLimiteVelocidadHUD
from src.tipos import EstadoLimiteVelocidadHUD


def _raw(limite: int | None, conf: float = 0.8, visible: bool = True) -> EstadoLimiteVelocidadHUD:
    return EstadoLimiteVelocidadHUD(visible=visible, confianza=conf, limite_kmh=limite)


def test_no_activa_limite_con_una_sola_lectura():
    gov = GobernadorLimiteVelocidadHUD(max_kmh_norm=90.0, activar_tras_lecturas=2)

    estado = gov.actualizar_lectura(_raw(50))

    assert estado.activo is False
    assert estado.limite_activo_kmh is None


def test_activa_limite_tras_dos_lecturas_estables():
    gov = GobernadorLimiteVelocidadHUD(max_kmh_norm=90.0, activar_tras_lecturas=2)

    gov.actualizar_lectura(_raw(50))
    estado = gov.actualizar_lectura(_raw(50))

    assert estado.activo is True
    assert estado.limite_activo_kmh == 50
    assert estado.cambio_estado is True


def test_cambia_limite_solo_tras_nueva_lectura_estable():
    gov = GobernadorLimiteVelocidadHUD(
        max_kmh_norm=90.0,
        activar_tras_lecturas=2,
        cambiar_tras_lecturas=2,
    )

    gov.actualizar_lectura(_raw(50))
    gov.actualizar_lectura(_raw(50))
    intermedio = gov.actualizar_lectura(_raw(30))
    final = gov.actualizar_lectura(_raw(30))

    assert intermedio.limite_activo_kmh == 50
    assert final.limite_activo_kmh == 30
    assert final.cambio_estado is True


def test_retiene_limite_activo_por_varias_invalidas():
    gov = GobernadorLimiteVelocidadHUD(
        max_kmh_norm=90.0,
        activar_tras_lecturas=2,
        retener_lecturas_invalidas=2,
    )

    gov.actualizar_lectura(_raw(60))
    gov.actualizar_lectura(_raw(60))
    estado = gov.actualizar_lectura(_raw(None, conf=0.0, visible=False))

    assert estado.activo is True
    assert estado.limite_activo_kmh == 60


def test_limpia_limite_activo_tras_demasiadas_invalidas():
    gov = GobernadorLimiteVelocidadHUD(
        max_kmh_norm=90.0,
        activar_tras_lecturas=2,
        retener_lecturas_invalidas=1,
    )

    gov.actualizar_lectura(_raw(60))
    gov.actualizar_lectura(_raw(60))
    gov.actualizar_lectura(_raw(None, conf=0.0, visible=False))
    estado = gov.actualizar_lectura(_raw(None, conf=0.0, visible=False))

    assert estado.activo is False
    assert estado.limite_activo_kmh is None


def test_cap_norm_respeta_tolerancia():
    gov = GobernadorLimiteVelocidadHUD(
        max_kmh_norm=90.0,
        activar_tras_lecturas=1,
        tolerancia_kmh=2.0,
    )
    gov.actualizar_lectura(_raw(50))

    estado = gov.estado_actual()

    assert estado.cap_velocidad_norm == (52.0 / 90.0)
    assert estado.freno_minimo == 0.0


def test_freno_suave_por_exceso_moderado():
    gov = GobernadorLimiteVelocidadHUD(
        max_kmh_norm=90.0,
        activar_tras_lecturas=1,
        tolerancia_kmh=2.0,
        exceso_freno_suave_kmh=6.0,
        exceso_freno_moderado_kmh=12.0,
        freno_suave=0.04,
        freno_moderado=0.08,
    )
    gov.actualizar_lectura(_raw(50))

    estado = gov.estado_actual(velocidad_actual_kmh=59)

    assert estado.cap_velocidad_norm == (52.0 / 90.0)
    assert estado.exceso_kmh == 7.0
    assert estado.freno_minimo == 0.04


def test_freno_moderado_por_exceso_alto():
    gov = GobernadorLimiteVelocidadHUD(
        max_kmh_norm=90.0,
        activar_tras_lecturas=1,
        tolerancia_kmh=2.0,
        exceso_freno_suave_kmh=6.0,
        exceso_freno_moderado_kmh=12.0,
        freno_suave=0.04,
        freno_moderado=0.08,
    )
    gov.actualizar_lectura(_raw(50))

    estado = gov.estado_actual(velocidad_actual_kmh=66)

    assert estado.exceso_kmh == 14.0
    assert estado.freno_minimo == 0.08


def test_ignora_limites_absurdos_y_retiene_un_limite_sano():
    gov = GobernadorLimiteVelocidadHUD(
        max_kmh_norm=90.0,
        limite_min_kmh=10,
        limite_max_kmh=130,
        activar_tras_lecturas=2,
        cambiar_tras_lecturas=2,
    )

    gov.actualizar_lectura(_raw(50))
    activo = gov.actualizar_lectura(_raw(50))
    assert activo.limite_activo_kmh == 50

    intermedio = gov.actualizar_lectura(_raw(1828, conf=0.60))
    final = gov.actualizar_lectura(_raw(18238, conf=0.60))

    assert intermedio.limite_raw_kmh == 1828
    assert intermedio.limite_activo_kmh == 50
    assert final.limite_raw_kmh == 18238
    assert final.limite_activo_kmh == 50
