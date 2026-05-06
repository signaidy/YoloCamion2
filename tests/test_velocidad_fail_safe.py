from src.control.velocidad_fail_safe import limites_por_velocidad_desconocida


def test_no_aplica_guardia_antes_de_varios_frames_sin_lectura():
    cap, freno = limites_por_velocidad_desconocida(
        frames_sin_lectura=5,
        fuente_carril="ll",
        curva=0.02,
        estado_con_carril=True,
    )
    assert cap is None
    assert freno == 0.0


def test_cap_suave_cuando_falta_velocidad_por_un_rato():
    cap, freno = limites_por_velocidad_desconocida(
        frames_sin_lectura=20,
        fuente_carril="ll",
        curva=0.03,
        estado_con_carril=True,
    )
    assert cap == 0.16
    assert freno == 0.0


def test_cap_fuerte_y_freno_si_sigue_ciego_en_da_o_curva():
    cap, freno = limites_por_velocidad_desconocida(
        frames_sin_lectura=60,
        fuente_carril="da",
        curva=0.10,
        estado_con_carril=True,
    )
    assert cap == 0.10
    assert freno == 0.06


def test_fuera_de_estados_de_carril_no_interfiere():
    cap, freno = limites_por_velocidad_desconocida(
        frames_sin_lectura=60,
        fuente_carril="ll",
        curva=0.20,
        estado_con_carril=False,
    )
    assert cap is None
    assert freno == 0.0
