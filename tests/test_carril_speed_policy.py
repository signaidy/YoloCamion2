from src.control.carril_speed_policy import limites_velocidad_por_carril


def test_decay_baja_velocidad_mantiene_empuje_sin_frenar():
    factor, freno = limites_velocidad_por_carril(
        fuente_carril="decay",
        carril_perdido=True,
        velocidad_actual_norm=0.0,
        estado_con_carril=True,
    )
    assert factor == 0.70
    assert freno == 0.0


def test_da_baja_velocidad_no_convierte_fallback_en_frenada():
    factor, freno = limites_velocidad_por_carril(
        fuente_carril="da",
        carril_perdido=False,
        velocidad_actual_norm=0.10,
        estado_con_carril=True,
    )
    assert factor == 0.78
    assert freno == 0.0


def test_da_velocidad_moderada_limita_sin_meter_lt():
    factor, freno = limites_velocidad_por_carril(
        fuente_carril="da",
        carril_perdido=False,
        velocidad_actual_norm=0.20,
        estado_con_carril=True,
    )
    assert factor == 0.60
    assert freno == 0.0


def test_fuera_de_estados_de_carril_no_hay_ajuste():
    factor, freno = limites_velocidad_por_carril(
        fuente_carril="da",
        carril_perdido=False,
        velocidad_actual_norm=0.20,
        estado_con_carril=False,
    )
    assert factor == 1.0
    assert freno == 0.0
