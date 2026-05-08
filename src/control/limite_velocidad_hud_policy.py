"""Gobernador de velocidad objetivo según el límite detectado en el HUD."""

from __future__ import annotations

from dataclasses import dataclass

from src.tipos import EstadoLimiteVelocidadHUD


@dataclass(frozen=True)
class EstadoLimiteVelocidadAplicado:
    visible_raw: bool
    confianza_raw: float
    limite_raw_kmh: int | None
    activo: bool
    limite_activo_kmh: int | None
    cap_velocidad_norm: float | None
    freno_minimo: float
    exceso_kmh: float
    cambio_estado: bool = False


class GobernadorLimiteVelocidadHUD:
    def __init__(
        self,
        *,
        max_kmh_norm: float,
        min_confianza_control: float = 0.55,
        limite_min_kmh: int = 10,
        limite_max_kmh: int = 130,
        activar_tras_lecturas: int = 2,
        cambiar_tras_lecturas: int = 2,
        retener_lecturas_invalidas: int = 10,
        tolerancia_kmh: float = 2.0,
        exceso_freno_suave_kmh: float = 6.0,
        exceso_freno_moderado_kmh: float = 12.0,
        freno_suave: float = 0.04,
        freno_moderado: float = 0.08,
    ):
        self._max_kmh_norm = max(1.0, float(max_kmh_norm))
        self._min_conf = float(min_confianza_control)
        self._limite_min_kmh = max(1, int(limite_min_kmh))
        self._limite_max_kmh = max(self._limite_min_kmh, int(limite_max_kmh))
        self._activar = max(1, int(activar_tras_lecturas))
        self._cambiar = max(1, int(cambiar_tras_lecturas))
        self._retener_invalidas = max(0, int(retener_lecturas_invalidas))
        self._tolerancia_kmh = max(0.0, float(tolerancia_kmh))
        self._exceso_freno_suave = max(0.0, float(exceso_freno_suave_kmh))
        self._exceso_freno_moderado = max(self._exceso_freno_suave, float(exceso_freno_moderado_kmh))
        self._freno_suave = max(0.0, float(freno_suave))
        self._freno_moderado = max(self._freno_suave, float(freno_moderado))

        self._candidato_kmh: int | None = None
        self._candidato_streak = 0
        self._limite_activo_kmh: int | None = None
        self._lecturas_invalidas = 0
        self._ultimo_raw = EstadoLimiteVelocidadHUD()

    def actualizar_lectura(self, estado: EstadoLimiteVelocidadHUD) -> EstadoLimiteVelocidadAplicado:
        previo = self._limite_activo_kmh
        self._ultimo_raw = estado

        if self._lectura_valida(estado):
            self._lecturas_invalidas = 0
            limite = int(estado.limite_kmh)
            if limite == self._candidato_kmh:
                self._candidato_streak += 1
            else:
                self._candidato_kmh = limite
                self._candidato_streak = 1

            if self._limite_activo_kmh is None:
                if self._candidato_streak >= self._activar:
                    self._limite_activo_kmh = self._candidato_kmh
            elif limite == self._limite_activo_kmh:
                self._candidato_kmh = limite
                self._candidato_streak = 1
            elif self._candidato_streak >= self._cambiar:
                self._limite_activo_kmh = self._candidato_kmh
        else:
            self._lecturas_invalidas += 1
            self._candidato_kmh = None
            self._candidato_streak = 0
            if self._lecturas_invalidas > self._retener_invalidas:
                self._limite_activo_kmh = None

        snapshot = self.estado_actual()
        return EstadoLimiteVelocidadAplicado(
            visible_raw=snapshot.visible_raw,
            confianza_raw=snapshot.confianza_raw,
            limite_raw_kmh=snapshot.limite_raw_kmh,
            activo=snapshot.activo,
            limite_activo_kmh=snapshot.limite_activo_kmh,
            cap_velocidad_norm=snapshot.cap_velocidad_norm,
            freno_minimo=snapshot.freno_minimo,
            exceso_kmh=snapshot.exceso_kmh,
            cambio_estado=previo != self._limite_activo_kmh,
        )

    def estado_actual(
        self,
        *,
        velocidad_actual_kmh: int | None = None,
        velocidad_actual_norm: float | None = None,
    ) -> EstadoLimiteVelocidadAplicado:
        cap_norm = None
        freno = 0.0
        exceso = 0.0
        activo = self._limite_activo_kmh is not None
        if activo:
            cap_kmh = min(self._max_kmh_norm, max(0.0, self._limite_activo_kmh + self._tolerancia_kmh))
            cap_norm = min(1.0, max(0.0, cap_kmh / self._max_kmh_norm))
            if velocidad_actual_kmh is not None:
                exceso = max(0.0, float(velocidad_actual_kmh) - cap_kmh)
            elif velocidad_actual_norm is not None:
                exceso = max(0.0, float(velocidad_actual_norm) * self._max_kmh_norm - cap_kmh)

            if exceso >= self._exceso_freno_moderado:
                freno = self._freno_moderado
            elif exceso >= self._exceso_freno_suave:
                freno = self._freno_suave

        return EstadoLimiteVelocidadAplicado(
            visible_raw=bool(self._ultimo_raw.visible),
            confianza_raw=float(self._ultimo_raw.confianza),
            limite_raw_kmh=self._ultimo_raw.limite_kmh,
            activo=activo,
            limite_activo_kmh=self._limite_activo_kmh,
            cap_velocidad_norm=cap_norm,
            freno_minimo=freno,
            exceso_kmh=exceso,
        )

    def _lectura_valida(self, estado: EstadoLimiteVelocidadHUD) -> bool:
        return (
            estado.visible
            and estado.limite_kmh is not None
            and estado.confianza >= self._min_conf
            and self._limite_min_kmh <= int(estado.limite_kmh) <= self._limite_max_kmh
        )
