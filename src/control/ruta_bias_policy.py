"""Politica de sesgo lateral guiado por ruta.

Fase 3 inicial: consume la salida observacional del minimapa y la convierte
en un sesgo lateral suave para pre-posicionar el camión hacia el carril
objetivo antes de una salida o bifurcación.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.tipos import EstadoEscena, EstadoRuta, ManiobraRuta


@dataclass(frozen=True)
class EstadoRutaAplicada:
    activo: bool = False
    retenido: bool = False
    bloqueado_por_satisfaccion: bool = False
    bloqueado_por_espejo: bool = False
    bloqueado_por_lateral: bool = False
    sin_carril_objetivo: bool = False
    confianza_raw: float = 0.0
    maniobra: ManiobraRuta = ManiobraRuta.DESCONOCIDA
    ramal_objetivo: str = "desconocido"
    sesgo_lateral_raw: float = 0.0
    sesgo_lateral_aplicado: float = 0.0
    requiere_cambio_carril: bool = False


class GobernadorRutaLateral:
    def __init__(
        self,
        min_confianza_aplicar: float = 0.60,
        activar_tras_lecturas: int = 2,
        activar_salida_fuerte_confianza: float = 0.95,
        activar_salida_fuerte_distancia: float = 0.08,
        retener_recto_lecturas: int = 6,
        retener_invalida_lecturas: int = 3,
        cambiar_opuesto_tras_lecturas: int = 2,
        factor_sesgo_lateral: float = 0.55,
        satisfacer_tras_lecturas_sin_carril: int = 2,
        factor_sesgo_satisfecha: float = 0.45,
        reset_satisfecha_tras_neutrales: int = 8,
    ) -> None:
        self.min_confianza_aplicar = float(min_confianza_aplicar)
        self.activar_tras_lecturas = max(1, int(activar_tras_lecturas))
        self.activar_salida_fuerte_confianza = float(activar_salida_fuerte_confianza)
        self.activar_salida_fuerte_distancia = max(0.0, float(activar_salida_fuerte_distancia))
        self.retener_recto_lecturas = max(0, int(retener_recto_lecturas))
        self.retener_invalida_lecturas = max(0, int(retener_invalida_lecturas))
        self.cambiar_opuesto_tras_lecturas = max(1, int(cambiar_opuesto_tras_lecturas))
        self.factor_sesgo_lateral = max(0.0, float(factor_sesgo_lateral))
        self.satisfacer_tras_lecturas_sin_carril = max(1, int(satisfacer_tras_lecturas_sin_carril))
        self.factor_sesgo_satisfecha = max(0.0, float(factor_sesgo_satisfecha))
        self.reset_satisfecha_tras_neutrales = max(1, int(reset_satisfecha_tras_neutrales))

        self._activo: EstadoRuta | None = None
        self._retenido = False
        self._lecturas_sin_senal = 0
        self._lado_opuesto_pendiente: str | None = None
        self._lecturas_opuestas = 0
        self._lado_pendiente: str | None = None
        self._estado_pendiente: EstadoRuta | None = None
        self._lecturas_pendientes = 0
        self._lado_satisfecho: str | None = None
        self._neutrales_desde_satisfecha = 0
        self._lecturas_sin_carril_objetivo = 0
        self._sesgo_satisfecho = 0.0
        self._ultima_lectura: EstadoRuta | None = None

    def actualizar_lectura(self, estado: EstadoRuta) -> EstadoRutaAplicada:
        self._ultima_lectura = estado
        self._actualizar_reset_satisfecha(estado)
        if self._es_intencion_aplicable(estado):
            lado = self._lado_estado(estado)
            lado_activo = self._lado_estado(self._activo)
            if self._activo is None or lado == lado_activo:
                if self._es_salida_fuerte(estado):
                    self._activar(estado, retenido=False)
                    return self.estado_actual()
                self._registrar_candidata(estado)
            else:
                if lado == self._lado_opuesto_pendiente:
                    self._lecturas_opuestas += 1
                else:
                    self._lado_opuesto_pendiente = lado
                    self._lecturas_opuestas = 1
                if self._lecturas_opuestas >= self.cambiar_opuesto_tras_lecturas:
                    self._activar(estado, retenido=False)
            return self.estado_actual()

        self._reiniciar_pendiente()
        if self._activo is not None:
            self._lecturas_sin_senal += 1
            self._retenido = True
            self._reiniciar_opuesto()

            if estado.visible and estado.maniobra is ManiobraRuta.SEGUIR_RECTO:
                if self._lecturas_sin_senal > self.retener_recto_lecturas:
                    self._limpiar()
            else:
                if self._lecturas_sin_senal > self.retener_invalida_lecturas:
                    self._limpiar()

        return self.estado_actual()

    def estado_actual(
        self,
        escena: EstadoEscena | None = None,
        hay_carril_objetivo: bool | None = None,
    ) -> EstadoRutaAplicada:
        if self._activo is None:
            sesgo_satisfecho = self._sesgo_satisfecho
            if escena is not None:
                if sesgo_satisfecho > 0.0 and (escena.espejo_der_ocupado or escena.lateral_der_ocupado):
                    sesgo_satisfecho = 0.0
                elif sesgo_satisfecho < 0.0 and (escena.espejo_izq_ocupado or escena.lateral_izq_ocupado):
                    sesgo_satisfecho = 0.0
            return EstadoRutaAplicada(
                bloqueado_por_satisfaccion=self._lado_satisfecho is not None,
                sesgo_lateral_aplicado=float(sesgo_satisfecho),
                sesgo_lateral_raw=float(sesgo_satisfecho),
            )

        activo = self._activo
        sesgo_raw = float(activo.sesgo_lateral_objetivo)
        sesgo_aplicado = sesgo_raw * self.factor_sesgo_lateral
        bloqueado_espejo = False
        bloqueado_lateral = False
        bloqueado_satisfaccion = False
        sin_carril_objetivo = hay_carril_objetivo is False
        lado = self._lado_estado(activo)
        if escena is not None and lado == "der" and escena.espejo_der_ocupado:
            sesgo_aplicado = 0.0
            bloqueado_espejo = True
        elif escena is not None and lado == "izq" and escena.espejo_izq_ocupado:
            sesgo_aplicado = 0.0
            bloqueado_espejo = True

        if escena is not None and lado == "der" and escena.lateral_der_ocupado:
            sesgo_aplicado = 0.0
            bloqueado_lateral = True
        elif escena is not None and lado == "izq" and escena.lateral_izq_ocupado:
            sesgo_aplicado = 0.0
            bloqueado_lateral = True

        if sin_carril_objetivo:
            # Si no existe un carril objetivo lateral confiable, nunca aplicar el
            # sesgo de cambio de carril completo. Esto evita confundir curvas o
            # bordes/banquetas con un carril válido de salida.
            sesgo_aplicado = 0.0
            if self._debe_mantenerse_activa_sin_carril(activo):
                self._lecturas_sin_carril_objetivo = 0
            else:
                self._lecturas_sin_carril_objetivo += 1
                if self._lecturas_sin_carril_objetivo >= self.satisfacer_tras_lecturas_sin_carril:
                    if lado in {"izq", "der"}:
                        self._lado_satisfecho = lado
                        self._neutrales_desde_satisfecha = 0
                        self._sesgo_satisfecho = (
                            sesgo_raw * self.factor_sesgo_lateral * self.factor_sesgo_satisfecha
                        )
                    self._activo = None
                    self._retenido = False
                    self._lecturas_sin_senal = 0
                    self._lecturas_sin_carril_objetivo = 0
                    self._reiniciar_opuesto()
                    self._reiniciar_pendiente()
                    bloqueado_satisfaccion = True
        else:
            self._lecturas_sin_carril_objetivo = 0

        return EstadoRutaAplicada(
            activo=self._activo is not None,
            retenido=bool(self._retenido),
            bloqueado_por_satisfaccion=bloqueado_satisfaccion or self._lado_satisfecho is not None,
            bloqueado_por_espejo=bloqueado_espejo,
            bloqueado_por_lateral=bloqueado_lateral,
            sin_carril_objetivo=sin_carril_objetivo,
            confianza_raw=float(activo.confianza),
            maniobra=activo.maniobra,
            ramal_objetivo=activo.ramal_objetivo,
            sesgo_lateral_raw=sesgo_raw,
            sesgo_lateral_aplicado=float(sesgo_aplicado),
            requiere_cambio_carril=bool(activo.requiere_cambio_carril),
        )

    def _activar(self, estado: EstadoRuta, retenido: bool) -> None:
        self._activo = estado
        self._retenido = retenido
        self._lecturas_sin_senal = 0
        self._lecturas_sin_carril_objetivo = 0
        self._reiniciar_opuesto()
        self._reiniciar_pendiente()

    def _limpiar(self) -> None:
        self._activo = None
        self._retenido = False
        self._lecturas_sin_senal = 0
        self._lecturas_sin_carril_objetivo = 0
        self._reiniciar_opuesto()
        self._reiniciar_pendiente()

    def _reiniciar_opuesto(self) -> None:
        self._lado_opuesto_pendiente = None
        self._lecturas_opuestas = 0

    def _reiniciar_pendiente(self) -> None:
        self._lado_pendiente = None
        self._estado_pendiente = None
        self._lecturas_pendientes = 0

    def _registrar_candidata(self, estado: EstadoRuta) -> None:
        lado = self._lado_estado(estado)
        if lado == self._lado_pendiente:
            self._lecturas_pendientes += 1
            self._estado_pendiente = estado
        else:
            self._lado_pendiente = lado
            self._estado_pendiente = estado
            self._lecturas_pendientes = 1
        if self._lecturas_pendientes >= self.activar_tras_lecturas and self._estado_pendiente is not None:
            self._activar(self._estado_pendiente, retenido=False)

    def _actualizar_reset_satisfecha(self, estado: EstadoRuta) -> None:
        if self._lado_satisfecho is None:
            return
        if self._es_intencion_base(estado):
            self._neutrales_desde_satisfecha = 0
            return
        self._neutrales_desde_satisfecha += 1
        if self._neutrales_desde_satisfecha >= self.reset_satisfecha_tras_neutrales:
            self._limpiar_satisfaccion()

    def _es_intencion_aplicable(self, estado: EstadoRuta) -> bool:
        if not self._es_intencion_base(estado):
            return False
        if self._lado_satisfecho is not None:
            return False
        return True

    def _es_intencion_base(self, estado: EstadoRuta) -> bool:
        if not estado.visible:
            return False
        if estado.confianza < self.min_confianza_aplicar:
            return False
        if not estado.requiere_cambio_carril:
            return False
        if self._lado_estado(estado) not in {"izq", "der"}:
            return False
        return abs(float(estado.sesgo_lateral_objetivo)) > 0.0

    def _es_salida_fuerte(self, estado: EstadoRuta) -> bool:
        if estado.maniobra not in {ManiobraRuta.SALIDA_IZQ, ManiobraRuta.SALIDA_DER}:
            return False
        if estado.confianza < self.activar_salida_fuerte_confianza:
            return False
        if estado.distancia_normalizada is None:
            return False
        return float(estado.distancia_normalizada) <= self.activar_salida_fuerte_distancia

    def _debe_mantenerse_activa_sin_carril(self, estado: EstadoRuta) -> bool:
        if not self._es_maniobra_comprometida(estado):
            return False
        lado = self._lado_estado(estado)
        if lado not in {"izq", "der"}:
            return False
        if self._lectura_confirma_mismo_lado(self._ultima_lectura, lado):
            return True
        return bool(self._retenido)

    def _limpiar_satisfaccion(self) -> None:
        self._lado_satisfecho = None
        self._neutrales_desde_satisfecha = 0
        self._sesgo_satisfecho = 0.0

    @staticmethod
    def _es_maniobra_comprometida(estado: EstadoRuta) -> bool:
        return estado.maniobra in {
            ManiobraRuta.SALIDA_IZQ,
            ManiobraRuta.SALIDA_DER,
            ManiobraRuta.GIRO_IZQ,
            ManiobraRuta.GIRO_DER,
        }

    @classmethod
    def _lectura_confirma_mismo_lado(cls, estado: EstadoRuta | None, lado_objetivo: str) -> bool:
        if estado is None or not estado.visible:
            return False
        if not estado.requiere_cambio_carril:
            return False
        if abs(float(estado.sesgo_lateral_objetivo)) <= 0.0:
            return False
        return cls._lado_estado(estado) == lado_objetivo

    @staticmethod
    def _lado_estado(estado: EstadoRuta | None) -> str | None:
        if estado is None:
            return None
        if estado.ramal_objetivo in {"izq", "der"}:
            return estado.ramal_objetivo
        sesgo = float(estado.sesgo_lateral_objetivo)
        if sesgo > 0.0:
            return "der"
        if sesgo < 0.0:
            return "izq"
        return None
