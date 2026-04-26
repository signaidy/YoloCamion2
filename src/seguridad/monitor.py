"""Monitor de seguridad: escucha F12 (paro duro) y watchdog de heartbeat.

Corre en un hilo separado. Si el loop principal no llama a heartbeat()
en `timeout_ms` ms, libera los controles y termina el programa.
"""
import atexit
import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_TECLA_PARO_DEFAULT = "f12"
_TIMEOUT_MS_DEFAULT = 500


class MonitorSeguridad:
    def __init__(
        self,
        en_paro: Callable[[], None],
        tecla_paro: str = _TECLA_PARO_DEFAULT,
        timeout_ms: int = _TIMEOUT_MS_DEFAULT,
    ):
        self._en_paro = en_paro
        self._tecla_paro = tecla_paro
        self._timeout_s = timeout_ms / 1000.0
        self._activo = False
        self._ultimo_heartbeat = time.monotonic()
        self._hilo: Optional[threading.Thread] = None
        self._paro_activado = False
        atexit.register(self._limpiar)

    def iniciar(self) -> None:
        self._activo = True
        self._ultimo_heartbeat = time.monotonic()
        self._hilo = threading.Thread(target=self._loop, daemon=True, name="monitor-seg")
        self._hilo.start()
        logger.info("Monitor de seguridad iniciado (tecla=%s, timeout=%dms)",
                    self._tecla_paro, int(self._timeout_s * 1000))

    def heartbeat(self) -> None:
        self._ultimo_heartbeat = time.monotonic()

    def detener(self) -> None:
        self._activo = False

    def paro_activado(self) -> bool:
        return self._paro_activado

    def _loop(self) -> None:
        try:
            import keyboard
            keyboard.add_hotkey(self._tecla_paro, self._activar_paro)
        except Exception:
            logger.warning("No se pudo registrar hotkey '%s' — solo watchdog activo", self._tecla_paro)

        while self._activo:
            elapsed = time.monotonic() - self._ultimo_heartbeat
            if elapsed > self._timeout_s:
                logger.error("Watchdog: sin heartbeat por %.1fs — activando paro", elapsed)
                self._activar_paro()
                break
            time.sleep(0.05)

    def _activar_paro(self) -> None:
        if not self._paro_activado:
            self._paro_activado = True
            logger.critical("PARO DE EMERGENCIA activado")
            try:
                self._en_paro()
            except Exception as e:
                logger.error("Error en callback de paro: %s", e)

    def _limpiar(self) -> None:
        self._activo = False
