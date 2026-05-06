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
_TIMEOUT_MS_DEFAULT = 5000


class MonitorSeguridad:
    def __init__(
        self,
        en_paro: Callable[[], None],
        tecla_paro: str = _TECLA_PARO_DEFAULT,
        timeout_ms: int = _TIMEOUT_MS_DEFAULT,
    ):
        self._en_paro = en_paro
        self._tecla_paro = tecla_paro.lower()
        self._timeout_s = timeout_ms / 1000.0
        self._activo = False
        self._ultimo_heartbeat = time.monotonic()
        self._hilo_watchdog: Optional[threading.Thread] = None
        self._hilo_tecla: Optional[threading.Thread] = None
        self._paro_activado = False
        self._listener = None
        atexit.register(self._limpiar)

    def iniciar(self) -> None:
        self._activo = True
        self._ultimo_heartbeat = time.monotonic()

        # Hilo watchdog
        self._hilo_watchdog = threading.Thread(
            target=self._loop_watchdog, daemon=True, name="monitor-watchdog"
        )
        self._hilo_watchdog.start()

        # Hilo de tecla de paro (pynput, no requiere admin)
        self._hilo_tecla = threading.Thread(
            target=self._loop_tecla, daemon=True, name="monitor-tecla"
        )
        self._hilo_tecla.start()

        logger.info(
            "Monitor de seguridad iniciado (tecla=%s, timeout=%.1fs)",
            self._tecla_paro.upper(), self._timeout_s
        )

    def heartbeat(self) -> None:
        self._ultimo_heartbeat = time.monotonic()

    def detener(self) -> None:
        self._activo = False
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass

    def paro_activado(self) -> bool:
        return self._paro_activado

    # ── Hilos internos ────────────────────────────────────────────────────────

    def _loop_watchdog(self) -> None:
        while self._activo:
            elapsed = time.monotonic() - self._ultimo_heartbeat
            if elapsed > self._timeout_s:
                logger.error(
                    "Watchdog: sin heartbeat por %.1fs (límite %.1fs) — activando paro",
                    elapsed, self._timeout_s
                )
                self._activar_paro()
                break
            time.sleep(0.1)

    def _loop_tecla(self) -> None:
        try:
            from pynput import keyboard as kb

            # Construir el conjunto de teclas que disparan el paro
            tecla_objetivo = self._tecla_paro  # ej. "f12"

            def on_press(key):
                if not self._activo:
                    return False  # detiene el listener
                try:
                    nombre = key.name if hasattr(key, "name") else str(key)
                    if nombre.lower() == tecla_objetivo:
                        logger.warning("Tecla %s presionada — activando paro", tecla_objetivo.upper())
                        self._activar_paro()
                        return False
                except Exception:
                    pass

            self._listener = kb.Listener(on_press=on_press)
            self._listener.start()
            self._listener.join()

        except Exception as e:
            logger.warning("No se pudo iniciar listener de teclado (%s) — solo watchdog activo", e)

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
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
