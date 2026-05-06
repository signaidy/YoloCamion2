"""Logger JSONL: un evento por línea, fácil de parsear con pandas."""
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LoggerJSONL:
    def __init__(self, ruta_base: str | Path, nombre_sesion: str | None = None):
        ruta_base = Path(ruta_base)
        ruta_base.mkdir(parents=True, exist_ok=True)
        nombre = nombre_sesion or f"sesion_{int(time.time())}"
        self._ruta = ruta_base / f"{nombre}.jsonl"
        self._f = self._ruta.open("a", encoding="utf-8")
        logger.info("Logger JSONL iniciado: %s", self._ruta)

    def evento(self, tipo: str, datos: dict[str, Any]) -> None:
        linea = json.dumps({"t": time.monotonic(), "tipo": tipo, **datos}, ensure_ascii=False)
        self._f.write(linea + "\n")
        self._f.flush()

    def frame(self, indice: int, fps: float) -> None:
        self.evento("frame", {"indice": indice, "fps": round(fps, 2)})

    def decision(self, regla: int, accion: str, estado: str, razon: str) -> None:
        self.evento("decision", {"regla": regla, "accion": accion, "estado": estado, "razon": razon})

    def transicion(self, de: str, a: str, regla: int) -> None:
        self.evento("transicion", {"de": de, "a": a, "regla": regla})

    def error(self, msg: str) -> None:
        self.evento("error", {"msg": msg})

    def seguridad(self, msg: str) -> None:
        self.evento("seguridad", {"msg": msg})

    def cerrar(self) -> None:
        self._f.close()
        logger.info("Logger cerrado: %s", self._ruta)

    @property
    def ruta(self) -> Path:
        return self._ruta
