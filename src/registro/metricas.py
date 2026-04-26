"""Agrega métricas de sesión: FPS, latencia, cumplimiento de señales."""
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MetricasSesion:
    latencias_ms: list[float] = field(default_factory=list)
    fps_muestras: list[float] = field(default_factory=list)
    semaforos_respetados: int = 0
    semaforos_total: int = 0
    altos_respetados: int = 0
    altos_total: int = 0
    eventos_seguridad: int = 0
    t_inicio: float = field(default_factory=time.monotonic)

    def registrar_frame(self, fps: float, latencia_ms: float) -> None:
        self.fps_muestras.append(fps)
        self.latencias_ms.append(latencia_ms)

    def registrar_semaforo(self, respetado: bool) -> None:
        self.semaforos_total += 1
        if respetado:
            self.semaforos_respetados += 1

    def registrar_alto(self, respetado: bool) -> None:
        self.altos_total += 1
        if respetado:
            self.altos_respetados += 1

    def registrar_evento_seguridad(self) -> None:
        self.eventos_seguridad += 1

    def resumen(self) -> dict:
        lat = np.array(self.latencias_ms) if self.latencias_ms else np.array([0.0])
        fps = np.array(self.fps_muestras) if self.fps_muestras else np.array([0.0])
        duracion = time.monotonic() - self.t_inicio
        return {
            "duracion_s": round(duracion, 1),
            "frames_totales": len(self.fps_muestras),
            "fps_promedio": round(float(fps.mean()), 2),
            "fps_minimo": round(float(fps.min()), 2),
            "fps_p95": round(float(np.percentile(fps, 5)), 2),  # p5 de FPS = percentil bajo
            "latencia_ms_media": round(float(lat.mean()), 2),
            "latencia_ms_p95": round(float(np.percentile(lat, 95)), 2),
            "semaforos_cumplimiento": (
                round(self.semaforos_respetados / self.semaforos_total, 3)
                if self.semaforos_total > 0 else None
            ),
            "altos_cumplimiento": (
                round(self.altos_respetados / self.altos_total, 3)
                if self.altos_total > 0 else None
            ),
            "eventos_seguridad": self.eventos_seguridad,
        }
