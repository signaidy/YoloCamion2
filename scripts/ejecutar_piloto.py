"""Punto de entrada principal del sistema de conducción autónoma.

Integra los 7 módulos del pipeline:
  fuente → tracker → contexto → FSM → control → registro → seguridad

Uso:
  python scripts/ejecutar_piloto.py                     # usa config/default.yaml
  python scripts/ejecutar_piloto.py --config mi.yaml
  python scripts/ejecutar_piloto.py --control gamepad   # sobreescribe control
  python scripts/ejecutar_piloto.py --fuente pantalla   # captura en vivo
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# Asegurar que src/ está en el path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.control import ControladorGamepad, ControladorNulo, ControladorTeclado
from src.decision import FSMDecision
from src.fuente import FuentePantalla, FuenteVideo
from src.fuente.buffer import FuenteConBuffer
from src.percepcion import AnalizadorContexto, Tracker
from src.percepcion.contexto import cargar_rois_yaml
from src.registro import GrabadorVideo, LoggerJSONL, MetricasSesion
from src.seguridad import MonitorSeguridad
from src.tipos import Accion, ComandoControl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("piloto")

# Mapa acción → ComandoControl (valores del diseño)
_MAPA_ACCION: dict[Accion, tuple[float, float, float]] = {
    Accion.MANTENER:      (0.3, 0.0, 0.0),
    Accion.ACELERAR:      (0.6, 0.0, 0.0),
    Accion.FRENAR_SUAVE:  (0.0, 0.4, 0.0),
    Accion.FRENAR_FUERTE: (0.0, 0.8, 0.0),
    Accion.ALTO_TOTAL:    (0.0, 1.0, 0.0),
    Accion.GIRAR_IZQ:     (0.2, 0.0, -0.5),
    Accion.GIRAR_DER:     (0.2, 0.0,  0.5),
    Accion.REBASAR_IZQ:   (0.8, 0.0, -0.3),
    Accion.REBASAR_DER:   (0.8, 0.0,  0.3),
    Accion.ESPERAR:       (0.0, 0.0,  0.0),
}


def cargar_config(ruta: str) -> dict:
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def construir_fuente(cfg: dict):
    tipo = cfg["fuente"]["tipo"]
    if tipo == "video":
        return FuenteVideo(cfg["fuente"]["ruta_video"])
    elif tipo == "pantalla":
        escalar = cfg["fuente"].get("escalar_a")
        pantalla = FuentePantalla(
            monitor=cfg["fuente"].get("monitor", 0),
            region=cfg["fuente"].get("region"),
            escalar_a=tuple(escalar) if escalar else None,
        )
        return FuenteConBuffer(pantalla)  # captura en hilo separado para no bloquear YOLO
    raise ValueError(f"Tipo de fuente desconocido: {tipo}")


def construir_controlador(cfg: dict, tipo_override: str | None = None):
    tipo = tipo_override or cfg["control"]["tipo"]
    if tipo == "nulo":
        return ControladorNulo()
    elif tipo == "gamepad":
        ctrl = ControladorGamepad()
        ctrl.iniciar()
        return ctrl
    elif tipo == "teclado":
        return ControladorTeclado()
    raise ValueError(f"Tipo de control desconocido: {tipo}")


def accion_a_comando(accion: Accion) -> ComandoControl:
    acel, freno, vol = _MAPA_ACCION.get(accion, (0.0, 0.0, 0.0))
    return ComandoControl(acelerador=acel, freno=freno, volante=vol, timestamp=time.monotonic())


def countdown(segundos: int) -> None:
    """Cuenta regresiva visible en consola para que el usuario cambie al juego."""
    print("\n" + "="*50)
    print("  Cambia al juego ETS2 AHORA")
    print("  El piloto arrancará en:")
    for i in range(segundos, 0, -1):
        print(f"    {i}...", flush=True)
        time.sleep(1)
    print("  ¡INICIANDO!\n" + "="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Piloto autónomo ETS2")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--control", default=None, help="Sobreescribir tipo de control")
    parser.add_argument("--fuente", default=None, help="Sobreescribir tipo de fuente")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = sin límite")
    parser.add_argument("--delay", type=int, default=0,
                        help="Segundos de countdown antes de arrancar (útil para cambiar al juego)")
    parser.add_argument("--sin-video", action="store_true",
                        help="No grabar video (más rápido, recomendado para pruebas en vivo)")
    args = parser.parse_args()

    cfg = cargar_config(args.config)

    # Sobreescrituras de CLI
    if args.control:
        cfg["control"]["tipo"] = args.control
    if args.fuente:
        cfg["fuente"]["tipo"] = args.fuente

    logger.info("=== Iniciando piloto autónomo ETS2 ===")
    logger.info("Fuente: %s | Control: %s", cfg["fuente"]["tipo"], cfg["control"]["tipo"])

    if args.delay > 0:
        countdown(args.delay)

    # Cargar ROI calibradas
    ruta_rois = Path("config/regiones_interes.yaml")
    rois = cargar_rois_yaml(ruta_rois) if ruta_rois.exists() else None
    if rois:
        logger.info("ROI cargadas desde %s (%d regiones)", ruta_rois, len(rois))
    else:
        logger.warning("No se encontró %s — usando ROI por defecto", ruta_rois)

    # Construir componentes
    fuente = construir_fuente(cfg)
    tracker = Tracker(
        ruta_modelo=cfg["modelo"]["pesos"],
        confianza_min=cfg["modelo"]["conf_min"],
        imgsz=cfg["modelo"]["imgsz"],
        device=cfg["modelo"]["device"],
    )
    contexto = AnalizadorContexto(rois=rois)
    fsm = FSMDecision()
    controlador = construir_controlador(cfg)
    metricas = MetricasSesion()
    log = LoggerJSONL(cfg["registro"]["ruta_base"])
    grabar = cfg["registro"]["grabar_video"] and not args.sin_video
    grabador = GrabadorVideo(cfg["registro"]["ruta_base"]) if grabar else None
    if args.sin_video:
        logger.info("Grabación de video desactivada (--sin-video)")

    def en_paro():
        fsm.activar_paro_manual()
        controlador.liberar()
        log.seguridad("paro de emergencia activado")
        metricas.registrar_evento_seguridad()

    monitor = MonitorSeguridad(
        en_paro=en_paro,
        tecla_paro=cfg["seguridad"]["tecla_paro"],
        timeout_ms=cfg["seguridad"]["timeout_watchdog_ms"],
    )

    try:
        logger.info("Cargando modelo YOLO...")
        tracker.cargar()

        # Warmup: pre-compila kernels CUDA para que el primer frame real sea rápido
        # Sin esto, la primera inferencia tarda ~3s y el watchdog dispararía
        logger.info("Warmup YOLO (compilando kernels CUDA)...")
        import numpy as _np
        _frame_dummy = _np.zeros((1080, 1920, 3), dtype=_np.uint8)
        tracker.rastrear(_frame_dummy)
        logger.info("Warmup completado — CUDA listo")

        fuente.iniciar()
        monitor.iniciar()

        primer_frame = True
        estado_anterior = fsm.estado_actual
        n_frame = 0

        logger.info("Pipeline iniciado — presiona %s para parar", cfg["seguridad"]["tecla_paro"].upper())

        while fuente.esta_activa:
            if monitor.paro_activado():
                break
            if args.max_frames > 0 and n_frame >= args.max_frames:
                logger.info("Límite de frames alcanzado (%d)", args.max_frames)
                break

            monitor.heartbeat()
            t0 = time.perf_counter()

            cuadro = fuente.siguiente()
            if cuadro is None:
                # FuenteConBuffer devuelve None mientras espera el primer frame
                # o entre frames. Solo salir si la fuente ya no está activa.
                if not fuente.esta_activa:
                    break
                time.sleep(0.01)
                continue

            # ── Percepción ──────────────────────────────────────────────────
            seguimientos = tracker.rastrear(cuadro.imagen)
            escena = contexto.analizar(seguimientos, cuadro.imagen)

            # ── Decisión ────────────────────────────────────────────────────
            resultado = fsm.decidir(escena)

            # ── Control ─────────────────────────────────────────────────────
            cmd = accion_a_comando(resultado.accion)
            controlador.aplicar(cmd)

            # ── Registro ────────────────────────────────────────────────────
            latencia_ms = (time.perf_counter() - t0) * 1000
            metricas.registrar_frame(cuadro.fps_instantaneo, latencia_ms)
            log.frame(cuadro.indice, cuadro.fps_instantaneo)
            log.decision(resultado.regla, resultado.accion.value,
                         resultado.estado_nuevo.value, resultado.razon)

            if resultado.estado_nuevo != estado_anterior:
                log.transicion(estado_anterior.value, resultado.estado_nuevo.value, resultado.regla)
                logger.info("[R%d] %s → %s | %s",
                            resultado.regla, estado_anterior.value,
                            resultado.estado_nuevo.value, resultado.razon)
                estado_anterior = resultado.estado_nuevo

            if grabador is not None:
                if primer_frame:
                    h, w = cuadro.imagen.shape[:2]
                    grabador.iniciar(w, h)
                    primer_frame = False
                grabador.escribir_frame(
                    cuadro.imagen, seguimientos, resultado.accion,
                    resultado.estado_nuevo.value, cuadro.fps_instantaneo, resultado.regla,
                )

            n_frame += 1

    except KeyboardInterrupt:
        logger.info("Interrumpido por el usuario (Ctrl+C)")
    finally:
        monitor.detener()
        controlador.liberar()
        controlador.cerrar()
        fuente.cerrar()
        if grabador:
            grabador.cerrar()
        log.cerrar()

        resumen = metricas.resumen()
        logger.info("=== Sesión terminada ===")
        for k, v in resumen.items():
            logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
