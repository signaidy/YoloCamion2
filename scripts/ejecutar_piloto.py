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
from src.control.gamepad_pid import ControladorGamepadPID
from src.decision import FSMDecision
from src.fuente import FuentePantalla, FuenteVideo
from src.fuente.buffer import FuenteConBuffer
from src.fuente.ventana import FuenteVentana, buscar_ventana
from src.percepcion import AnalizadorContexto, Tracker
from src.percepcion.contexto import cargar_rois_yaml
from src.percepcion.carriles import DetectorCarriles
from src.percepcion.fisica import EstimadorFisicaVisual
from src.percepcion.flujo_optico import EstimadorFlujoOpticoLK
from src.percepcion.velocidad_propia import EstimadorVelocidadPropia
from src.registro import GrabadorVideo, LoggerJSONL, MetricasSesion
from src.seguridad import MonitorSeguridad
from src.tipos import Accion, ComandoControl, SetpointControl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("piloto")

# Acciones con giro intencional: el FSM ya fija desviacion_volante segun la
# accion y el detector de carriles NO debe sobreescribirla (es maniobra activa,
# no seguimiento de carril).
_ACCIONES_CON_GIRO: frozenset[Accion] = frozenset({
    Accion.GIRAR_IZQ, Accion.GIRAR_DER,
    Accion.REBASAR_IZQ, Accion.REBASAR_DER,
})


def _setpoint_a_comando(sp: SetpointControl) -> ComandoControl:
    """Adaptador: SetpointControl -> ComandoControl para controladores no-PID."""
    return ComandoControl(
        acelerador=sp.velocidad_objetivo_norm,
        freno=sp.freno_objetivo,
        volante=sp.desviacion_volante,
        timestamp=time.monotonic(),
    )


def _roi_franja_inferior(w: int, h: int) -> tuple[int, int, int, int]:
    """ROI para flujo optico de velocidad propia: franja inferior central."""
    return (
        int(round(0.30 * w)),
        int(round(0.78 * h)),
        int(round(0.70 * w)),
        int(round(0.97 * h)),
    )


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
        return FuenteConBuffer(pantalla)
    elif tipo == "ventana":
        titulo = cfg["fuente"].get("titulo_ventana", "Euro Truck Simulator 2")
        escalar = cfg["fuente"].get("escalar_a", [1920, 1080])
        ventana = FuenteVentana(
            titulo=titulo,
            escalar_a=tuple(escalar) if escalar else None,
        )
        return FuenteConBuffer(ventana)
    raise ValueError(f"Tipo de fuente desconocido: {tipo}")


def construir_controlador(cfg: dict, tipo_override: str | None = None):
    tipo = tipo_override or cfg["control"]["tipo"]
    if tipo == "nulo":
        return ControladorNulo()
    elif tipo == "gamepad":
        # Pure-vision: gamepad analogico con tres PIDs (Tarea 3.2-3.4)
        ctrl = ControladorGamepadPID()
        ctrl.iniciar()
        return ctrl
    elif tipo == "gamepad_directo":
        # Fallback sin PID: pasthrough analogico (no recomendado)
        ctrl = ControladorGamepad()
        ctrl.iniciar()
        return ctrl
    elif tipo == "teclado":
        return ControladorTeclado()
    raise ValueError(f"Tipo de control desconocido: {tipo}")


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
    parser.add_argument("--debug-carril", action="store_true",
                        help="Mostrar estado del carril cada 30 frames para calibración")
    parser.add_argument("--debug-carril-img", action="store_true",
                        help="Guardar imagen de debug con líneas detectadas cada 60 frames")
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
    estimador_fisica = EstimadorFisicaVisual()
    contexto = AnalizadorContexto(rois=rois, estimador_fisica=estimador_fisica)
    fsm = FSMDecision()
    detector_carriles = DetectorCarriles()
    controlador = construir_controlador(cfg)

    # Capa pure-vision para velocidad propia (RNF-07): flujo optico LK
    # restringido a franja inferior central, normalizado a [0, 1].
    escalar = cfg["fuente"].get("escalar_a") or [1920, 1080]
    w_capt, h_capt = int(escalar[0]), int(escalar[1])
    cfg_velprop = cfg.get("velocidad_propia", {}) or {}
    estimador_flujo = EstimadorFlujoOpticoLK(roi=_roi_franja_inferior(w_capt, h_capt))
    estimador_velocidad = EstimadorVelocidadPropia(
        factor_calibracion=float(cfg_velprop.get("factor_calibracion", 0.01)),
        alpha_ema=float(cfg_velprop.get("alpha_ema", 0.3)),
    )
    velocidad_actual_norm = 0.0
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
        # Cache del último resultado YOLO/FSM — se actualiza cada YOLO_CADA frames
        YOLO_CADA = 3        # YOLO cada 3 frames → ~10 FPS detección, ~30 FPS carril
        yolo_contador = 0
        resultado_cache = None

        # EMA de la desviación lateral del carril.
        # alpha=0.50: respuesta más rápida en curvas; el deque interno de DetectorCarriles
        # (4 frames) ya aporta suavizado base, así que no necesitamos más inercia aquí.
        _ALPHA_EMA_CARRIL = 0.50
        desv_ema: float = 0.0

        from src.decision.estado import EstadoFSM
        _ESTADOS_CARRIL = (
            EstadoFSM.CONDUCIENDO_NORMAL,
            EstadoFSM.SIGUIENDO_VEHICULO,
            EstadoFSM.RECUPERACION,
        )

        logger.info("Pipeline iniciado — presiona %s para parar", cfg["seguridad"]["tecla_paro"].upper())
        logger.info("YOLO cada %d frames | Carril cada frame", YOLO_CADA)

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
                if not fuente.esta_activa:
                    break
                time.sleep(0.005)
                continue

            # ── Detección de carril (cada frame — rápida ~3ms) ───────────────
            carril = detector_carriles.detectar(cuadro.imagen)

            # Actualizar EMA de desviación: cuando no hay detección confiable,
            # dejamos que la EMA decaiga hacia 0 para no aplicar una corrección
            # obsoleta (el camión quizás ya se centró solo).
            if carril.confianza >= 0.5:
                desv_ema = (_ALPHA_EMA_CARRIL * carril.desviacion
                            + (1.0 - _ALPHA_EMA_CARRIL) * desv_ema)
            else:
                desv_ema *= 0.85  # decaimiento rápido sin señal confiable

            # ── Velocidad propia visual (pure-vision, RNF-07) ────────────────
            flujo_lk = estimador_flujo.calcular(cuadro.imagen, cuadro.timestamp)
            velocidad_actual_norm = estimador_velocidad.estimar(flujo_lk)
            if isinstance(controlador, ControladorGamepadPID):
                controlador.actualizar_velocidad_actual(velocidad_actual_norm)

            # ── YOLO + FSM (cada YOLO_CADA frames — lenta ~100ms) ───────────
            yolo_contador += 1
            if yolo_contador >= YOLO_CADA or resultado_cache is None:
                yolo_contador = 0
                seguimientos = tracker.rastrear(cuadro.imagen)
                escena = contexto.analizar(seguimientos, cuadro.imagen, cuadro.timestamp)
                resultado_cache = fsm.decidir(escena)

            resultado = resultado_cache

            # ── Setpoint del FSM (mutable) + override de carril ──────────────
            setpoint = SetpointControl(
                velocidad_objetivo_norm=resultado.setpoint.velocidad_objetivo_norm,
                freno_objetivo=resultado.setpoint.freno_objetivo,
                desviacion_volante=resultado.setpoint.desviacion_volante,
            )

            # Override de carril:
            #  - confianza >= 0.85: solo cuando se detecta la línea derecha (o ambas).
            #    La detección solo-izquierda (conf=0.50) es poco fiable en tráfico
            #    denso y produce giros erróneos al confundir luces/vehículos con líneas.
            #  - abs(desv_ema) > 0.06: zona muerta ligeramente más sensible que el
            #    deque interno (0.07) para que correcciones pequeñas no se pierdan.
            #  - velocidad > 0.05: evitar bucle muerto cuando el camión está parado.
            if (resultado.accion not in _ACCIONES_CON_GIRO
                    and resultado.estado_nuevo in _ESTADOS_CARRIL
                    and carril.confianza >= 0.85
                    and abs(desv_ema) > 0.06
                    and velocidad_actual_norm > 0.05):
                desv = float(max(-1.0, min(1.0, desv_ema)))
                setpoint.desviacion_volante = desv

            if args.debug_carril and n_frame % 30 == 0:
                logger.info(
                    "CARRIL desv=%+.3f ema=%+.3f conf=%.2f izq=%s der=%s vol=%+.2f vel=%.2f",
                    carril.desviacion, desv_ema, carril.confianza,
                    "✓" if carril.linea_izq_detectada else "✗",
                    "✓" if carril.linea_der_detectada else "✗",
                    setpoint.desviacion_volante, velocidad_actual_norm,
                )

            if args.debug_carril_img and n_frame % 60 == 0:
                import cv2 as _cv2
                from pathlib import Path as _Path
                dbg = detector_carriles.debug_frame(cuadro.imagen)
                ruta_dbg = _Path(cfg["registro"]["ruta_base"]) / f"debug_carril_{n_frame:06d}.jpg"
                _cv2.imwrite(str(ruta_dbg), dbg)
                logger.info("Debug imagen guardada: %s", ruta_dbg)

            if isinstance(controlador, ControladorGamepadPID):
                controlador.aplicar(setpoint)
            else:
                controlador.aplicar(_setpoint_a_comando(setpoint))

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
