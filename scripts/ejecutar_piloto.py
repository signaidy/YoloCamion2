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
import numpy as np

try:
    from pynput.keyboard import Controller as _KeyboardController
    _keyboard_ctrl = _KeyboardController()
except Exception:
    _keyboard_ctrl = None

# Asegurar que src/ está en el path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.control import ControladorGamepad, ControladorNulo, ControladorTeclado
from src.control.carril_steering_policy import comando_direccion_por_carril
from src.control.carril_speed_policy import limites_velocidad_por_carril
from src.control.velocidad_feedback_policy import velocidad_feedback_para_control
from src.control.gamepad_pid import ControladorGamepadPID
from src.control.velocidad_fail_safe import limites_por_velocidad_desconocida
from src.decision import FSMDecision
from src.fuente import FuentePantalla, FuenteVideo
from src.fuente.buffer import FuenteConBuffer
from src.fuente.ventana import FuenteVentana, buscar_ventana
from src.percepcion import AnalizadorContexto, Tracker
from src.percepcion.contexto import cargar_rois_yaml
from src.percepcion.carriles import DetectorCarriles
from src.percepcion.yolop_inference import InferenciaYOLOP
from src.percepcion.analisis_carriles import AnalizadorCarriles, superponer_carriles
from src.control.pure_pursuit import PurePursuitVisual
from src.percepcion.fisica import EstimadorFisicaVisual
from src.percepcion.velocidad_dashboard import EstimadorVelocidadDashboard
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

_MIN_PIXELES_LL_YOLOP = 800
_MIN_PIXELES_LL_LADO_YOLOP = 250
def _setpoint_a_comando(sp: SetpointControl) -> ComandoControl:
    """Adaptador: SetpointControl -> ComandoControl para controladores no-PID."""
    return ComandoControl(
        acelerador=sp.velocidad_objetivo_norm,
        freno=sp.freno_objetivo,
        volante=sp.desviacion_volante,
        timestamp=time.monotonic(),
    )


def _ll_yolop_valida(ll_mask_roi: np.ndarray) -> tuple[bool, int, int, int]:
    """Valida que YOLOP vea marcas suficientes a ambos lados del camion."""
    alto, ancho = ll_mask_roi.shape
    total_full = int(np.count_nonzero(ll_mask_roi))
    # Las marcas de carril detectadas por YOLOP aparecen en filas 25–55% del frame.
    # El rango anterior (66–92%) estaba por debajo de donde las líneas son visibles.
    y0 = int(round(alto * 0.35))
    y1 = int(round(alto * 0.62))
    x_ref = ancho // 2
    half = int(round(ancho * 0.30))
    x0 = max(0, x_ref - half)
    x2 = min(ancho, x_ref + half)

    roi = ll_mask_roi[y0:y1, x0:x2]
    xs = np.nonzero(roi)[1]
    total = int(xs.size)
    if total == 0:
        return False, total_full, 0, 0

    x_split = x_ref - x0
    pix_izq = int(np.count_nonzero(xs < x_split))
    pix_der = int(np.count_nonzero(xs >= x_split))
    valida = (
        total >= _MIN_PIXELES_LL_YOLOP
        and pix_izq >= _MIN_PIXELES_LL_LADO_YOLOP
        and pix_der >= _MIN_PIXELES_LL_LADO_YOLOP
        and max(pix_izq, pix_der) / min(pix_izq, pix_der) <= 3.0
    )
    return valida, total_full, pix_izq, pix_der


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
    parser.add_argument("--debug-yolop", action="store_true",
                        help="Guardar imagen compuesta cada 60 frames: entrada del modelo | máscaras superpuestas")
    parser.add_argument("--debug-clasif-carriles", action="store_true",
                        help="Guardar imagen con clasificación de carriles (ego/contrario/mismo) cada 60 frames")
    args = parser.parse_args()

    if args.debug_yolop:
        logging.getLogger("src.percepcion.velocidad_dashboard").setLevel(logging.DEBUG)

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
    yolop = InferenciaYOLOP(
        imgsz=cfg["modelo"]["imgsz"],
        device=cfg["modelo"]["device"],
    )
    analizador_carriles = AnalizadorCarriles(usar_suavizado=True)
    pure_pursuit = PurePursuitVisual()
    controlador = construir_controlador(cfg)

    # Velocidad propia desde el HUD. Es mas fiable que flujo optico para detectar
    # 0 km/h; en ETS2, aplicar LT parado engrana reversa.
    cfg_vel_dash = cfg.get("velocidad_dashboard", {}) or {}
    estimador_velocidad = EstimadorVelocidadDashboard(
        max_kmh_norm=float(cfg_vel_dash.get("max_kmh_norm", 90.0)),
        retener_frames=int(cfg_vel_dash.get("retener_frames", 15)),
        prototypes_path=cfg_vel_dash.get("prototypes_path"),
    )
    velocidad_actual_norm = 0.0
    velocidad_actual_kmh: int | None = None
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
        logger.info("Cargando modelos YOLO...")
        tracker.cargar()
        yolop.cargar()

        # Warmup: pre-compila kernels CUDA para que el primer frame real sea rápido.
        # Se ejecutan 3 pasadas para asegurar que todos los caminos CUDA están compilados.
        logger.info("Warmup YOLO (compilando kernels CUDA — puede tardar 60s en arranque en frío)...")
        import numpy as _np
        _frame_dummy = _np.zeros((1080, 1920, 3), dtype=_np.uint8)
        for _ in range(3):
            tracker.rastrear(_frame_dummy)
            _, _da, _ll = yolop.procesar_frame(_frame_dummy)
            _pp = PurePursuitVisual()
            _pp.calcular_giro(_da, _ll)
        logger.info("Warmup completado — CUDA listo")

        fuente.iniciar()
        monitor.iniciar()

        primer_frame = True
        estado_anterior = fsm.estado_actual
        n_frame = 0
        seguimientos = []    # se actualiza cada YOLO_CADA frames
        # Cache del último resultado YOLO/FSM — se actualiza cada YOLO_CADA frames
        YOLO_CADA = 3        # YOLO cada 3 frames → ~10 FPS detección, ~30 FPS carril
        yolo_contador = 0
        resultado_cache = None
        # Cache de YOLOP — corre cada 2 frames para reducir latencia (~5 Hz → ~15 Hz)
        YOLOP_CADA = 2
        yolop_contador_carril = 0
        da_mask_cache: np.ndarray | None = None
        ll_mask_cache: np.ndarray | None = None

        # EMA de la desviación lateral del carril.
        # EMA rapida: con gamepad sin deadzone fisica, el piloto puede corregir
        # temprano; demasiada memoria retrasa la salida hasta estar cerca del muro.
        _ALPHA_EMA_CARRIL = 0.72
        desv_ema: float = 0.0
        frames_velocidad_invalida = 0

        # Alineación de cámara: presiona '8' cada N frames para que ETS2 vuelva
        # a la vista de cabina por defecto. El velocímetro está siempre en la esquina
        # inferior izquierda del HUD (no se mueve con la cámara), así que basta con
        # mantener la vista alineada para no perder visión de la carretera.
        _FRAMES_RESET_CAMARA = 20       # ~0.67 s a 30 fps

        from src.decision.estado import EstadoFSM
        _ESTADOS_CARRIL = (
            EstadoFSM.CONDUCIENDO_NORMAL,
            EstadoFSM.SIGUIENDO_VEHICULO,
            EstadoFSM.FRENANDO_PREVENTIVO,
            EstadoFSM.APROXIMANDO_ALTO,
            EstadoFSM.APROXIMANDO_SEMAFORO,
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

            # ── Detección de carril (cada YOLOP_CADA frames — inferencia ~100ms) ──
            yolop_contador_carril += 1
            if yolop_contador_carril >= YOLOP_CADA or da_mask_cache is None:
                yolop_contador_carril = 0
                _, da_mask_cache, ll_mask_cache = yolop.procesar_frame(cuadro.imagen)
            da_mask = da_mask_cache.copy()
            ll_mask = ll_mask_cache

            # DA: enmascarar zona superior (60%) — espejos virtuales ocupan hasta y≈55%.
            # LL: solo enmascarar hasta 30% — las marcas de carril visibles en cámara
            # aparecen en filas 25–55% del frame; el recorte al 60% las eliminaba todas.
            _fila_roi_da = int(da_mask.shape[0] * 0.60)
            da_mask[:_fila_roi_da, :] = 0
            _fila_roi_ll = int(ll_mask.shape[0] * 0.30)
            ll_mask_roi = ll_mask.copy()
            ll_mask_roi[:_fila_roi_ll, :] = 0
            ll_yolop_valida, pixeles_ll_yolop, pixeles_ll_izq, pixeles_ll_der = _ll_yolop_valida(ll_mask_roi)

            # Clasificación de carriles (ego / contrario / mismo sentido).
            # Disponible para futuras decisiones del FSM o el control;
            # de momento solo se visualiza con --debug-clasif-carriles.
            carriles_clasif = analizador_carriles.analizar(ll_mask, da_mask)

            # Pure Pursuit: ll_mask (nivel 1) → da_mask centroide (nivel 2) → decay (nivel 3).
            # No usamos el detector clásico de brillo como respaldo: su ROI no tiene en cuenta
            # el offset de cámara (_BIAS_CAM_PX=80), lo que produce un sesgo sistemático a la
            # derecha que hace que el camión se estrelle contra la barrera derecha.
            giro_pure_pursuit, carril_perdido = pure_pursuit.calcular_giro(da_mask, ll_mask_roi)
            fuente_carril = pure_pursuit.ultima_fuente_debug
            detalle_carril = ""
            comando_carril_directo: float | None = None

            # ── Velocidad propia desde HUD ──────────────────────────────────
            # Estimado antes de la EMA para poder usarlo en la decimación
            # con el valor del frame actual (no del anterior).
            lectura_velocidad = estimador_velocidad.estimar(cuadro.imagen)
            velocidad_actual_norm = lectura_velocidad.norm
            velocidad_actual_kmh = lectura_velocidad.kmh
            if lectura_velocidad.valido:
                frames_velocidad_invalida = 0
            else:
                frames_velocidad_invalida += 1

            # EMA de suavizado rapido: PurePursuit ya limita saltos grandes.
            desv_ema = (_ALPHA_EMA_CARRIL * giro_pure_pursuit + (1.0 - _ALPHA_EMA_CARRIL) * desv_ema)

            # Cuando el camión está casi parado el volante no produce corrección
            # lateral efectiva y la EMA acumula sesgo que dispara un sobreimpulso
            # al retomar la marcha. Por debajo de 3 km/h (lectura válida) se
            # decae la EMA a la mitad por frame para mantener la memoria pequeña.
            # Se exige lectura válida para evitar que lecturas falsas del velocímetro
            # (p.ej. marcas del cuadrante analógico) provoquen decimación espuria.
            if lectura_velocidad.valido and velocidad_actual_kmh is not None and velocidad_actual_kmh <= 2:
                desv_ema *= 0.5

            # Alineación periódica de cámara: presiona '8' cada _FRAMES_RESET_CAMARA.
            if _keyboard_ctrl is not None and n_frame > 0 and n_frame % _FRAMES_RESET_CAMARA == 0:
                _keyboard_ctrl.press('8')
                _keyboard_ctrl.release('8')

            # ── YOLO + FSM (cada YOLO_CADA frames — lenta ~100ms) ───────────
            yolo_contador += 1
            if yolo_contador >= YOLO_CADA or resultado_cache is None:
                yolo_contador = 0
                seguimientos = tracker.rastrear(cuadro.imagen)
                escena = contexto.analizar(seguimientos, cuadro.imagen, cuadro.timestamp)
                resultado_cache = fsm.decidir(escena)

            resultado = resultado_cache
            velocidad_feedback_norm = velocidad_feedback_para_control(
                velocidad_norm=velocidad_actual_norm,
                velocidad_kmh=velocidad_actual_kmh,
                lectura_valida=lectura_velocidad.valido,
                estado_fsm=resultado.estado_nuevo,
            )
            if isinstance(controlador, ControladorGamepadPID):
                controlador.actualizar_velocidad_actual(velocidad_feedback_norm)

            # ── Setpoint del FSM (mutable) + override de carril ──────────────
            setpoint = SetpointControl(
                velocidad_objetivo_norm=resultado.setpoint.velocidad_objetivo_norm,
                freno_objetivo=resultado.setpoint.freno_objetivo,
                desviacion_volante=resultado.setpoint.desviacion_volante,
            )

            # Override de carril: activo mientras el camion sigue avanzando.
            # Incluso al aproximarse a alto/semaforo debe mantenerse centrado;
            # solo los estados detenidos dejan el volante al FSM.
            # Zona muerta pequena: corrige deriva antes de acercarse a la linea.
            if (resultado.accion not in _ACCIONES_CON_GIRO
                    and resultado.estado_nuevo in _ESTADOS_CARRIL):
                if comando_carril_directo is not None:
                    setpoint.desviacion_volante = comando_carril_directo
                else:
                    setpoint.desviacion_volante = comando_direccion_por_carril(
                        desviacion_ema=float(desv_ema),
                        fuente_carril=fuente_carril,
                        velocidad_kmh=velocidad_actual_kmh if lectura_velocidad.valido else None,
                    )

            # Reducir velocidad cuando solo queda el fallback de carril, pero sin
            # convertir `da` en una frenada automática a ~8-10 km/h. En las
            # sesiones 1778021426 / 1778021893 esa combinación cortaba RT y metía
            # LT justo cuando el camión intentaba recuperar velocidad.
            factor_carril, freno_carril = limites_velocidad_por_carril(
                fuente_carril=fuente_carril,
                carril_perdido=carril_perdido,
                velocidad_actual_norm=velocidad_actual_norm,
                estado_con_carril=resultado.estado_nuevo in _ESTADOS_CARRIL,
            )
            setpoint.velocidad_objetivo_norm *= factor_carril
            setpoint.freno_objetivo = max(setpoint.freno_objetivo, freno_carril)

            if resultado.estado_nuevo in _ESTADOS_CARRIL:
                curva = max(abs(giro_pure_pursuit), abs(desv_ema), pure_pursuit.ultima_curvatura_debug)
                _VEL_FRENO_CURVA = 0.17  # ~15 km/h a 90 km/h max
                setpoint.velocidad_objetivo_norm *= 0.80
                if curva > 0.06:
                    escala_curva = float(np.interp(curva, [0.06, 0.45], [0.85, 0.35]))
                    setpoint.velocidad_objetivo_norm *= escala_curva
                if curva > 0.12:
                    # Frenar en curva pronunciada aunque la velocidad no se pueda leer:
                    # sin lectura el PID cree que el camión está parado y aceleraría
                    # al máximo, así que se fuerza freno preventivo.
                    if velocidad_actual_norm >= _VEL_FRENO_CURVA or not lectura_velocidad.valido:
                        freno_curva = 0.06 if curva < 0.25 else 0.10
                        setpoint.freno_objetivo = max(setpoint.freno_objetivo, freno_curva)
                cap_vel_desc, freno_vel_desc = limites_por_velocidad_desconocida(
                    frames_sin_lectura=frames_velocidad_invalida,
                    fuente_carril=fuente_carril,
                    curva=curva,
                    estado_con_carril=True,
                )
                if cap_vel_desc is not None:
                    setpoint.velocidad_objetivo_norm = min(setpoint.velocidad_objetivo_norm, cap_vel_desc)
                setpoint.freno_objetivo = max(setpoint.freno_objetivo, freno_vel_desc)

            if args.debug_carril and n_frame % 30 == 0:
                logger.info(
                    "CARRIL (%s%s) desv_pp=%+.3f ema=%+.3f curv=%.2f stick_obj=%+.2f vel=%.2f kmh=%s ll_px=%d/%d/%d",
                    fuente_carril,
                    detalle_carril,
                    giro_pure_pursuit, desv_ema,
                    pure_pursuit.ultima_curvatura_debug,
                    setpoint.desviacion_volante, velocidad_actual_norm,
                    "-" if velocidad_actual_kmh is None else str(velocidad_actual_kmh),
                    pixeles_ll_yolop, pixeles_ll_izq, pixeles_ll_der,
                )

            if args.debug_carril_img and n_frame % 60 == 0:
                import cv2 as _cv2
                from pathlib import Path as _Path
                # Crear imagen de debug combinando mascara verde sobre el frame
                dbg = cuadro.imagen.copy()
                mask_color = np.zeros_like(dbg)
                mask_color[da_mask > 0] = (0, 255, 0) # Verde para drivable area
                mask_color[ll_mask > 0] = (0, 0, 255) # Rojo para lane lines
                dbg = _cv2.addWeighted(dbg, 0.7, mask_color, 0.3, 0)

                # Dibujar look-ahead point
                punto = pure_pursuit.ultimo_punto_debug
                if punto:
                    _cv2.circle(dbg, punto, 10, (255, 255, 0), -1)

                ruta_dbg = _Path(cfg["registro"]["ruta_base"]) / f"debug_yolop_{n_frame:06d}.jpg"
                _cv2.imwrite(str(ruta_dbg), dbg)
                logger.info("Debug imagen YOLOP guardada: %s", ruta_dbg)

            if args.debug_yolop and n_frame % 60 == 0:
                import cv2 as _cv2
                from pathlib import Path as _Path
                _H = 486  # altura del panel; ancho proporcional a 16:9

                # Panel izquierdo: frame original capturado (lo que el piloto ve)
                orig = _cv2.resize(cuadro.imagen, (_H * 16 // 9, _H))
                _cv2.putText(orig, "CAPTURA ORIGINAL", (8, 22),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Panel central: entrada real del modelo (CLAHE + sharpened + letterbox)
                entrada = yolop.ultima_imagen_debug
                if entrada is not None:
                    # Escalar manteniendo el cuadrado letterbox visible
                    entrada = _cv2.resize(entrada, (_H, _H))
                    _cv2.putText(entrada, "ENTRADA MODELO (CLAHE+sharp+LB)", (8, 22),
                                 _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                else:
                    entrada = np.zeros((_H, _H, 3), dtype=np.uint8)

                # Panel derecho: máscaras + look-ahead sobre frame original
                mascaras = cuadro.imagen.copy()
                mc = np.zeros_like(mascaras)
                mc[da_mask > 0] = (0, 200, 0)   # verde = drivable area
                mc[ll_mask > 0] = (0, 80, 255)  # naranja-rojo = lane lines
                mascaras = _cv2.addWeighted(mascaras, 0.65, mc, 0.35, 0)
                punto = pure_pursuit.ultimo_punto_debug
                if punto:
                    _cv2.circle(mascaras, punto, 12, (0, 255, 255), -1)
                    _cv2.circle(mascaras, punto, 12, (0, 0, 0), 2)
                # Anotar error de carril actual
                _cv2.putText(mascaras,
                             f"src={fuente_carril} ll={pixeles_ll_yolop}/{pixeles_ll_izq}/{pixeles_ll_der} err={giro_pure_pursuit:+.3f} ema={desv_ema:+.3f} cmd={setpoint.desviacion_volante:+.3f} kmh={velocidad_actual_kmh if velocidad_actual_kmh is not None else '-'} curv={pure_pursuit.ultima_curvatura_debug:.2f}{detalle_carril}",
                             (8, 22), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                _cv2.putText(mascaras, "MASCARAS + LOOK-AHEAD", (8, 48),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                mascaras = _cv2.resize(mascaras, (_H * 16 // 9, _H))

                # Separadores verticales de 4px
                sep = np.full((_H, 4, 3), 60, dtype=np.uint8)
                compuesto = np.hstack([orig, sep, entrada, sep, mascaras])

                ruta_cmp = _Path(cfg["registro"]["ruta_base"]) / f"debug_modelo_{n_frame:06d}.jpg"
                _cv2.imwrite(str(ruta_cmp), compuesto, [_cv2.IMWRITE_JPEG_QUALITY, 90])
                logger.info("Debug modelo guardado: %s", ruta_cmp)

                # ROI del velocímetro ampliada 4× para calibración visual
                roi_vel = estimador_velocidad.roi_debug(cuadro.imagen)
                roi_vel_big = _cv2.resize(roi_vel, (roi_vel.shape[1] * 4, roi_vel.shape[0] * 4),
                                          interpolation=_cv2.INTER_NEAREST)
                _cv2.putText(roi_vel_big, f"VEL ROI kmh={velocidad_actual_kmh if velocidad_actual_kmh is not None else '-'}",
                             (4, 18), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                ruta_roi = _Path(cfg["registro"]["ruta_base"]) / f"debug_vel_roi_{n_frame:06d}.jpg"
                _cv2.imwrite(str(ruta_roi), roi_vel_big, [_cv2.IMWRITE_JPEG_QUALITY, 95])
                dump_comp_dir = cfg_vel_dash.get("dump_componentes_dir")
                if dump_comp_dir:
                    estimador_velocidad.guardar_componentes_debug(cuadro.imagen, dump_comp_dir, n_frame)

            if args.debug_clasif_carriles and n_frame % 60 == 0:
                import cv2 as _cv2
                from pathlib import Path as _Path
                clasif_img = superponer_carriles(
                    cuadro.imagen,
                    carriles_clasif,
                    area_mask=da_mask,
                    fps=cuadro.fps_instantaneo,
                    frame_idx=n_frame,
                )
                ruta_clasif = _Path(cfg["registro"]["ruta_base"]) / f"debug_carriles_{n_frame:06d}.jpg"
                _cv2.imwrite(str(ruta_clasif), clasif_img, [_cv2.IMWRITE_JPEG_QUALITY, 90])
                logger.info(
                    "Debug clasificación carriles guardado: %s | estado=%s offset=%s",
                    ruta_clasif, carriles_clasif.estado,
                    f"{carriles_clasif.offset_px:+.0f}px" if carriles_clasif.offset_px is not None else "-",
                )

            if isinstance(controlador, ControladorGamepadPID):
                controlador.aplicar(setpoint)
                if args.debug_carril and n_frame % 30 == 0:
                    rt_aplicado, lt_aplicado, stick_aplicado = controlador.ultimo_comando_aplicado
                    logger.info(
                        "GAMEPAD aplicado rt=%d lt=%d stick=%+.2f",
                        rt_aplicado, lt_aplicado, stick_aplicado,
                    )
                    log.evento("carril_control", {
                        "frame": n_frame,
                        "fuente": fuente_carril,
                        "detalle": detalle_carril.strip(),
                        "err": round(float(giro_pure_pursuit), 4),
                        "ema": round(float(desv_ema), 4),
                        "cmd": round(float(setpoint.desviacion_volante), 4),
                        "stick": round(float(stick_aplicado), 4),
                        "kmh": velocidad_actual_kmh,
                        "rt": int(rt_aplicado),
                        "lt": int(lt_aplicado),
                        "ll_total": int(pixeles_ll_yolop),
                        "ll_izq": int(pixeles_ll_izq),
                        "ll_der": int(pixeles_ll_der),
                        "perdido": bool(carril_perdido),
                    })
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
