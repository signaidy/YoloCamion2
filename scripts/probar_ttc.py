"""Smoke test visual del pipeline TTC + flujo optico sobre un video grabado.

NO requiere ETS2 abierto. Toma un video local (default: el de evidencia),
corre YOLO+tracker+EstimadorFisicaVisual, anota cada bbox con id/area/TTC
y dibuja el campo de flujo optico restringido al ROI frontal. Genera
salida MP4 anotada para revision manual.

Uso:
  python scripts/probar_ttc.py
  python scripts/probar_ttc.py --video datos/videos/ets2_volvo_fh16.f299.mp4
  python scripts/probar_ttc.py --max-frames 300 --salida datos/evidencia/ttc.mp4
"""
import argparse
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fuente.video import FuenteVideo
from src.percepcion.contexto import AnalizadorContexto, cargar_rois_yaml
from src.percepcion.fisica import EstimadorFisicaVisual
from src.percepcion.flujo_optico import (
    EstimadorFlujoOptico,
    EstimadorFlujoOpticoLK,
    promediar_flujo_en_caja,
)
from src.percepcion.tracker import Tracker
from src.tipos import Region

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("probar_ttc")


def _color_para_ttc(ttc: float) -> tuple[int, int, int]:
    """Verde si TTC > 4s, amarillo entre 1.5 y 4s, rojo si < 1.5s."""
    if math.isinf(ttc) or ttc > 4.0:
        return (0, 200, 0)
    if ttc > 1.5:
        return (0, 200, 200)
    return (0, 0, 220)


def _dibujar_flujo(
    frame: np.ndarray,
    flujo: np.ndarray,
    paso: int = 32,
    escala: float = 0.25,
) -> None:
    """Dibuja muestreo del campo de flujo como flechas amarillas."""
    h, w = flujo.shape[:2]
    for y in range(paso // 2, h, paso):
        for x in range(paso // 2, w, paso):
            u, v = flujo[y, x]
            if abs(u) < 1.0 and abs(v) < 1.0:
                continue
            x2 = int(x + u * escala)
            y2 = int(y + v * escala)
            cv2.arrowedLine(frame, (x, y), (x2, y2), (0, 220, 220),
                            1, tipLength=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="datos/videos/ets2_volvo_fh16.f299.mp4")
    parser.add_argument("--rois", default="config/regiones_interes.yaml")
    parser.add_argument("--modelo", default="datos/modelos/yolo26n.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--salida", default="datos/evidencia/ttc_smoke.mp4")
    parser.add_argument("--mostrar", action="store_true",
                        help="Mostrar ventana en vivo (requiere display).")
    parser.add_argument("--flujo", choices=["lk", "denso"], default="lk",
                        help="Backend de flujo optico (default lk; denso es ~40x mas lento).")
    args = parser.parse_args()

    rois = cargar_rois_yaml(args.rois) if Path(args.rois).exists() else None
    if rois is None:
        log.warning("ROI no encontrado en %s -- usando defaults", args.rois)

    fuente = FuenteVideo(args.video)
    fuente.iniciar()

    tracker = Tracker(ruta_modelo=args.modelo, device=args.device)
    log.info("Cargando YOLO26 desde %s ...", args.modelo)
    tracker.cargar()

    estimador_fisica = EstimadorFisicaVisual()
    contexto = AnalizadorContexto(rois=rois, estimador_fisica=estimador_fisica)

    # ROI de flujo = bounding box que cubre frente_cercano + frente_lejano
    if rois is None:
        # Defaults razonables para 1920x1080
        roi_flujo = (282, 231, 1661, 1080)
    else:
        fc = rois[Region.FRENTE_CERCANO]
        fl = rois[Region.FRENTE_LEJANO]
        roi_flujo = (
            min(fc[0], fl[0]), min(fc[1], fl[1]),
            max(fc[2], fl[2]), max(fc[3], fl[3]),
        )
    log.info("ROI flujo optico: %s (backend=%s)", roi_flujo, args.flujo)
    if args.flujo == "lk":
        estimador_flujo = EstimadorFlujoOpticoLK(roi=roi_flujo)
    else:
        estimador_flujo = EstimadorFlujoOptico(roi=roi_flujo)

    salida_path = Path(args.salida)
    salida_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None

    n_frames = 0
    n_ttc_validos = 0
    suma_latencia_ms = 0.0
    t0_global = time.perf_counter()

    try:
        while fuente.esta_activa and n_frames < args.max_frames:
            cuadro = fuente.siguiente()
            if cuadro is None:
                break

            t0 = time.perf_counter()
            seguimientos = tracker.rastrear(cuadro.imagen)
            flujo = estimador_flujo.calcular(cuadro.imagen, cuadro.timestamp)
            estado = contexto.analizar(seguimientos, cuadro.imagen,
                                       timestamp=cuadro.timestamp)
            latencia_ms = (time.perf_counter() - t0) * 1000
            suma_latencia_ms += latencia_ms

            # ── Anotar frame ────────────────────────────────────────────────
            anotado = cuadro.imagen.copy()
            _dibujar_flujo(anotado, flujo)

            for seg in seguimientos:
                x1, y1, x2, y2 = seg.caja
                fis = seg.fisica
                ttc = fis.ttc_segundos if fis else math.inf
                color = _color_para_ttc(ttc)
                cv2.rectangle(anotado, (x1, y1), (x2, y2), color, 2)

                ttc_str = "inf" if math.isinf(ttc) else f"{ttc:.1f}s"
                u, v = (0.0, 0.0)
                if fis is not None:
                    u, v = promediar_flujo_en_caja(flujo, seg.caja)
                etiqueta = (f"id={seg.id_seguimiento} {seg.clase.value} "
                            f"a={seg.area} ttc={ttc_str} "
                            f"flow=({u:+.0f},{v:+.0f})")
                cv2.putText(anotado, etiqueta, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                            cv2.LINE_AA)

            # ROI frontal en azul
            cv2.rectangle(anotado, (roi_flujo[0], roi_flujo[1]),
                          (roi_flujo[2], roi_flujo[3]), (255, 100, 0), 1)

            # HUD
            ttc_min_str = ("inf" if math.isinf(estado.ttc_minimo_frente_s)
                           else f"{estado.ttc_minimo_frente_s:.2f}s")
            hud = (f"frame={n_frames} lat={latencia_ms:.0f}ms "
                   f"ttc_min={ttc_min_str} critico={estado.vehiculo_critico_id} "
                   f"vehs={estado.vehiculos_totales}")
            cv2.putText(anotado, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if not math.isinf(estado.ttc_minimo_frente_s):
                n_ttc_validos += 1

            # ── Guardar ─────────────────────────────────────────────────────
            if writer is None:
                h, w = anotado.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(salida_path), fourcc, 30.0, (w, h))
                log.info("Escribiendo %dx%d a %s", w, h, salida_path)
            writer.write(anotado)

            if args.mostrar:
                cv2.imshow("probar_ttc", anotado)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            n_frames += 1

    finally:
        if writer is not None:
            writer.release()
        fuente.cerrar()
        if args.mostrar:
            cv2.destroyAllWindows()

    total_s = time.perf_counter() - t0_global
    fps = n_frames / total_s if total_s > 0 else 0.0
    lat_med = suma_latencia_ms / n_frames if n_frames else 0.0
    log.info("=== Smoke TTC terminado ===")
    log.info("frames procesados: %d", n_frames)
    log.info("FPS efectivo:      %.1f", fps)
    log.info("Latencia media:    %.1f ms/frame", lat_med)
    log.info("Frames con TTC<inf en frente: %d (%.1f%%)",
             n_ttc_validos, 100 * n_ttc_validos / max(n_frames, 1))
    log.info("Salida:            %s", salida_path)


if __name__ == "__main__":
    main()
