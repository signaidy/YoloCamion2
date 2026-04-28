"""Benchmark de FPS end-to-end del pipeline pure-vision.

Compara cuatro configuraciones para cuantificar el costo de cada capa:

  A baseline       YOLO solo
  B + TTC          YOLO + tracker + EstimadorFisicaVisual
  C + flujo denso  YOLO + tracker + TTC + Farneback denso (ROI frontal)
  D + flujo LK     YOLO + tracker + TTC + Lucas-Kanade disperso (ROI frontal)

Criterio F0: >= 30 FPS sostenidos a 1920x1080.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.percepcion.contexto import AnalizadorContexto, cargar_rois_yaml
from src.percepcion.fisica import EstimadorFisicaVisual
from src.percepcion.flujo_optico import EstimadorFlujoOptico, EstimadorFlujoOpticoLK
from src.percepcion.tracker import Tracker
from src.tipos import Region


# ── Helpers de fuente ────────────────────────────────────────────────────────


def _crear_fuente(ruta_video: str | None):
    if ruta_video and Path(ruta_video).exists():
        cap = cv2.VideoCapture(ruta_video)
        print(f"Benchmark sobre video: {ruta_video}")
        def fuente():
            ok, f = cap.read()
            return f if ok else None
        return fuente, cap
    print("Benchmark sobre frames sinteticos 1920x1080")
    base = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    def fuente():
        # Pequeno desplazamiento para que el flujo no sea cero
        return np.roll(base, np.random.randint(-3, 4), axis=1)
    return fuente, None


def _resumen(latencias_ms: list[float], total_s: float) -> dict:
    arr = np.array(latencias_ms)
    fps = len(arr) / total_s if total_s > 0 else 0.0
    return {
        "n": len(arr),
        "fps": round(fps, 2),
        "lat_ms_med": round(float(arr.mean()), 1),
        "lat_ms_p95": round(float(np.percentile(arr, 95)), 1),
        "lat_ms_max": round(float(arr.max()), 1),
    }


def _roi_frontal(rois) -> tuple[int, int, int, int]:
    if rois is None:
        return (282, 231, 1661, 691)
    fc = rois[Region.FRENTE_CERCANO]
    fl = rois[Region.FRENTE_LEJANO]
    return (min(fc[0], fl[0]), min(fc[1], fl[1]),
            max(fc[2], fl[2]), max(fc[3], fl[3]))


# ── Modos de benchmark ──────────────────────────────────────────────────────


def benchmark_yolo_solo(fuente, n_frames: int, modelo: str, device: str) -> dict:
    from ultralytics import YOLO
    m = YOLO(modelo)
    for _ in range(5):
        m(fuente(), device=device, verbose=False, imgsz=640)
    lats = []
    t0 = time.perf_counter()
    for _ in range(n_frames):
        f = fuente()
        if f is None:
            break
        t = time.perf_counter()
        m(f, device=device, verbose=False, imgsz=640, conf=0.35)
        lats.append((time.perf_counter() - t) * 1000)
    return _resumen(lats, time.perf_counter() - t0)


def benchmark_yolo_ttc(fuente, n_frames: int, modelo: str, device: str,
                        rois) -> dict:
    tracker = Tracker(ruta_modelo=modelo, device=device)
    tracker.cargar()
    contexto = AnalizadorContexto(rois=rois,
                                   estimador_fisica=EstimadorFisicaVisual())
    for _ in range(5):
        tracker.rastrear(fuente())
    lats = []
    t0 = time.perf_counter()
    for i in range(n_frames):
        f = fuente()
        if f is None:
            break
        t = time.perf_counter()
        seg = tracker.rastrear(f)
        contexto.analizar(seg, f, timestamp=t)
        lats.append((time.perf_counter() - t) * 1000)
    return _resumen(lats, time.perf_counter() - t0)


def benchmark_completo(fuente, n_frames: int, modelo: str, device: str,
                        rois, modo_flujo: str = "denso") -> dict:
    tracker = Tracker(ruta_modelo=modelo, device=device)
    tracker.cargar()
    contexto = AnalizadorContexto(rois=rois,
                                   estimador_fisica=EstimadorFisicaVisual())
    roi_f = _roi_frontal(rois)
    if modo_flujo == "lk":
        flujo = EstimadorFlujoOpticoLK(roi=roi_f)
    else:
        flujo = EstimadorFlujoOptico(roi=roi_f)
    for _ in range(5):
        f0 = fuente()
        tracker.rastrear(f0)
        flujo.calcular(f0, time.perf_counter())
    lats = []
    t0 = time.perf_counter()
    for i in range(n_frames):
        f = fuente()
        if f is None:
            break
        t = time.perf_counter()
        seg = tracker.rastrear(f)
        flujo.calcular(f, t)
        contexto.analizar(seg, f, timestamp=t)
        lats.append((time.perf_counter() - t) * 1000)
    return _resumen(lats, time.perf_counter() - t0)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="datos/modelos/yolo26n.pt")
    parser.add_argument("--video", default="datos/videos/ets2_volvo_fh16.f299.mp4")
    parser.add_argument("--rois", default="config/regiones_interes.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--modo", default="todos",
                        choices=["todos", "yolo", "ttc", "denso", "lk"])
    args = parser.parse_args()

    rois = cargar_rois_yaml(args.rois) if Path(args.rois).exists() else None

    resultados: dict[str, dict] = {}

    if args.modo in ("todos", "yolo"):
        fuente, cap = _crear_fuente(args.video)
        try:
            print("\n[A] YOLO solo ...")
            resultados["A_yolo"] = benchmark_yolo_solo(
                fuente, args.frames, args.modelo, args.device)
        finally:
            if cap: cap.release()

    if args.modo in ("todos", "ttc"):
        fuente, cap = _crear_fuente(args.video)
        try:
            print("[B] YOLO + tracker + TTC ...")
            resultados["B_yolo_ttc"] = benchmark_yolo_ttc(
                fuente, args.frames, args.modelo, args.device, rois)
        finally:
            if cap: cap.release()

    if args.modo in ("todos", "denso"):
        fuente, cap = _crear_fuente(args.video)
        try:
            print("[C] YOLO + TTC + flujo denso (Farneback) ...")
            resultados["C_completo_denso"] = benchmark_completo(
                fuente, args.frames, args.modelo, args.device, rois, "denso")
        finally:
            if cap: cap.release()

    if args.modo in ("todos", "lk"):
        fuente, cap = _crear_fuente(args.video)
        try:
            print("[D] YOLO + TTC + flujo LK disperso ...")
            resultados["D_completo_lk"] = benchmark_completo(
                fuente, args.frames, args.modelo, args.device, rois, "lk")
        finally:
            if cap: cap.release()

    print("\n=== RESULTADOS ===")
    print(f"  {'config':<20} {'fps':>7} {'lat_med':>9} {'lat_p95':>9} {'lat_max':>9}")
    for nombre, r in resultados.items():
        print(f"  {nombre:<20} {r['fps']:>7.2f} {r['lat_ms_med']:>9.1f} "
              f"{r['lat_ms_p95']:>9.1f} {r['lat_ms_max']:>9.1f}")

    print("\n=== CRITERIO F0 (>= 30 FPS) ===")
    for nombre, r in resultados.items():
        marca = "OK" if r["fps"] >= 30 else "FAIL"
        print(f"  [{marca}] {nombre}: {r['fps']} FPS")


if __name__ == "__main__":
    main()
