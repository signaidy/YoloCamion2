"""Benchmark de FPS end-to-end: captura simulada + YOLO.

Criterio F0: >= 30 FPS sostenidos a 1920x1080.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def benchmark(ruta_modelo: str, ruta_video: str | None, n_frames: int = 300) -> dict:
    modelo = YOLO(ruta_modelo)

    cap = None
    if ruta_video and Path(ruta_video).exists():
        cap = cv2.VideoCapture(ruta_video)
        print(f"Benchmark sobre video: {ruta_video}")
        def fuente():
            ok, f = cap.read()
            return f
    else:
        print("Benchmark sobre frames sintéticos 1920x1080")
        frame_sintetico = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        def fuente():
            return frame_sintetico

    print("Warmup (10 frames)...")
    for _ in range(10):
        modelo(fuente(), device="cuda", verbose=False, imgsz=640)

    print(f"Benchmark ({n_frames} frames)...")
    latencias_ms = []
    inicio_total = time.perf_counter()
    for _ in range(n_frames):
        frame = fuente()
        if frame is None:
            break
        t0 = time.perf_counter()
        modelo(frame, device="cuda", verbose=False, imgsz=640, conf=0.35)
        latencias_ms.append((time.perf_counter() - t0) * 1000)
    fin_total = time.perf_counter()

    if cap:
        cap.release()

    latencias = np.array(latencias_ms)
    fps = len(latencias) / (fin_total - inicio_total)

    return {
        "n_frames": len(latencias),
        "tiempo_total_s": round(fin_total - inicio_total, 2),
        "fps_promedio": round(fps, 2),
        "latencia_ms_min": round(float(latencias.min()), 2),
        "latencia_ms_media": round(float(latencias.mean()), 2),
        "latencia_ms_p95": round(float(np.percentile(latencias, 95)), 2),
        "latencia_ms_max": round(float(latencias.max()), 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="datos/modelos/yolo11n.pt")
    parser.add_argument("--video", default=None)
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()

    resultado = benchmark(args.modelo, args.video, args.frames)

    print("\n=== RESULTADOS ===")
    for k, v in resultado.items():
        print(f"  {k}: {v}")

    print("\n=== CRITERIO F0 ===")
    if resultado["fps_promedio"] >= 30:
        print(f"  APROBADO: {resultado['fps_promedio']} FPS >= 30 FPS objetivo")
    else:
        print(f"  RECHAZADO: {resultado['fps_promedio']} FPS < 30 FPS objetivo")
        print("  Acción: probar con imgsz=480, o bajar a modelo nano")
