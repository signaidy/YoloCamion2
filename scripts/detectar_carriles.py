"""Detección y clasificación de carriles con YOLOP (Mac-friendly).

Usa el analizador del proyecto (src/percepcion/analisis_carriles.py) para que
los resultados aquí coincidan con lo que verá el piloto en Windows.

Uso:
    python scripts/detectar_carriles.py                 # bus.jpg
    python scripts/detectar_carriles.py imagen.jpg
    python scripts/detectar_carriles.py video.mp4
    python scripts/detectar_carriles.py video.mp4 --salida resultado.mp4
"""
import argparse
import importlib.util
import sys
import time
import warnings
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

# Importar el módulo de análisis sin pasar por src/percepcion/__init__.py
# (que en Mac/3.9 trae módulos con sintaxis 3.10+ que no compilan)
_modulo_path = ROOT / "src" / "percepcion" / "analisis_carriles.py"
_spec = importlib.util.spec_from_file_location("analisis_carriles", _modulo_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["analisis_carriles"] = _mod  # necesario para dataclasses
_spec.loader.exec_module(_mod)
AnalizadorCarriles = _mod.AnalizadorCarriles
superponer_carriles = _mod.superponer_carriles

IMGSZ = 640


# ────────────────────────────────────────────────────────────
# YOLOP — carga e inferencia (autocontenida, no toca src/)
# ────────────────────────────────────────────────────────────
def cargar_modelo(device: torch.device):
    print("[YOLOP] Cargando modelo (puede tardar la primera vez, ~500 MB)...")
    warnings.filterwarnings("ignore")
    modelo = torch.hub.load("hustvl/yolop", "yolop", pretrained=True, trust_repo=True)
    modelo.to(device).eval()
    print(f"[YOLOP] Modelo listo en {device}")
    return modelo


def _letterbox(img: np.ndarray, size: int = IMGSZ):
    h, w = img.shape[:2]
    r = size / max(h, w)
    new_w, new_h = round(w * r), round(h * r)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_l = (size - new_w) // 2
    pad_t = (size - new_h) // 2
    pad_r = size - new_w - pad_l
    pad_b = size - new_h - pad_t
    img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, pad_l, pad_t, new_w, new_h


def _realzar(img: np.ndarray, clahe) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _preprocesar(img: np.ndarray, device: torch.device, clahe):
    shape_orig = img.shape[:2]
    img = _realzar(img, clahe)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    img = cv2.addWeighted(img, 1.25, blur, -0.25, 0)
    img, pad_l, pad_t, new_w, new_h = _letterbox(img, IMGSZ)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    t = torch.from_numpy(np.ascontiguousarray(img)).float().to(device) / 255.0
    t = t.unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    t = (t - mean) / std
    return t, pad_l, pad_t, new_w, new_h, shape_orig


def _mascara(seg_out, pad_l, pad_t, new_w, new_h, shape_orig) -> np.ndarray:
    pred = seg_out[:, 1, :, :] > seg_out[:, 0, :, :]
    m = pred.byte().cpu().numpy()[0]
    m = m[pad_t:pad_t + new_h, pad_l:pad_l + new_w]
    return cv2.resize(m, (shape_orig[1], shape_orig[0]), interpolation=cv2.INTER_NEAREST)


def inferir(modelo, img: np.ndarray, device: torch.device, clahe) -> Tuple[np.ndarray, np.ndarray]:
    t, pad_l, pad_t, new_w, new_h, shape_orig = _preprocesar(img, device, clahe)
    with torch.no_grad():
        _, area_out, lineas_out = modelo(t)
    area = _mascara(area_out, pad_l, pad_t, new_w, new_h, shape_orig)
    lineas = _mascara(lineas_out, pad_l, pad_t, new_w, new_h, shape_orig)
    return area, lineas


# ────────────────────────────────────────────────────────────
# Drivers
# ────────────────────────────────────────────────────────────
def procesar_imagen(modelo, device, clahe, ruta_in: Path, ruta_out: Path):
    img = cv2.imread(str(ruta_in))
    if img is None:
        sys.exit(f"No pude leer la imagen: {ruta_in}")
    analizador = AnalizadorCarriles(usar_suavizado=False)
    t0 = time.perf_counter()
    area, lineas_mask = inferir(modelo, img, device, clahe)
    carriles = analizador.analizar(lineas_mask, area)
    fps = 1.0 / max(1e-6, time.perf_counter() - t0)
    salida = superponer_carriles(img, carriles, area_mask=area, fps=fps)
    cv2.imwrite(str(ruta_out), salida)
    print(f"Guardado: {ruta_out}  | estado={carriles.estado}")


def procesar_video(modelo, device, clahe, ruta_in: Path, ruta_out: Path):
    cap = cv2.VideoCapture(str(ruta_in))
    if not cap.isOpened():
        sys.exit(f"No pude abrir el video: {ruta_in}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(ruta_out), fourcc, fps_in, (w, h))
    analizador = AnalizadorCarriles(usar_suavizado=True)
    i = 0
    t_inicio = time.perf_counter()
    fps_actual = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_frame = time.perf_counter()
        area, lineas_mask = inferir(modelo, frame, device, clahe)
        carriles = analizador.analizar(lineas_mask, area)
        salida = superponer_carriles(frame, carriles, area_mask=area,
                                      fps=fps_actual, frame_idx=i)
        writer.write(salida)
        dt = time.perf_counter() - t_frame
        fps_actual = (0.9 * fps_actual + 0.1 * (1.0 / max(1e-6, dt))) if i else 1.0 / max(1e-6, dt)
        i += 1
        if i % 10 == 0:
            print(f"  frame {i}/{total}  fps≈{fps_actual:.1f}")
    cap.release()
    writer.release()
    dur = time.perf_counter() - t_inicio
    print(f"Guardado: {ruta_out}  | {i} frames en {dur:.1f}s ({i/max(1e-6,dur):.1f} fps prom)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("entrada", nargs="?", default=str(ROOT / "bus.jpg"))
    p.add_argument("--salida")
    args = p.parse_args()

    ruta_in = Path(args.entrada)
    if not ruta_in.exists():
        sys.exit(f"No existe: {ruta_in}")

    es_video = ruta_in.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    salida_dir = ROOT / "datos" / "evidencia"
    salida_dir.mkdir(parents=True, exist_ok=True)
    ruta_out = Path(args.salida) if args.salida else \
        salida_dir / f"carriles_{ruta_in.stem}{'.mp4' if es_video else '.jpg'}"

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    modelo = cargar_modelo(device)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    if es_video:
        procesar_video(modelo, device, clahe, ruta_in, ruta_out)
    else:
        procesar_imagen(modelo, device, clahe, ruta_in, ruta_out)


if __name__ == "__main__":
    main()
