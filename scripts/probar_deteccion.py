"""Prueba visual del detector + tracker sobre el video del Volvo FH16.

Muestra en tiempo real:
  - Cajas de detección YOLO con clase y confianza
  - ROI calibradas como overlay semitransparente
  - Estado del FSM y acción decidida
  - FPS y latencia

Uso:
  python scripts/probar_deteccion.py
  python scripts/probar_deteccion.py --frame-inicio 6000   # empezar en minuto ~1:40
  python scripts/probar_deteccion.py --velocidad 2         # reproducir a 2x

Controles:
  ESPACIO  → pausar / reanudar
  → / D    → avanzar 5 segundos (mientras pausado)
  ← / A    → retroceder 5 segundos (mientras pausado)
  Q / ESC  → salir
  G        → guardar frame actual como PNG
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.decision import FSMDecision
from src.percepcion import AnalizadorContexto, Tracker
from src.percepcion.contexto import cargar_rois_yaml
from src.tipos import Region

FONT = cv2.FONT_HERSHEY_SIMPLEX

_COLOR_CLASE = {
    "vehiculo":    (0, 165, 255),
    "motocicleta": (0, 165, 255),
    "peaton":      (0, 0, 255),
    "semaforo":    (255, 255, 0),
    "senal_alto":  (0, 255, 255),
    "desconocido": (128, 128, 128),
}

_COLOR_ROI = {
    Region.FRENTE_CERCANO: (0, 220, 0),
    Region.FRENTE_LEJANO:  (0, 200, 180),
    Region.ESPEJO_IZQ:     (255, 120, 0),
    Region.ESPEJO_DER:     (255, 0, 120),
    Region.LATERAL_IZQ:    (180, 0, 255),
    Region.LATERAL_DER:    (100, 0, 255),
}

_NOMBRE_ROI = {
    Region.FRENTE_CERCANO: "FC",
    Region.FRENTE_LEJANO:  "FL",
    Region.ESPEJO_IZQ:     "EI",
    Region.ESPEJO_DER:     "ED",
    Region.LATERAL_IZQ:    "LI",
    Region.LATERAL_DER:    "LD",
}

_COLOR_ACCION = {
    "alto_total":    (0, 0, 255),
    "frenar_fuerte": (0, 60, 220),
    "frenar_suave":  (0, 140, 255),
    "mantener":      (200, 200, 200),
    "acelerar":      (0, 220, 0),
    "rebasar_izq":   (220, 0, 255),
    "rebasar_der":   (180, 0, 200),
    "esperar":       (160, 160, 0),
}


def dibujar_rois(canvas: np.ndarray, rois: dict) -> np.ndarray:
    overlay = canvas.copy()
    for region, coords in rois.items():
        x1, y1, x2, y2 = coords
        color = _COLOR_ROI.get(region, (128, 128, 128))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        cv2.putText(overlay, _NOMBRE_ROI.get(region, "?"),
                    (x1 + 4, y1 + 16), FONT, 0.45, color, 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0)


def dibujar_detecciones(canvas: np.ndarray, seguimientos) -> np.ndarray:
    for seg in seguimientos:
        x1, y1, x2, y2 = seg.caja
        color = _COLOR_CLASE.get(seg.clase.value, (128, 128, 128))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        texto = f"{seg.clase.value} #{seg.id_seguimiento} {seg.confianza:.0%}"
        cv2.putText(canvas, texto, (x1, max(y1 - 5, 12)),
                    FONT, 0.48, color, 1, cv2.LINE_AA)
    return canvas


def dibujar_hud(canvas: np.ndarray, resultado, escena, fps: float, latencia_ms: float,
                n_frame: int, total: int, pausado: bool) -> np.ndarray:
    h, w = canvas.shape[:2]

    # Barra superior
    barra = np.zeros((65, w, 3), dtype=np.uint8)
    accion = resultado.accion.value
    color_acc = _COLOR_ACCION.get(accion, (200, 200, 200))

    estado_str = f"{resultado.estado_nuevo.value}  [R{resultado.regla}]"
    cv2.putText(barra, f"Accion: {accion.upper()}", (10, 24),
                FONT, 0.75, color_acc, 2, cv2.LINE_AA)
    cv2.putText(barra, estado_str, (10, 50),
                FONT, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    fps_str = f"FPS: {fps:.1f}  Latencia: {latencia_ms:.1f}ms"
    cv2.putText(barra, fps_str, (w - 280, 24),
                FONT, 0.55, (100, 220, 100), 1, cv2.LINE_AA)

    progreso = f"Frame {n_frame}/{total}  {'[PAUSADO]' if pausado else ''}"
    cv2.putText(barra, progreso, (w - 280, 50),
                FONT, 0.48, (160, 160, 160), 1, cv2.LINE_AA)

    # Indicadores de escena (fila compacta)
    indicadores = [
        ("FC", escena.frente_cercano_ocupado, (0, 220, 0)),
        ("FL", escena.frente_lejano_ocupado,  (0, 180, 160)),
        ("PEA", escena.peaton_en_riesgo,       (0, 0, 255)),
        ("SEM", escena.semaforo_visible is not None, (255, 200, 0)),
        ("ALTO", escena.senal_alto_cercana,    (0, 220, 220)),
        ("EI",  escena.espejo_izq_ocupado,     (255, 120, 0)),
        ("ED",  escena.espejo_der_ocupado,     (255, 0, 120)),
    ]
    x_ind = 10
    for nombre, activo, color in indicadores:
        col = color if activo else (50, 50, 50)
        cv2.putText(barra, nombre, (x_ind, 50 + 28),
                    FONT, 0.48, col, 1 if not activo else 2, cv2.LINE_AA)
        x_ind += len(nombre) * 12 + 12

    return np.vstack([barra, canvas])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="datos/videos/ets2_volvo_fh16.f299.mp4")
    parser.add_argument("--modelo", default="datos/modelos/yolo11n.pt")
    parser.add_argument("--rois", default="config/regiones_interes.yaml")
    parser.add_argument("--frame-inicio", type=int, default=6000)
    parser.add_argument("--velocidad", type=float, default=1.0,
                        help="Factor de velocidad de reproducción (1.0 = normal, 2.0 = doble)")
    args = parser.parse_args()

    # Cargar ROI
    rois_raw = cargar_rois_yaml(args.rois) if Path(args.rois).exists() else {}
    print(f"ROI cargadas: {len(rois_raw)} regiones")

    # Inicializar componentes
    print("Cargando modelo YOLO...")
    tracker = Tracker(ruta_modelo=args.modelo, device="cuda")
    tracker.cargar()
    contexto = AnalizadorContexto(rois=rois_raw or None)
    fsm = FSMDecision()

    cap = cv2.VideoCapture(args.video)
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_inicio)
    n_frame = args.frame_inicio

    ventana = "Detector ETS2 — Q/ESC para salir | ESPACIO pausa | ← → navegar"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana, 1280, 780)

    pausado = False
    paso_frames = int(fps_video * 5)  # 5 segundos
    Path("datos/evidencia").mkdir(parents=True, exist_ok=True)

    print(f"\nReproduciendo desde frame {args.frame_inicio}...")
    print("ESPACIO = pausar | ← → = ±5s | G = guardar frame | Q = salir\n")

    t_ultimo = time.perf_counter()

    while cap.isOpened():
        if not pausado:
            ok, frame = cap.read()
            if not ok:
                break
            n_frame += 1
        else:
            # En pausa mostramos el último frame sin avanzar
            pass

        t0 = time.perf_counter()

        # Pipeline percepción + decisión
        seguimientos = tracker.rastrear(frame)
        escena = contexto.analizar(seguimientos, frame)
        resultado = fsm.decidir(escena)

        latencia_ms = (time.perf_counter() - t0) * 1000
        fps_actual = 1.0 / max(time.perf_counter() - t_ultimo, 1e-9)
        t_ultimo = time.perf_counter()

        # Visualización
        canvas = frame.copy()
        canvas = dibujar_rois(canvas, rois_raw)
        canvas = dibujar_detecciones(canvas, seguimientos)
        canvas = dibujar_hud(canvas, resultado, escena, fps_actual, latencia_ms,
                             n_frame, total, pausado)

        cv2.imshow(ventana, canvas)

        # Control de velocidad: esperar el tiempo correspondiente
        ms_espera = max(1, int(1000 / (fps_video * args.velocidad))) if not pausado else 30
        key = cv2.waitKey(ms_espera) & 0xFF

        if key in (ord('q'), 27):       # Q o ESC — salir
            break
        elif key == 32:                  # ESPACIO — pausa
            pausado = not pausado
            print("PAUSADO" if pausado else "REANUDADO")
        elif key in (83, ord('d')):      # → o D — adelantar (en pausa)
            idx = min(total - 1, n_frame + paso_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            n_frame = idx
        elif key in (81, ord('a')):      # ← o A — retroceder (en pausa)
            idx = max(0, n_frame - paso_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            n_frame = idx
        elif key == ord('g'):            # G — guardar frame
            nombre = f"datos/evidencia/frame_{n_frame:06d}.png"
            cv2.imwrite(nombre, canvas)
            print(f"Frame guardado: {nombre}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nFin de la prueba de detección.")


if __name__ == "__main__":
    main()
