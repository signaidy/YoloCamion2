"""Calibrador interactivo de Regiones de Interés (ROI) para el Volvo FH16.

Permite navegar el video para encontrar un frame en primera persona, y luego
dibujar con el mouse los rectángulos de cada zona.

Uso:
  python scripts/calibrar_regiones.py --video datos/videos/ets2_volvo_fh16.f299.mp4
  python scripts/calibrar_regiones.py --imagen captura.png

Controles en el NAVEGADOR DE FRAMES:
  → / D          → avanzar 1 segundo (60 frames)
  ← / A          → retroceder 1 segundo
  Page Down      → avanzar 30 segundos
  Page Up        → retroceder 30 segundos
  Fin            → ir al final del video
  Inicio         → ir al principio
  ENTER / SPACE  → usar este frame para calibrar
  ESC            → salir

Controles durante la CALIBRACIÓN:
  Clic + arrastrar  → dibuja el rectángulo de la región actual
  ENTER / SPACE     → confirma la región y pasa a la siguiente
  R                 → rehace la región actual
  S                 → guarda y sale
  ESC               → sale sin guardar
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Orden y nombres de las regiones a calibrar
REGIONES = [
    ("frente_cercano", "FRENTE CERCANO",  (0, 200, 0),    "Zona frontal baja (vehículos cerca, señales de alto)"),
    ("frente_lejano",  "FRENTE LEJANO",   (0, 255, 200),  "Zona frontal alta (vehículos a distancia, semáforos)"),
    ("espejo_izq",     "ESPEJO IZQ",      (255, 100, 0),  "Espejo retrovisor izquierdo"),
    ("espejo_der",     "ESPEJO DER",      (255, 0, 100),  "Espejo retrovisor derecho"),
    ("lateral_izq",    "LATERAL IZQ",     (200, 0, 255),  "Lateral izquierdo (rebase)"),
    ("lateral_der",    "LATERAL DER",     (100, 0, 255),  "Lateral derecho (rebase)"),
]

VENTANA = "Calibrador ROI — ETS2 Volvo FH16"
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Calibrador:
    def __init__(self, imagen: np.ndarray):
        self._original = imagen.copy()
        self._h, self._w = imagen.shape[:2]
        self._rois: dict[str, list[int]] = {}

        # Estado del dibujo actual
        self._dibujando = False
        self._pt1 = (0, 0)
        self._pt2 = (0, 0)
        self._rect_actual: list[int] | None = None

    def _canvas(self) -> np.ndarray:
        canvas = self._original.copy()

        # Dibujar regiones ya confirmadas
        for nombre, (_, label, color, _) in zip(self._rois.keys(),
                [r for r in REGIONES if r[0] in self._rois]):
            x1, y1, x2, y2 = self._rois[nombre]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, label, (x1 + 4, y1 + 20),
                        FONT, 0.6, color, 2, cv2.LINE_AA)

        # Rectángulo en curso
        if self._rect_actual:
            x1, y1, x2, y2 = self._rect_actual
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

        return canvas

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dibujando = True
            self._pt1 = (x, y)
            self._rect_actual = None

        elif event == cv2.EVENT_MOUSEMOVE and self._dibujando:
            x1, y1 = self._pt1
            self._rect_actual = [min(x1, x), min(y1, y), max(x1, x), max(y1, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            self._dibujando = False
            x1, y1 = self._pt1
            self._rect_actual = [min(x1, x), min(y1, y), max(x1, x), max(y1, y)]

    def _instruccion(self, canvas: np.ndarray, idx: int) -> np.ndarray:
        """Panel de instrucciones en la parte inferior."""
        nombre, label, color, descripcion = REGIONES[idx]
        panel_h = 80
        panel = np.zeros((panel_h, self._w, 3), dtype=np.uint8)

        progreso = f"[{idx + 1}/{len(REGIONES)}]"
        cv2.putText(panel, f"{progreso} Dibuja: {label}", (10, 24),
                    FONT, 0.75, color, 2, cv2.LINE_AA)
        cv2.putText(panel, descripcion, (10, 48),
                    FONT, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(panel, "ENTER/SPACE = confirmar  |  R = rehacer  |  S = guardar ya  |  ESC = salir sin guardar",
                    (10, 70), FONT, 0.42, (120, 120, 120), 1, cv2.LINE_AA)

        return np.vstack([canvas, panel])

    def ejecutar(self) -> dict[str, list[int]] | None:
        cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(VENTANA, min(self._w, 1600), min(self._h + 80, 1000))
        cv2.setMouseCallback(VENTANA, self._mouse_cb)

        idx = 0
        while idx < len(REGIONES):
            nombre, label, color, descripcion = REGIONES[idx]
            self._rect_actual = None
            self._dibujando = False

            # Si ya estaba definida (re-calibración), mostrarla pre-cargada
            if nombre in self._rois:
                self._rect_actual = list(self._rois[nombre])

            while True:
                canvas = self._canvas()
                frame = self._instruccion(canvas, idx)
                cv2.imshow(VENTANA, frame)
                key = cv2.waitKey(30) & 0xFF

                if key in (13, 32):  # ENTER o SPACE — confirmar
                    if self._rect_actual:
                        x1, y1, x2, y2 = self._rect_actual
                        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                            self._rois[nombre] = [x1, y1, x2, y2]
                            print(f"  ✓ {label}: [{x1}, {y1}, {x2}, {y2}]")
                            idx += 1
                            break
                        else:
                            print(f"  ! Rectángulo demasiado pequeño para {label}, dibuja de nuevo")
                    else:
                        print(f"  ! Dibuja el rectángulo para {label} antes de confirmar")

                elif key == ord('r'):  # R — rehacer
                    self._rect_actual = None
                    if nombre in self._rois:
                        del self._rois[nombre]
                    print(f"  ↺ Rehaciendo {label}")

                elif key == ord('s'):  # S — guardar lo que hay y salir
                    print("  Guardando regiones parciales...")
                    cv2.destroyAllWindows()
                    return self._rois if self._rois else None

                elif key == 27:  # ESC — salir sin guardar
                    print("  Calibración cancelada sin guardar.")
                    cv2.destroyAllWindows()
                    return None

        cv2.destroyAllWindows()
        return self._rois


def navegar_video(ruta_video: str, frame_inicio: int = 0) -> np.ndarray | None:
    """Abre una ventana para navegar el video frame a frame hasta encontrar
    la vista en primera persona correcta. Devuelve el frame seleccionado."""
    cap = cv2.VideoCapture(ruta_video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    paso_seg   = max(1, int(fps))        # 1 segundo
    paso_30s   = max(1, int(fps * 30))   # 30 segundos
    paso_5min  = max(1, int(fps * 300))  # 5 minutos

    idx = max(0, min(frame_inicio, total - 1))
    ventana = "Navegador de frames — ENTER para calibrar, ESC para salir"

    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana, 1280, 760)

    frame_actual = None

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            idx = max(0, idx - 1)
            continue

        frame_actual = frame.copy()
        canvas = frame.copy()

        # HUD de navegación
        t_seg = idx / fps
        t_str = f"{int(t_seg // 60):02d}:{int(t_seg % 60):02d}"
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 56), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        cv2.putText(canvas,
                    f"Frame: {idx}/{total}   Tiempo: {t_str}   "
                    f"(← → = 1s  |  PgUp PgDn = 30s  |  ENTER = usar este frame)",
                    (10, 22), FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas,
                    "Busca un frame en PRIMERA PERSONA donde se vean los espejos del Volvo FH16",
                    (10, 46), FONT, 0.55, (80, 220, 80), 1, cv2.LINE_AA)

        cv2.imshow(ventana, canvas)
        key = cv2.waitKey(0) & 0xFF

        if key in (13, 32):          # ENTER / SPACE — seleccionar
            print(f"  Frame seleccionado: {idx} (tiempo {t_str})")
            break
        elif key == 27:              # ESC — cancelar
            frame_actual = None
            break
        elif key in (83, 100):       # → o D — +1 segundo
            idx = min(total - 1, idx + paso_seg)
        elif key in (81, 97):        # ← o A — -1 segundo
            idx = max(0, idx - paso_seg)
        elif key == 118:             # Page Down (código 118 en algunos sistemas) — +30s
            idx = min(total - 1, idx + paso_30s)
        elif key == 117:             # Page Up — -30s
            idx = max(0, idx - paso_30s)
        elif key == 54:              # 6 — +5 min
            idx = min(total - 1, idx + paso_5min)
        elif key == 52:              # 4 — -5 min
            idx = max(0, idx - paso_5min)
        elif key == 103:             # G — ir al frame por número
            pass  # simplificado: ignorar

        # También manejar teclas especiales de OpenCV (flechas = 2228224, etc.)
        # En Windows las flechas devuelven valores distintos
        if key == 0:
            key2 = cv2.waitKey(0) & 0xFF
            if key2 == 75:   # flecha izq
                idx = max(0, idx - paso_seg)
            elif key2 == 77: # flecha der
                idx = min(total - 1, idx + paso_seg)
            elif key2 == 73: # Page Up
                idx = max(0, idx - paso_30s)
            elif key2 == 81: # Page Down
                idx = min(total - 1, idx + paso_30s)

    cap.release()
    cv2.destroyAllWindows()
    return frame_actual


def cargar_frame(ruta_video: str, n_frame: int = 300) -> np.ndarray:
    cap = cv2.VideoCapture(ruta_video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frame = min(n_frame, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"No se pudo leer el frame {n_frame} de {ruta_video}")
    return frame


def guardar_yaml(rois: dict, ruta: str) -> None:
    datos = {nombre: coords for nombre, coords in rois.items()}
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("# Regiones de interés calibradas para el Volvo FH16\n")
        f.write(f"# Resolución de referencia: definida al calibrar\n")
        f.write("# Formato: [x1, y1, x2, y2] en píxeles\n\n")
        yaml.dump(datos, f, default_flow_style=True, allow_unicode=True)
    print(f"\n✓ Guardado en {ruta}")


def previsualizar(imagen: np.ndarray, rois: dict) -> None:
    """Muestra el resultado final con todas las regiones."""
    canvas = imagen.copy()
    for nombre, (_, label, color, _) in zip(rois.keys(),
            [r for r in REGIONES if r[0] in rois]):
        x1, y1, x2, y2 = rois[nombre]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1 + 4, y1 + 20),
                    FONT, 0.65, color, 2, cv2.LINE_AA)

    cv2.putText(canvas, "Resultado final — presiona cualquier tecla para cerrar",
                (10, canvas.shape[0] - 10), FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.namedWindow("ROI Calibradas", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI Calibradas", min(canvas.shape[1], 1600), min(canvas.shape[0], 900))
    cv2.imshow("ROI Calibradas", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Calibrador interactivo de ROI")
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--video", help="Ruta al video de gameplay")
    grupo.add_argument("--imagen", help="Ruta a una imagen/screenshot")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame inicial para el navegador (default: 0)")
    parser.add_argument("--salida", default="config/regiones_interes.yaml",
                        help="Ruta del YAML de salida")
    args = parser.parse_args()

    # Cargar imagen de referencia
    if args.video:
        print(f"\nAbriendo navegador de frames para: {args.video}")
        print("Usa ← → para moverte 1 segundo, Page Up/Down para 30 segundos.")
        print("Presiona ENTER cuando veas un frame en primera persona con los espejos visibles.\n")
        imagen = navegar_video(args.video, args.frame)
        if imagen is None:
            print("Calibración cancelada.")
            sys.exit(0)
        h, w = imagen.shape[:2]
        print(f"Frame seleccionado: {w}x{h}")
    else:
        imagen = cv2.imread(args.imagen)
        if imagen is None:
            print(f"Error: no se pudo abrir {args.imagen}")
            sys.exit(1)
        h, w = imagen.shape[:2]
        print(f"Imagen cargada: {w}x{h}")

    print(f"\n{'='*60}")
    print("  CALIBRADOR DE REGIONES DE INTERÉS — Volvo FH16 ETS2")
    print(f"{'='*60}")
    print("\nInstrucciones:")
    print("  1. Arrastra el mouse para dibujar cada región")
    print("  2. Presiona ENTER o SPACE para confirmar")
    print("  3. Presiona R para rehacer la región actual")
    print("  4. Presiona S para guardar lo que hay y salir")
    print("  5. Presiona ESC para salir sin guardar")
    print("\nRegiones a calibrar en orden:")
    for i, (nombre, label, _, desc) in enumerate(REGIONES, 1):
        print(f"  {i}. {label}: {desc}")
    print()

    calibrador = Calibrador(imagen)
    rois = calibrador.ejecutar()

    if rois:
        guardar_yaml(rois, args.salida)
        print("\nRegiones guardadas:")
        for nombre, coords in rois.items():
            print(f"  {nombre}: {coords}")
        previsualizar(imagen, rois)
    else:
        print("No se guardaron regiones.")


if __name__ == "__main__":
    main()
