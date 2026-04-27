"""Captura la ventana de ETS2 y muestra la ROI y lo que detecta el algoritmo.

Ejecutar con ETS2 abierto en modo Ventana. No necesita que el juego
esté en primer plano — captura la ventana aunque esté detrás de la terminal.

Guarda: datos/evidencia/debug_roi.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from src.fuente.ventana import buscar_ventana, _capturar_hwnd
from src.percepcion.carriles import DetectorCarriles

import ctypes.wintypes, ctypes

Path("datos/evidencia").mkdir(parents=True, exist_ok=True)

print("Buscando ventana de ETS2...")
hwnd = buscar_ventana("Euro Truck Simulator 2")
if not hwnd:
    print("ERROR: ETS2 no encontrado. Abrir el juego primero.")
    sys.exit(1)

rect = ctypes.wintypes.RECT()
ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
w_orig, h_orig = rect.right, rect.bottom
print(f"ETS2 hwnd={hwnd} resolución={w_orig}x{h_orig}")

frame_raw = _capturar_hwnd(hwnd, w_orig, h_orig)
if frame_raw is None:
    print("ERROR: No se pudo capturar la ventana.")
    sys.exit(1)

# Escalar a 1920x1080 igual que hace el piloto
frame = cv2.resize(frame_raw, (1920, 1080), interpolation=cv2.INTER_LINEAR)
h, w = frame.shape[:2]

# ROI del detector
ROI = (0.15, 0.52, 0.85, 0.83)
x1 = int(w * ROI[0])
y1 = int(h * ROI[1])
x2 = int(w * ROI[2])
y2 = int(h * ROI[3])

roi_img = frame[y1:y2, x1:x2]

# Aplicar el mismo preprocesado que usa DetectorCarriles
gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_eq = clahe.apply(gray)
_, mask = cv2.threshold(gray_eq, 110, 255, cv2.THRESH_BINARY)

# ── Imagen 1: frame completo con ROI y overlay de máscara ────────────────────
canvas = frame.copy()
cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 3)
cv2.putText(canvas, f"ROI ({x1},{y1})→({x2},{y2})", (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Colorear en verde las zonas que el detector ve como líneas brillantes
mask_overlay = np.zeros_like(roi_img)
mask_overlay[:, :, 1] = mask   # canal verde = zonas brillantes
canvas[y1:y2, x1:x2] = cv2.addWeighted(canvas[y1:y2, x1:x2], 0.6,
                                         mask_overlay, 0.4, 0)

ruta1 = "datos/evidencia/debug_roi.png"
cv2.imwrite(ruta1, canvas)

# ── Imagen 2: zoom de la ROI con máscara y bordes ────────────────────────────
edges = cv2.Canny(mask, 30, 90)
roi_debug = roi_img.copy()
roi_debug[mask > 0] = [0, 200, 0]         # verde = zona brillante
roi_debug[edges > 0] = [0, 0, 255]        # rojo = bordes detectados

ruta2 = "datos/evidencia/debug_roi_zoom.png"
cv2.imwrite(ruta2, roi_debug)

# ── Ejecutar el detector y mostrar resultado ─────────────────────────────────
det = DetectorCarriles()
estado = det.detectar(frame)

print(f"\nResultado del detector:")
print(f"  Desviación:  {estado.desviacion:+.3f}")
print(f"  Confianza:   {estado.confianza:.2f}")
print(f"  Línea izq:   {'✓' if estado.linea_izq_detectada else '✗'}")
print(f"  Línea der:   {'✓' if estado.linea_der_detectada else '✗'}")
print(f"\nImágenes guardadas:")
print(f"  {ruta1}  ← frame completo con ROI")
print(f"  {ruta2} ← zoom ROI con detecciones")
print(f"\nEn debug_roi_zoom.png:")
print(f"  Verde = zonas brillantes (posibles líneas de carril)")
print(f"  Rojo  = bordes usados por HoughLines")
print(f"\nSi no hay verde, el umbral de brillo es muy alto o")
print(f"la ROI no apunta a la zona con líneas del carril.")
