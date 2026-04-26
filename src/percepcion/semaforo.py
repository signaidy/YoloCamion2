import cv2
import numpy as np

from src.tipos import EstadoSemaforo

# Rangos HSV para cada color (H en 0-179 en OpenCV)
_HSV_ROJO_1 = (np.array([0, 100, 100]),   np.array([10, 255, 255]))
_HSV_ROJO_2 = (np.array([160, 100, 100]), np.array([179, 255, 255]))
_HSV_AMARILLO = (np.array([15, 100, 100]), np.array([35, 255, 255]))
_HSV_VERDE   = (np.array([40, 80, 80]),   np.array([90, 255, 255]))


def clasificar_semaforo(imagen: np.ndarray, caja: tuple[int, int, int, int]) -> EstadoSemaforo:
    """Clasifica el estado de un semáforo por análisis de color HSV en su ROI."""
    x1, y1, x2, y2 = caja
    roi = imagen[y1:y2, x1:x2]
    if roi.size == 0:
        return EstadoSemaforo.DESCONOCIDO

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total = roi.shape[0] * roi.shape[1]
    if total == 0:
        return EstadoSemaforo.DESCONOCIDO

    px_rojo = (
        cv2.countNonZero(cv2.inRange(hsv, *_HSV_ROJO_1))
        + cv2.countNonZero(cv2.inRange(hsv, *_HSV_ROJO_2))
    )
    px_amarillo = cv2.countNonZero(cv2.inRange(hsv, *_HSV_AMARILLO))
    px_verde = cv2.countNonZero(cv2.inRange(hsv, *_HSV_VERDE))

    conteos = {
        EstadoSemaforo.ROJO: px_rojo,
        EstadoSemaforo.AMARILLO: px_amarillo,
        EstadoSemaforo.VERDE: px_verde,
    }
    ganador, max_px = max(conteos.items(), key=lambda x: x[1])

    umbral_minimo = max(total * 0.05, 10)
    if max_px < umbral_minimo:
        return EstadoSemaforo.DESCONOCIDO
    return ganador
