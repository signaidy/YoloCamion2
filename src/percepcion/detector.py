from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO

from src.tipos import Clase, Deteccion

# Mapeo de IDs COCO a clases propias del proyecto
_COCO_A_CLASE: dict[int, Clase] = {
    0: Clase.PEATON,
    2: Clase.VEHICULO,      # car
    3: Clase.MOTOCICLETA,
    5: Clase.VEHICULO,      # bus
    7: Clase.VEHICULO,      # truck
    9: Clase.SEMAFORO,
    11: Clase.SENAL_ALTO,
}


class Detector:
    """Carga YOLO y convierte resultados a Deteccion con clases propias."""

    def __init__(
        self,
        ruta_modelo: str | Path = "datos/modelos/yolo11n.pt",
        confianza_min: float = 0.35,
        imgsz: int = 640,
        device: str = "cuda",
    ):
        self._ruta = str(ruta_modelo)
        self._confianza_min = confianza_min
        self._imgsz = imgsz
        self._device = device
        self._modelo: Optional[YOLO] = None

    def cargar(self) -> None:
        self._modelo = YOLO(self._ruta)

    def detectar(self, imagen: np.ndarray) -> list[Deteccion]:
        if self._modelo is None:
            raise RuntimeError("Llama a cargar() antes de detectar()")

        resultados = self._modelo(
            imagen,
            conf=self._confianza_min,
            device=self._device,
            imgsz=self._imgsz,
            verbose=False,
        )

        detecciones: list[Deteccion] = []
        for caja in resultados[0].boxes:
            id_coco = int(caja.cls[0])
            clase = _COCO_A_CLASE.get(id_coco, Clase.DESCONOCIDO)
            x1, y1, x2, y2 = (int(v) for v in caja.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            detecciones.append(
                Deteccion(
                    clase=clase,
                    caja=(x1, y1, x2, y2),
                    confianza=float(caja.conf[0]),
                    area=area,
                )
            )
        return detecciones
