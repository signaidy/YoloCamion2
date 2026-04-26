"""Smoke test: verifica que YOLO carga, corre en GPU y detecta objetos."""
import time
from pathlib import Path

from ultralytics import YOLO

MODELO_DEFAULT = "datos/modelos/yolo11n.pt"
URL_PRUEBA = "https://ultralytics.com/images/bus.jpg"


def main():
    ruta = Path(MODELO_DEFAULT)
    print(f"Cargando modelo desde {ruta} (se descarga si no existe)...")
    modelo = YOLO(str(ruta) if ruta.exists() else "yolo11n.pt")

    print(f"Corriendo inferencia sobre {URL_PRUEBA}...")
    inicio = time.perf_counter()
    resultados = modelo(URL_PRUEBA, device="cuda", verbose=False)
    fin = time.perf_counter()

    print(f"\nTiempo de inferencia: {(fin - inicio) * 1000:.2f} ms")
    print(f"Detecciones encontradas: {len(resultados[0].boxes)}")
    for caja in resultados[0].boxes:
        clase = modelo.names[int(caja.cls[0])]
        conf = float(caja.conf[0])
        print(f"  {clase}: confianza={conf:.2f}")

    # Mover modelo a datos/modelos/ si se descargó en cwd
    for nombre in ("yolo11n.pt", "yolov8n.pt"):
        descargado = Path(nombre)
        if descargado.exists():
            destino = Path("datos/modelos") / nombre
            destino.parent.mkdir(parents=True, exist_ok=True)
            descargado.rename(destino)
            print(f"\nModelo movido a {destino}")
            break


if __name__ == "__main__":
    main()
