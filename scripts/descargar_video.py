"""Descarga un video de YouTube (gameplay de ETS2) para pruebas reproducibles."""
import argparse
import subprocess
import sys
from pathlib import Path


def descargar(url: str, salida: Path, resolucion: str = "1080") -> Path:
    salida.parent.mkdir(parents=True, exist_ok=True)
    comando = [
        sys.executable, "-m", "yt_dlp",
        "-f", f"bestvideo[height<={resolucion}][ext=mp4]+bestaudio[ext=m4a]/best[height<={resolucion}]",
        "--merge-output-format", "mp4",
        "-o", str(salida),
        url,
    ]
    print("Ejecutando:", " ".join(comando))
    subprocess.run(comando, check=True)
    return salida


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga gameplay de ETS2 de YouTube para pruebas"
    )
    parser.add_argument("url", help="URL de YouTube")
    parser.add_argument("--salida", default="datos/videos/ets2_gameplay.mp4")
    parser.add_argument("--resolucion", default="1080")
    args = parser.parse_args()

    ruta = descargar(args.url, Path(args.salida), args.resolucion)
    print(f"\nGuardado en: {ruta}")
    print(f"Tamaño: {ruta.stat().st_size / 1e6:.1f} MB")
