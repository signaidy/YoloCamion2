"""Registra prototipos de dígitos del velocímetro en un banco JSON.

Uso:
  python scripts/registrar_prototipos_velocidad.py \
      --output config/velocidad_dashboard_prototypes.json \
      --sample 3=datos/protos/comp_003.png \
      --sample 4=datos/protos/comp_014.png

Cada muestra debe ser una imagen monocromática del dígito ya recortado. Cualquier
pixel > 0 se considera trazo activo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def _rows_from_image(path: Path) -> list[str]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    _, mask = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    ys, xs = (mask > 0).nonzero()
    if xs.size == 0 or ys.size == 0:
        raise ValueError(f"Sin pixeles activos: {path}")
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    rows: list[str] = []
    for row in crop:
        rows.append("".join("#" if px > 0 else "." for px in row))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Ruta del banco JSON")
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Muestra en formato DIGITO=ruta_imagen",
    )
    args = parser.parse_args()

    output = Path(args.output)
    data: dict[str, list[dict[str, object]]] = {}
    if output.exists():
        data = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Banco inválido")

    for item in args.sample:
        if "=" not in item:
            raise ValueError(f"Muestra inválida: {item}")
        digit_str, path_str = item.split("=", 1)
        digit = int(digit_str)
        if digit < 0 or digit > 9:
            raise ValueError(f"Dígito fuera de rango: {digit}")
        rows = _rows_from_image(Path(path_str))
        data.setdefault(str(digit), []).append(
            {
                "rows": rows,
                "source": path_str,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
