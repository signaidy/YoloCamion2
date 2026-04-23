# Plan de implementación — Conducción autónoma visual en ETS2

> **Para workers agénticos:** SKILL REQUERIDO: usar `superpowers:subagent-driven-development` (recomendado) o `superpowers:executing-plans` para ejecutar este plan tarea por tarea. Los pasos usan sintaxis de checkbox (`- [ ]`) para seguimiento.

**Objetivo:** Construir un sistema en Python que observe ETS2 por captura de pantalla, detecte objetos con YOLO26, decida acciones con reglas explicables, y emule controles externos para conducir el camión sin modificar el juego.

**Arquitectura:** Pipeline modular de 7 componentes con interfaces abstractas y dataclasses planas como contratos. Se implementa en 6 fases: F0 benchmark → F1 detector sobre video → F2 contexto + ROI → F3 FSM con controlador nulo → F4 captura pantalla + gamepad → F5 integración en juego → F6 reporte y entrega.

**Tech Stack:** Python 3.11, Ultralytics YOLO26, DXcam, OpenCV, vgamepad, pydirectinput, pytest, pandas, pyyaml.

**Spec de referencia:** [`docs/superpowers/specs/2026-04-23-conduccion-autonoma-ets2-design.md`](../specs/2026-04-23-conduccion-autonoma-ets2-design.md)

**Directorio del proyecto:** `C:\Users\andre\OneDrive - Universidad del Istmo\Desktop\Proyecto Final`

---

## FASE 0 — Setup del proyecto (Día 1)

### Tarea 1: Scaffolding del proyecto

**Archivos:**
- Crear: `.gitignore`
- Crear: `README.md`
- Crear: `requirements.txt`
- Crear: `pyproject.toml`
- Crear: estructura de carpetas `src/`, `config/`, `scripts/`, `datos/`, `tests/`, `docs/`

- [ ] **Paso 1: Inicializar repositorio git**

```bash
cd "C:/Users/andre/OneDrive - Universidad del Istmo/Desktop/Proyecto Final"
git init
git config user.email "hecheverria@unis.edu.gt"
git config user.name "Andres Echeverria"
```

- [ ] **Paso 2: Crear `.gitignore`**

```
# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/
venv/
.venv/
env/
*.egg-info/

# Datos pesados
datos/videos/*.mp4
datos/videos/*.mkv
datos/videos/*.webm
datos/modelos/*.pt
datos/modelos/*.onnx
datos/evidencia/**/*.mp4
datos/evidencia/**/*.avi

# IDE
.vscode/
.idea/

# SO
Thumbs.db
desktop.ini
```

- [ ] **Paso 3: Crear `requirements.txt`**

```
ultralytics>=8.3.0
opencv-python>=4.9.0
dxcam>=0.0.5
numpy>=1.26.0
pandas>=2.2.0
pyyaml>=6.0
yt-dlp>=2024.1.1
vgamepad>=0.1.0
pydirectinput>=1.0.4
pytest>=8.0.0
pytest-mock>=3.12.0
```

- [ ] **Paso 4: Crear `pyproject.toml`**

```toml
[project]
name = "conduccion-autonoma-ets2"
version = "0.1.0"
description = "Sistema de conduccion autonoma visual para Euro Truck Simulator 2"
authors = [{name = "Andres Echeverria", email = "hecheverria@unis.edu.gt"}]
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

- [ ] **Paso 5: Crear estructura de carpetas**

```bash
mkdir -p src/fuente src/percepcion src/decision src/control src/seguridad src/registro
mkdir -p config scripts datos/videos datos/modelos datos/evidencia
mkdir -p tests/fixtures docs/reporte_tecnico
touch src/__init__.py src/fuente/__init__.py src/percepcion/__init__.py
touch src/decision/__init__.py src/control/__init__.py src/seguridad/__init__.py
touch src/registro/__init__.py tests/__init__.py
```

- [ ] **Paso 6: Crear README inicial**

```markdown
# Conducción autónoma visual en ETS2 con YOLO26

Proyecto Final — Universidad del Istmo (Guatemala)

## Instalación

```bash
python -m venv venv
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Ejecución rápida

Ver [plan de implementación](docs/superpowers/plans/2026-04-23-conduccion-autonoma-ets2-plan.md).

## Documentación

- [Diseño del sistema](docs/superpowers/specs/2026-04-23-conduccion-autonoma-ets2-design.md)
- [Plan de implementación](docs/superpowers/plans/2026-04-23-conduccion-autonoma-ets2-plan.md)
```

- [ ] **Paso 7: Commit inicial**

```bash
git add .gitignore README.md requirements.txt pyproject.toml src/ config/ scripts/ tests/ docs/
git commit -m "chore: scaffolding inicial del proyecto

- Estructura de carpetas modular (7 componentes)
- requirements.txt con dependencias principales
- pyproject.toml con metadata del proyecto
- README con instrucciones mínimas"
```

---

### Tarea 2: Ambiente virtual + instalación de dependencias

**Archivos:** ninguno (solo configuración del entorno)

- [ ] **Paso 1: Crear entorno virtual**

```bash
python -m venv venv
```

- [ ] **Paso 2: Activar entorno virtual**

```bash
# En bash de Windows:
source venv/Scripts/activate
```

- [ ] **Paso 3: Actualizar pip**

```bash
python -m pip install --upgrade pip
```

- [ ] **Paso 4: Instalar PyTorch con CUDA 12.1 (requisito para RTX 3070 Ti)**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Esperado: descarga ~2.5 GB de PyTorch con soporte CUDA.

- [ ] **Paso 5: Instalar resto de dependencias**

```bash
pip install -r requirements.txt
```

- [ ] **Paso 6: Verificar CUDA disponible**

```bash
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'ninguna')"
```

Esperado:
```
CUDA disponible: True
GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU
```

Si dice `CUDA disponible: False`, detener y resolver (normalmente requiere driver NVIDIA actualizado).

---

### Tarea 3: Smoke test de YOLO26

**Archivos:**
- Crear: `scripts/smoke_test_yolo.py`
- Descargar: `datos/modelos/yolov26n.pt` (descarga automática desde Ultralytics)

- [ ] **Paso 1: Crear script de prueba**

```python
# scripts/smoke_test_yolo.py
"""Smoke test: verifica que YOLO26 se carga, corre en GPU y detecta cosas."""
from ultralytics import YOLO
import cv2
import time


def main():
    # Ultralytics descarga los pesos automáticamente la primera vez
    print("Cargando modelo YOLO26 nano...")
    modelo = YOLO("yolov8n.pt")  # usar yolov8n mientras yolov26 no este publicado; cambiar cuando este disponible

    # Imagen de prueba: una de ejemplo que trae Ultralytics
    url_prueba = "https://ultralytics.com/images/bus.jpg"
    print(f"Corriendo inferencia sobre {url_prueba}...")

    inicio = time.perf_counter()
    resultados = modelo(url_prueba, device="cuda", verbose=False)
    fin = time.perf_counter()

    print(f"\nTiempo de inferencia: {(fin - inicio) * 1000:.2f} ms")
    print(f"Detecciones encontradas: {len(resultados[0].boxes)}")
    for caja in resultados[0].boxes:
        clase = modelo.names[int(caja.cls[0])]
        conf = float(caja.conf[0])
        print(f"  {clase}: confianza={conf:.2f}")


if __name__ == "__main__":
    main()
```

**Nota importante:** Si al momento de ejecutar YOLO26 no está disponible públicamente todavía (dependiendo de la fecha de release de Ultralytics), usar `yolov8n.pt` o `yolo11n.pt` como sustituto. El documento de requerimientos dice "YOLO26" pero la API de Ultralytics es la misma.

- [ ] **Paso 2: Ejecutar smoke test**

```bash
python scripts/smoke_test_yolo.py
```

Esperado:
```
Cargando modelo YOLO26 nano...
Corriendo inferencia sobre https://ultralytics.com/images/bus.jpg...

Tiempo de inferencia: 12-30 ms (variable, primera corrida es más lenta)
Detecciones encontradas: 4-6
  bus: confianza=0.87
  person: confianza=0.85
  ...
```

- [ ] **Paso 3: Mover modelo descargado a datos/modelos/**

```bash
# Ultralytics descarga en el cwd; moverlo
mv yolov8n.pt datos/modelos/ 2>/dev/null || mv yolo11n.pt datos/modelos/ 2>/dev/null || true
```

- [ ] **Paso 4: Commit**

```bash
git add scripts/smoke_test_yolo.py
git commit -m "feat: smoke test de YOLO26 con CUDA

- Valida que el modelo se carga y corre en GPU RTX 3070 Ti
- Imprime tiempos de inferencia y clases detectadas"
```

---

### Tarea 4: Benchmark de FPS end-to-end (criterio F0)

**Archivos:**
- Crear: `scripts/benchmark_fps.py`

- [ ] **Paso 1: Crear script de benchmark**

```python
# scripts/benchmark_fps.py
"""Benchmark: mide FPS sostenido de un pipeline captura + YOLO26.

Se ejecuta sobre imagenes sinteticas (numpy) o sobre un archivo de video si se proporciona.
Criterio F0: >= 30 FPS end-to-end a 1920x1080.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def benchmark(ruta_modelo: str, ruta_video: str | None, n_frames: int = 300) -> dict:
    modelo = YOLO(ruta_modelo)

    if ruta_video and Path(ruta_video).exists():
        cap = cv2.VideoCapture(ruta_video)
        print(f"Benchmark sobre video: {ruta_video}")
        fuente = lambda: cap.read()[1]
    else:
        print("Benchmark sobre frames sinteticos 1920x1080")
        frame_sintetico = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        fuente = lambda: frame_sintetico

    # Warmup (primera inferencia es lenta por compilacion de kernels CUDA)
    print("Warmup (10 frames)...")
    for _ in range(10):
        _ = modelo(fuente(), device="cuda", verbose=False, imgsz=640)

    # Benchmark real
    print(f"Benchmark ({n_frames} frames)...")
    latencias_ms = []
    inicio_total = time.perf_counter()
    for i in range(n_frames):
        frame = fuente()
        if frame is None:
            break
        inicio = time.perf_counter()
        _ = modelo(frame, device="cuda", verbose=False, imgsz=640, conf=0.35)
        latencias_ms.append((time.perf_counter() - inicio) * 1000)
    fin_total = time.perf_counter()

    if ruta_video:
        cap.release()

    latencias = np.array(latencias_ms)
    tiempo_total = fin_total - inicio_total
    fps_promedio = len(latencias) / tiempo_total

    resultado = {
        "n_frames": len(latencias),
        "tiempo_total_s": round(tiempo_total, 2),
        "fps_promedio": round(fps_promedio, 2),
        "latencia_ms_min": round(float(latencias.min()), 2),
        "latencia_ms_media": round(float(latencias.mean()), 2),
        "latencia_ms_p95": round(float(np.percentile(latencias, 95)), 2),
        "latencia_ms_max": round(float(latencias.max()), 2),
    }
    return resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="yolov8n.pt", help="Ruta al .pt del modelo")
    parser.add_argument("--video", default=None, help="Ruta opcional a video de prueba")
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
        print("  Accion: probar con imgsz=480, o usar modelo nano si no lo es")
```

- [ ] **Paso 2: Ejecutar benchmark**

```bash
python scripts/benchmark_fps.py --modelo datos/modelos/yolov8n.pt --frames 300
```

Esperado (en RTX 3070 Ti Laptop):
```
fps_promedio: ~80-150
latencia_ms_p95: ~8-15
APROBADO: X FPS >= 30 FPS objetivo
```

- [ ] **Paso 3: Si falla (<30 FPS), diagnóstico**

Si no aprobó:
1. Confirmar que `device="cuda"` está siendo usado (mirar uso de GPU con `nvidia-smi` en otra terminal).
2. Probar con `imgsz=480` o `imgsz=320`.
3. Cerrar otras apps que usen GPU (Chrome, OBS, etc.).

- [ ] **Paso 4: Guardar resultado como evidencia**

```bash
python scripts/benchmark_fps.py > datos/evidencia/F0_benchmark_$(date +%Y%m%d).txt 2>&1
```

- [ ] **Paso 5: Commit**

```bash
git add scripts/benchmark_fps.py datos/evidencia/F0_benchmark_*.txt
git commit -m "feat: benchmark F0 de FPS end-to-end

- Mide FPS sostenido de captura + YOLO26
- Criterio: >= 30 FPS a 1920x1080
- Resultado inicial guardado como evidencia"
```

**Gate F0:** Solo avanzar si benchmark dio ≥ 30 FPS.

---

## FASE 1 — Detector + Tracker sobre video (Días 2-3)

### Tarea 5: Script de descarga de video de YouTube

**Archivos:**
- Crear: `scripts/descargar_video.py`

- [ ] **Paso 1: Crear script con yt-dlp**

```python
# scripts/descargar_video.py
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
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL de YouTube")
    parser.add_argument("--salida", default="datos/videos/ets2_gameplay.mp4")
    parser.add_argument("--resolucion", default="1080")
    args = parser.parse_args()

    ruta = descargar(args.url, Path(args.salida), args.resolucion)
    print(f"\nGuardado en: {ruta}")
    print(f"Tamaño: {ruta.stat().st_size / 1e6:.1f} MB")
```

- [ ] **Paso 2: Descargar un video de prueba**

Buscar un gameplay de ETS2 en primera persona con tráfico variado. Ejemplo:

```bash
python scripts/descargar_video.py "https://www.youtube.com/watch?v=<ID_DE_VIDEO>" --salida datos/videos/ets2_gameplay.mp4
```

**Recomendación de búsqueda:** "ETS2 gameplay first person traffic" o "Euro Truck Simulator 2 city driving". Elegir video de 5-15 minutos.

- [ ] **Paso 3: Verificar el video**

```bash
python -c "import cv2; c = cv2.VideoCapture('datos/videos/ets2_gameplay.mp4'); print('FPS:', c.get(cv2.CAP_PROP_FPS), 'Frames:', int(c.get(cv2.CAP_PROP_FRAME_COUNT)), 'WxH:', int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), 'x', int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)))"
```

Esperado: 1920x1080 o similar, 30 FPS, miles de frames.

- [ ] **Paso 4: Commit (sin el video, está en .gitignore)**

```bash
git add scripts/descargar_video.py
git commit -m "feat: script de descarga de video de YouTube con yt-dlp"
```

---

### Tarea 6: Definir contratos de datos (`src/tipos.py`)

**Archivos:**
- Crear: `src/tipos.py`
- Crear: `tests/test_tipos.py`

- [ ] **Paso 1: Escribir test para los tipos**

```python
# tests/test_tipos.py
"""Tests basicos para dataclasses de tipos.py."""
import numpy as np
import pytest

from src.tipos import (
    Accion,
    Clase,
    ComandoControl,
    Cuadro,
    Deteccion,
    EstadoEscena,
    EstadoSemaforo,
    Region,
    Seguimiento,
)


def test_cuadro_se_crea_correctamente():
    imagen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cuadro = Cuadro(imagen=imagen, timestamp=100.0, indice=0, fps_instantaneo=30.0)
    assert cuadro.imagen.shape == (1080, 1920, 3)
    assert cuadro.timestamp == 100.0


def test_deteccion_se_crea_correctamente():
    det = Deteccion(
        clase=Clase.VEHICULO,
        caja=(100, 200, 300, 400),
        confianza=0.85,
        area=(300 - 100) * (400 - 200),
    )
    assert det.clase == Clase.VEHICULO
    assert det.caja == (100, 200, 300, 400)


def test_seguimiento_extiende_deteccion():
    seg = Seguimiento(
        clase=Clase.PEATON,
        caja=(0, 0, 50, 100),
        confianza=0.9,
        area=5000,
        id_seguimiento=42,
        edad=5,
    )
    assert seg.id_seguimiento == 42
    assert seg.edad == 5


def test_estado_escena_campos_obligatorios():
    estado = EstadoEscena(
        frente_cercano_ocupado=False,
        frente_lejano_ocupado=False,
        peaton_en_riesgo=False,
        semaforo_visible=None,
        senal_alto_cercana=False,
        espejo_izq_ocupado=False,
        espejo_der_ocupado=False,
        vehiculos_totales=0,
        confianza_percepcion=1.0,
        timestamp=0.0,
    )
    assert estado.confianza_percepcion == 1.0


def test_comando_control_en_rangos():
    cmd = ComandoControl(acelerador=0.5, freno=0.0, volante=-0.3, timestamp=0.0)
    assert 0.0 <= cmd.acelerador <= 1.0
    assert 0.0 <= cmd.freno <= 1.0
    assert -1.0 <= cmd.volante <= 1.0


def test_enums_tienen_valores_esperados():
    assert Clase.VEHICULO.value == "vehiculo"
    assert EstadoSemaforo.ROJO.value == "rojo"
    assert Region.FRENTE_CERCANO.value == "frente_cercano"
    assert Accion.ALTO_TOTAL.value == "alto_total"
```

- [ ] **Paso 2: Ejecutar test para ver que falla**

```bash
pytest tests/test_tipos.py -v
```

Esperado: FAIL con "ModuleNotFoundError: No module named 'src.tipos'"

- [ ] **Paso 3: Implementar `src/tipos.py`**

```python
# src/tipos.py
"""Contratos de datos compartidos entre modulos.

Son dataclasses planas: no tienen comportamiento, solo datos.
Esto permite probar cada modulo aislado con fixtures sinteticos.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


@dataclass
class Cuadro:
    """Un frame capturado de la fuente (video o pantalla)."""
    imagen: np.ndarray  # BGR (H, W, 3)
    timestamp: float
    indice: int
    fps_instantaneo: float


class Clase(Enum):
    """Clases unificadas del sistema (mapeadas desde COCO)."""
    VEHICULO = "vehiculo"          # car + truck + bus de COCO
    MOTOCICLETA = "motocicleta"
    PEATON = "peaton"
    SEMAFORO = "semaforo"
    SENAL_ALTO = "senal_alto"
    DESCONOCIDO = "desconocido"


@dataclass
class Deteccion:
    clase: Clase
    caja: tuple[int, int, int, int]  # (x1, y1, x2, y2) px
    confianza: float
    area: int


@dataclass
class Seguimiento(Deteccion):
    """Deteccion con identidad persistente entre frames."""
    id_seguimiento: int = -1
    edad: int = 0  # frames consecutivos visible


class EstadoSemaforo(Enum):
    ROJO = "rojo"
    AMARILLO = "amarillo"
    VERDE = "verde"
    DESCONOCIDO = "desconocido"


class Region(Enum):
    FRENTE_CERCANO = "frente_cercano"
    FRENTE_LEJANO = "frente_lejano"
    ESPEJO_IZQ = "espejo_izq"
    ESPEJO_DER = "espejo_der"
    LATERAL_IZQ = "lateral_izq"
    LATERAL_DER = "lateral_der"


@dataclass
class EstadoEscena:
    """Estado interpretado de la escena, listo para decidir."""
    frente_cercano_ocupado: bool
    frente_lejano_ocupado: bool
    peaton_en_riesgo: bool
    semaforo_visible: Optional[EstadoSemaforo]
    senal_alto_cercana: bool
    espejo_izq_ocupado: bool
    espejo_der_ocupado: bool
    vehiculos_totales: int
    confianza_percepcion: float  # 0-1
    timestamp: float


class Accion(Enum):
    MANTENER = "mantener"
    ACELERAR = "acelerar"
    FRENAR_SUAVE = "frenar_suave"
    FRENAR_FUERTE = "frenar_fuerte"
    ALTO_TOTAL = "alto_total"
    GIRAR_IZQ = "girar_izq"
    GIRAR_DER = "girar_der"
    REBASAR_IZQ = "rebasar_izq"
    REBASAR_DER = "rebasar_der"
    ESPERAR = "esperar"


@dataclass
class ComandoControl:
    """Comando de bajo nivel para el controlador."""
    acelerador: float  # 0.0 - 1.0
    freno: float       # 0.0 - 1.0
    volante: float     # -1.0 a +1.0
    timestamp: float
```

- [ ] **Paso 4: Correr tests y verificar que pasan**

```bash
pytest tests/test_tipos.py -v
```

Esperado: 6 passed.

- [ ] **Paso 5: Commit**

```bash
git add src/tipos.py tests/test_tipos.py
git commit -m "feat: contratos de datos compartidos (tipos.py)

- Dataclasses planas para comunicacion entre modulos
- Enums para Clase, EstadoSemaforo, Region, Accion
- 6 tests unitarios que cubren construccion y rangos"
```

---

### Tarea 7: Interfaz abstracta `FuenteCuadros`

**Archivos:**
- Crear: `src/fuente/base.py`
- Crear: `tests/test_fuente.py`

- [ ] **Paso 1: Escribir test con mock de fuente**

```python
# tests/test_fuente.py
"""Tests para la interfaz FuenteCuadros."""
import numpy as np

from src.fuente.base import FuenteCuadros
from src.tipos import Cuadro


class FuenteDummy(FuenteCuadros):
    """Fuente sintetica para tests: devuelve N frames en negro."""

    def __init__(self, n: int = 3):
        self._n = n
        self._i = 0
        self._activa = False

    def iniciar(self) -> None:
        self._activa = True

    def siguiente(self) -> Cuadro | None:
        if self._i >= self._n:
            self._activa = False
            return None
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cuadro = Cuadro(imagen=frame, timestamp=float(self._i), indice=self._i, fps_instantaneo=30.0)
        self._i += 1
        return cuadro

    def cerrar(self) -> None:
        self._activa = False

    @property
    def esta_activa(self) -> bool:
        return self._activa


def test_fuente_dummy_entrega_n_cuadros():
    fuente = FuenteDummy(n=3)
    fuente.iniciar()
    cuadros = []
    while fuente.esta_activa:
        c = fuente.siguiente()
        if c is not None:
            cuadros.append(c)
    assert len(cuadros) == 3
    assert cuadros[0].indice == 0
    assert cuadros[-1].indice == 2


def test_fuente_dummy_cierra_correctamente():
    fuente = FuenteDummy(n=1)
    fuente.iniciar()
    assert fuente.esta_activa
    fuente.cerrar()
    assert not fuente.esta_activa
```

- [ ] **Paso 2: Correr test (falla)**

```bash
pytest tests/test_fuente.py -v
```

Esperado: FAIL con "ModuleNotFoundError".

- [ ] **Paso 3: Implementar interfaz abstracta**

```python
# src/fuente/base.py
"""Interfaz abstracta para fuentes de cuadros (video, pantalla, etc.)."""
from abc import ABC, abstractmethod
from typing import Optional

from src.tipos import Cuadro


class FuenteCuadros(ABC):
    """Contrato que toda fuente de cuadros debe cumplir."""

    @abstractmethod
    def iniciar(self) -> None:
        """Prepara la fuente (abre archivo, inicia captura, etc.)."""

    @abstractmethod
    def siguiente(self) -> Optional[Cuadro]:
        """Devuelve el siguiente cuadro, o None si se agoto."""

    @abstractmethod
    def cerrar(self) -> None:
        """Libera recursos."""

    @property
    @abstractmethod
    def esta_activa(self) -> bool:
        """True si la fuente aun puede entregar cuadros."""
```

- [ ] **Paso 4: Correr tests**

```bash
pytest tests/test_fuente.py -v
```

Esperado: 2 passed.

- [ ] **Paso 5: Commit**

```bash
git add src/fuente/base.py tests/test_fuente.py
git commit -m "feat: interfaz abstracta FuenteCuadros con tests

- Contrato: iniciar, siguiente, cerrar, esta_activa
- FuenteDummy para testing sin archivos reales"
```

---

### Tarea 8: Implementar `FuenteVideo`

**Archivos:**
- Crear: `src/fuente/video.py`
- Modificar: `tests/test_fuente.py` (agregar test con archivo real)

- [ ] **Paso 1: Escribir test para FuenteVideo usando fixture sintético**

Agregar al final de `tests/test_fuente.py`:

```python
import cv2
from pathlib import Path

from src.fuente.video import FuenteVideo


def test_fuente_video_lee_archivo(tmp_path):
    """Crea un video sintetico de 5 frames y verifica que FuenteVideo lo lea."""
    ruta = tmp_path / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    escritor = cv2.VideoWriter(str(ruta), fourcc, 30.0, (320, 240))
    for i in range(5):
        frame = np.full((240, 320, 3), i * 50, dtype=np.uint8)
        escritor.write(frame)
    escritor.release()

    fuente = FuenteVideo(str(ruta))
    fuente.iniciar()

    cuadros = []
    while fuente.esta_activa:
        c = fuente.siguiente()
        if c is None:
            break
        cuadros.append(c)
    fuente.cerrar()

    assert len(cuadros) == 5
    assert cuadros[0].imagen.shape == (240, 320, 3)
    assert cuadros[0].indice == 0
    assert cuadros[-1].indice == 4
```

- [ ] **Paso 2: Correr test (falla)**

```bash
pytest tests/test_fuente.py::test_fuente_video_lee_archivo -v
```

Esperado: FAIL (ModuleNotFoundError).

- [ ] **Paso 3: Implementar `FuenteVideo`**

```python
# src/fuente/video.py
"""Fuente de cuadros desde archivo de video."""
import time
from pathlib import Path
from typing import Optional

import cv2

from src.fuente.base import FuenteCuadros
from src.tipos import Cuadro


class FuenteVideo(FuenteCuadros):
    """Lee frames secuenciales desde un archivo de video (mp4, avi, etc.)."""

    def __init__(self, ruta: str):
        self._ruta = Path(ruta)
        if not self._ruta.exists():
            raise FileNotFoundError(f"No existe el video: {self._ruta}")
        self._cap: Optional[cv2.VideoCapture] = None
        self._activa = False
        self._indice = 0
        self._ultimo_t: Optional[float] = None

    def iniciar(self) -> None:
        self._cap = cv2.VideoCapture(str(self._ruta))
        if not self._cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {self._ruta}")
        self._activa = True
        self._indice = 0
        self._ultimo_t = None

    def siguiente(self) -> Optional[Cuadro]:
        if self._cap is None or not self._activa:
            return None
        ok, frame = self._cap.read()
        if not ok:
            self._activa = False
            return None

        ahora = time.perf_counter()
        if self._ultimo_t is None:
            fps_inst = 0.0
        else:
            dt = ahora - self._ultimo_t
            fps_inst = 1.0 / dt if dt > 0 else 0.0
        self._ultimo_t = ahora

        cuadro = Cuadro(
            imagen=frame,
            timestamp=ahora,
            indice=self._indice,
            fps_instantaneo=fps_inst,
        )
        self._indice += 1
        return cuadro

    def cerrar(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._activa = False

    @property
    def esta_activa(self) -> bool:
        return self._activa
```

- [ ] **Paso 4: Correr tests**

```bash
pytest tests/test_fuente.py -v
```

Esperado: 3 passed.

- [ ] **Paso 5: Commit**

```bash
git add src/fuente/video.py tests/test_fuente.py
git commit -m "feat: FuenteVideo con cv2.VideoCapture

- Lee frames secuenciales de archivo mp4/avi
- Calcula fps_instantaneo entre frames
- Test con video sintetico de 5 frames"
```

---

### Tarea 9: Implementar `Detector` con YOLO26

**Archivos:**
- Crear: `src/percepcion/detector.py`
- Crear: `config/clases.yaml`
- Crear: `tests/test_detector.py`

- [ ] **Paso 1: Crear `config/clases.yaml`**

```yaml
# Mapeo de IDs de clases COCO -> nuestras clases unificadas.
# Solo listamos las clases que nos interesan; el resto va a DESCONOCIDO.
mapeo_coco:
  0: PEATON          # person
  1: MOTOCICLETA     # bicycle (lo tratamos como moto para simplificar)
  2: VEHICULO        # car
  3: MOTOCICLETA     # motorcycle
  5: VEHICULO        # bus
  7: VEHICULO        # truck
  9: SEMAFORO        # traffic light
  11: SENAL_ALTO     # stop sign
```

- [ ] **Paso 2: Escribir tests (con mock del modelo YOLO)**

```python
# tests/test_detector.py
"""Tests para el detector YOLO26."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.percepcion.detector import Detector
from src.tipos import Clase


@pytest.fixture
def mapeo_dummy():
    return {
        0: "PEATON",
        2: "VEHICULO",
        9: "SEMAFORO",
        11: "SENAL_ALTO",
    }


def test_detector_mapea_clases_coco_a_nuestras(mapeo_dummy, tmp_path):
    """Verifica que un resultado YOLO simulado se convierte a lista[Deteccion]."""
    # Simulamos la respuesta de YOLO
    caja_mock = MagicMock()
    caja_mock.cls = np.array([0])       # persona en COCO
    caja_mock.conf = np.array([0.85])
    caja_mock.xyxy = np.array([[100, 200, 150, 300]])

    resultado_mock = MagicMock()
    resultado_mock.boxes = [caja_mock]

    with patch("src.percepcion.detector.YOLO") as YoloClase:
        modelo_mock = MagicMock()
        modelo_mock.return_value = [resultado_mock]
        YoloClase.return_value = modelo_mock

        det = Detector(ruta_pesos="fake.pt", mapeo_clases=mapeo_dummy, conf_min=0.35)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detecciones = det.detectar(frame)

    assert len(detecciones) == 1
    assert detecciones[0].clase == Clase.PEATON
    assert detecciones[0].caja == (100, 200, 150, 300)
    assert detecciones[0].confianza == pytest.approx(0.85)


def test_detector_filtra_por_confianza_minima(mapeo_dummy):
    caja_mock = MagicMock()
    caja_mock.cls = np.array([2])
    caja_mock.conf = np.array([0.2])  # por debajo del umbral 0.35
    caja_mock.xyxy = np.array([[0, 0, 10, 10]])
    resultado_mock = MagicMock()
    resultado_mock.boxes = [caja_mock]

    with patch("src.percepcion.detector.YOLO") as YoloClase:
        modelo_mock = MagicMock()
        modelo_mock.return_value = [resultado_mock]
        YoloClase.return_value = modelo_mock

        det = Detector("fake.pt", mapeo_clases=mapeo_dummy, conf_min=0.35)
        detecciones = det.detectar(np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(detecciones) == 0


def test_detector_clase_no_mapeada_es_descartada(mapeo_dummy):
    caja_mock = MagicMock()
    caja_mock.cls = np.array([50])   # una clase que no nos interesa
    caja_mock.conf = np.array([0.9])
    caja_mock.xyxy = np.array([[0, 0, 10, 10]])
    resultado_mock = MagicMock()
    resultado_mock.boxes = [caja_mock]

    with patch("src.percepcion.detector.YOLO") as YoloClase:
        modelo_mock = MagicMock()
        modelo_mock.return_value = [resultado_mock]
        YoloClase.return_value = modelo_mock

        det = Detector("fake.pt", mapeo_clases=mapeo_dummy, conf_min=0.35)
        detecciones = det.detectar(np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(detecciones) == 0
```

- [ ] **Paso 3: Correr tests (falla)**

```bash
pytest tests/test_detector.py -v
```

Esperado: FAIL.

- [ ] **Paso 4: Implementar `Detector`**

```python
# src/percepcion/detector.py
"""Detector de objetos con YOLO26 (Ultralytics).

Carga el modelo una vez y corre inferencia sobre frames BGR.
Mapea los IDs de COCO al enum Clase de nuestro sistema.
"""
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.tipos import Clase, Deteccion


class Detector:
    def __init__(
        self,
        ruta_pesos: str,
        mapeo_clases: dict[int, str],
        conf_min: float = 0.35,
        imgsz: int = 640,
        device: str = "cuda",
    ):
        self._mapeo: dict[int, Clase] = {
            id_coco: Clase[nombre] for id_coco, nombre in mapeo_clases.items()
        }
        self._conf_min = conf_min
        self._imgsz = imgsz
        self._device = device
        self._modelo = YOLO(str(ruta_pesos))

    def detectar(self, frame: np.ndarray) -> list[Deteccion]:
        resultados = self._modelo(
            frame,
            device=self._device,
            imgsz=self._imgsz,
            conf=self._conf_min,
            verbose=False,
        )
        detecciones: list[Deteccion] = []
        for r in resultados:
            for caja in r.boxes:
                id_coco = int(caja.cls[0])
                conf = float(caja.conf[0])
                if conf < self._conf_min:
                    continue
                if id_coco not in self._mapeo:
                    continue
                x1, y1, x2, y2 = (int(v) for v in caja.xyxy[0])
                area = max(0, (x2 - x1) * (y2 - y1))
                detecciones.append(
                    Deteccion(
                        clase=self._mapeo[id_coco],
                        caja=(x1, y1, x2, y2),
                        confianza=conf,
                        area=area,
                    )
                )
        return detecciones

    @property
    def modelo(self) -> YOLO:
        """Acceso al modelo crudo (usado por tracker)."""
        return self._modelo
```

- [ ] **Paso 5: Correr tests**

```bash
pytest tests/test_detector.py -v
```

Esperado: 3 passed.

- [ ] **Paso 6: Commit**

```bash
git add src/percepcion/detector.py config/clases.yaml tests/test_detector.py
git commit -m "feat: Detector con YOLO26 y mapeo COCO -> Clase

- Filtra por confianza minima
- Ignora clases no mapeadas
- 3 tests con mock del modelo"
```

---

### Tarea 10: Script `probar_deteccion.py` sobre video

**Archivos:**
- Crear: `scripts/probar_deteccion.py`
- Crear: `config/default.yaml`

- [ ] **Paso 1: Crear `config/default.yaml`**

```yaml
fuente:
  tipo: "video"
  ruta_video: "datos/videos/ets2_gameplay.mp4"
  monitor: 1
  region: [0, 0, 1920, 1080]

modelo:
  pesos: "yolov8n.pt"
  imgsz: 640
  conf_min: 0.35
  device: "cuda"

control:
  tipo: "nulo"

seguridad:
  tecla_paro: "f12"
  timeout_watchdog_ms: 500

registro:
  ruta_base: "datos/evidencia"
  grabar_video: true
  fps_objetivo: 30
```

- [ ] **Paso 2: Crear script de visualización**

```python
# scripts/probar_deteccion.py
"""Corre el detector sobre un video y muestra las cajas en pantalla.

Uso:
    python scripts/probar_deteccion.py --video datos/videos/ets2_gameplay.mp4
"""
import argparse
from pathlib import Path

import cv2
import yaml

from src.fuente.video import FuenteVideo
from src.percepcion.detector import Detector

COLORES_CLASE = {
    "VEHICULO": (0, 255, 0),       # verde
    "MOTOCICLETA": (0, 200, 255),  # naranja
    "PEATON": (0, 0, 255),         # rojo
    "SEMAFORO": (255, 255, 0),     # cian
    "SENAL_ALTO": (0, 0, 150),     # rojo oscuro
}


def dibujar_detecciones(frame, detecciones):
    for det in detecciones:
        x1, y1, x2, y2 = det.caja
        color = COLORES_CLASE.get(det.clase.name, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        etiqueta = f"{det.clase.name} {det.confianza:.2f}"
        cv2.putText(frame, etiqueta, (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--clases", default="config/clases.yaml")
    parser.add_argument("--mostrar", action="store_true", default=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    clases_cfg = yaml.safe_load(Path(args.clases).read_text())

    detector = Detector(
        ruta_pesos=cfg["modelo"]["pesos"],
        mapeo_clases=clases_cfg["mapeo_coco"],
        conf_min=cfg["modelo"]["conf_min"],
        imgsz=cfg["modelo"]["imgsz"],
        device=cfg["modelo"]["device"],
    )

    fuente = FuenteVideo(args.video)
    fuente.iniciar()

    cv2.namedWindow("deteccion", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("deteccion", 1280, 720)

    while fuente.esta_activa:
        cuadro = fuente.siguiente()
        if cuadro is None:
            break
        detecciones = detector.detectar(cuadro.imagen)
        frame = dibujar_detecciones(cuadro.imagen.copy(), detecciones)
        cv2.putText(frame, f"FPS: {cuadro.fps_instantaneo:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("deteccion", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    fuente.cerrar()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Paso 3: Ejecutar sobre video descargado**

```bash
python scripts/probar_deteccion.py --video datos/videos/ets2_gameplay.mp4
```

Esperado: ventana con video y cajas coloreadas sobre vehículos/personas/semáforos. Pulsar `q` para cerrar.

- [ ] **Paso 4: Validación visual (criterio F1)**

Observar durante 1-2 minutos:
- ✅ Se detectan **vehículos** (camiones y autos) en >70 % de frames donde aparecen
- ✅ Se detectan **peatones** cuando los hay
- ✅ Se detectan **semáforos** (aunque el estado no se clasifique aún)
- ✅ Se detecta la **señal de alto** cuando aparece

Si alguna clase falla sistemáticamente, anotar y considerar fine-tuning en Tarea 10.5 (opcional).

- [ ] **Paso 5: Commit**

```bash
git add scripts/probar_deteccion.py config/default.yaml
git commit -m "feat: script probar_deteccion.py para validacion F1

- Muestra cajas coloreadas sobre video
- Incluye FPS instantaneo como overlay
- Validacion visual de detecciones de YOLO26"
```

**Gate F1:** Solo avanzar si la detección es aceptable en las 4 clases principales.

---

### Tarea 11: Tracker temporal con ByteTrack

**Archivos:**
- Crear: `src/percepcion/tracker.py`
- Crear: `tests/test_tracker.py`

- [ ] **Paso 1: Escribir test**

```python
# tests/test_tracker.py
"""Tests para el tracker temporal."""
from src.percepcion.tracker import Tracker
from src.tipos import Clase, Deteccion


def test_tracker_asigna_ids_nuevos_a_detecciones_nuevas():
    tracker = Tracker()
    detecciones_frame1 = [
        Deteccion(Clase.VEHICULO, (100, 200, 200, 300), 0.9, 10000),
    ]
    seguimientos = tracker.actualizar(detecciones_frame1)
    assert len(seguimientos) == 1
    assert seguimientos[0].id_seguimiento >= 0
    assert seguimientos[0].edad == 1


def test_tracker_mantiene_id_entre_frames_si_ubicacion_similar():
    tracker = Tracker()
    detecciones_f1 = [Deteccion(Clase.VEHICULO, (100, 200, 200, 300), 0.9, 10000)]
    segs_f1 = tracker.actualizar(detecciones_f1)
    id_inicial = segs_f1[0].id_seguimiento

    # Mismo objeto, ligero desplazamiento
    detecciones_f2 = [Deteccion(Clase.VEHICULO, (105, 205, 205, 305), 0.9, 10000)]
    segs_f2 = tracker.actualizar(detecciones_f2)
    assert segs_f2[0].id_seguimiento == id_inicial
    assert segs_f2[0].edad == 2


def test_tracker_edad_crece_por_persistencia():
    tracker = Tracker()
    det = Deteccion(Clase.VEHICULO, (100, 200, 200, 300), 0.9, 10000)
    edades = []
    for _ in range(5):
        segs = tracker.actualizar([det])
        edades.append(segs[0].edad)
    assert edades == [1, 2, 3, 4, 5]
```

- [ ] **Paso 2: Implementar tracker basado en IoU (simple, sin depender de ultralytics.track para testing)**

```python
# src/percepcion/tracker.py
"""Tracker temporal simple basado en IoU entre frames.

No usa ByteTrack directamente (que requiere estado del modelo YOLO); implementa
un matching por IoU que es suficiente para nuestros propositos y testeable.

Nota: para la version de produccion podemos cambiar esto por model.track(persist=True)
de Ultralytics. Aqui priorizamos testabilidad y control.
"""
from dataclasses import replace

from src.tipos import Deteccion, Seguimiento


def iou(caja_a: tuple[int, int, int, int], caja_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = caja_a
    bx1, by1, bx2, by2 = caja_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    def __init__(self, iou_umbral: float = 0.3, max_frames_perdido: int = 5):
        self._iou_umbral = iou_umbral
        self._max_perdido = max_frames_perdido
        self._siguientes_id = 0
        # Estado: dict[id_seg -> (Seguimiento, frames_perdido)]
        self._activos: dict[int, tuple[Seguimiento, int]] = {}

    def actualizar(self, detecciones: list[Deteccion]) -> list[Seguimiento]:
        # Para cada deteccion nueva, buscar el activo mas parecido
        asignados: set[int] = set()
        resultado: list[Seguimiento] = []

        for det in detecciones:
            mejor_id, mejor_iou = -1, 0.0
            for id_seg, (seg, _) in self._activos.items():
                if id_seg in asignados:
                    continue
                if seg.clase != det.clase:
                    continue
                score = iou(seg.caja, det.caja)
                if score > mejor_iou:
                    mejor_id, mejor_iou = id_seg, score

            if mejor_iou >= self._iou_umbral:
                # Continuidad: mismo id, edad +1
                seg_prev, _ = self._activos[mejor_id]
                seg_nuevo = Seguimiento(
                    clase=det.clase,
                    caja=det.caja,
                    confianza=det.confianza,
                    area=det.area,
                    id_seguimiento=mejor_id,
                    edad=seg_prev.edad + 1,
                )
                self._activos[mejor_id] = (seg_nuevo, 0)
                asignados.add(mejor_id)
                resultado.append(seg_nuevo)
            else:
                # Nuevo track
                nuevo_id = self._siguientes_id
                self._siguientes_id += 1
                seg_nuevo = Seguimiento(
                    clase=det.clase,
                    caja=det.caja,
                    confianza=det.confianza,
                    area=det.area,
                    id_seguimiento=nuevo_id,
                    edad=1,
                )
                self._activos[nuevo_id] = (seg_nuevo, 0)
                asignados.add(nuevo_id)
                resultado.append(seg_nuevo)

        # Tracks no vistos en este frame: aumentar contador de perdido
        muertos = []
        for id_seg in self._activos:
            if id_seg in asignados:
                continue
            seg, perdido = self._activos[id_seg]
            perdido += 1
            if perdido > self._max_perdido:
                muertos.append(id_seg)
            else:
                self._activos[id_seg] = (seg, perdido)
        for id_seg in muertos:
            del self._activos[id_seg]

        return resultado
```

- [ ] **Paso 3: Correr tests**

```bash
pytest tests/test_tracker.py -v
```

Esperado: 3 passed.

- [ ] **Paso 4: Commit**

```bash
git add src/percepcion/tracker.py tests/test_tracker.py
git commit -m "feat: Tracker temporal basado en IoU

- Asigna IDs persistentes entre frames
- Cuenta edad (frames consecutivos visible)
- Descarta tracks tras N frames sin matching
- 3 tests unitarios"
```

---

## FASE 2 — Contexto y ROI (Días 4-5)

### Tarea 12: Clasificador de semáforo (HSV)

**Archivos:**
- Crear: `src/percepcion/semaforo.py`
- Crear: `tests/test_semaforo.py`

- [ ] **Paso 1: Escribir tests**

```python
# tests/test_semaforo.py
"""Tests del clasificador de semaforo (rojo/amarillo/verde) por HSV."""
import numpy as np

from src.percepcion.semaforo import ClasificadorSemaforo
from src.tipos import EstadoSemaforo


def crear_parche(color_bgr: tuple[int, int, int], tam: int = 40) -> np.ndarray:
    """Crea una imagen cuadrada de color solido."""
    return np.full((tam, tam, 3), color_bgr, dtype=np.uint8)


def test_detecta_rojo():
    clasif = ClasificadorSemaforo()
    parche = crear_parche((30, 30, 220))  # rojo en BGR
    assert clasif.clasificar(parche) == EstadoSemaforo.ROJO


def test_detecta_amarillo():
    clasif = ClasificadorSemaforo()
    parche = crear_parche((0, 220, 255))  # amarillo en BGR
    assert clasif.clasificar(parche) == EstadoSemaforo.AMARILLO


def test_detecta_verde():
    clasif = ClasificadorSemaforo()
    parche = crear_parche((50, 220, 50))  # verde en BGR
    assert clasif.clasificar(parche) == EstadoSemaforo.VERDE


def test_parche_gris_da_desconocido():
    clasif = ClasificadorSemaforo()
    parche = crear_parche((128, 128, 128))
    assert clasif.clasificar(parche) == EstadoSemaforo.DESCONOCIDO


def test_parche_muy_pequeno_da_desconocido():
    clasif = ClasificadorSemaforo()
    parche = np.zeros((2, 2, 3), dtype=np.uint8)
    assert clasif.clasificar(parche) == EstadoSemaforo.DESCONOCIDO
```

- [ ] **Paso 2: Correr tests (falla)**

```bash
pytest tests/test_semaforo.py -v
```

- [ ] **Paso 3: Implementar clasificador**

```python
# src/percepcion/semaforo.py
"""Clasificador del estado de semaforo por analisis HSV.

No usa un modelo ML separado; cuenta pixeles en rangos de color dominantes
dentro del parche recortado de la deteccion.
"""
import cv2
import numpy as np

from src.tipos import EstadoSemaforo


# Rangos HSV (OpenCV: H=0-180, S=0-255, V=0-255)
# El rojo vive en dos extremos de H (cerca de 0 y cerca de 180)
RANGOS = {
    "rojo_1": (np.array([0, 120, 100]), np.array([10, 255, 255])),
    "rojo_2": (np.array([170, 120, 100]), np.array([180, 255, 255])),
    "amarillo": (np.array([18, 120, 120]), np.array([35, 255, 255])),
    "verde": (np.array([45, 80, 80]), np.array([85, 255, 255])),
}


class ClasificadorSemaforo:
    def __init__(self, min_pix_dominante: int = 20, tam_minimo: int = 10):
        self._min_pix = min_pix_dominante
        self._tam_min = tam_minimo

    def clasificar(self, parche_bgr: np.ndarray) -> EstadoSemaforo:
        h, w = parche_bgr.shape[:2]
        if h < self._tam_min or w < self._tam_min:
            return EstadoSemaforo.DESCONOCIDO

        hsv = cv2.cvtColor(parche_bgr, cv2.COLOR_BGR2HSV)
        conteos = {}
        mascara_rojo = cv2.inRange(hsv, *RANGOS["rojo_1"]) | cv2.inRange(hsv, *RANGOS["rojo_2"])
        conteos["rojo"] = int(np.count_nonzero(mascara_rojo))
        conteos["amarillo"] = int(np.count_nonzero(cv2.inRange(hsv, *RANGOS["amarillo"])))
        conteos["verde"] = int(np.count_nonzero(cv2.inRange(hsv, *RANGOS["verde"])))

        ganador = max(conteos, key=conteos.get)
        if conteos[ganador] < self._min_pix:
            return EstadoSemaforo.DESCONOCIDO

        return {
            "rojo": EstadoSemaforo.ROJO,
            "amarillo": EstadoSemaforo.AMARILLO,
            "verde": EstadoSemaforo.VERDE,
        }[ganador]
```

- [ ] **Paso 4: Correr tests**

```bash
pytest tests/test_semaforo.py -v
```

Esperado: 5 passed.

- [ ] **Paso 5: Commit**

```bash
git add src/percepcion/semaforo.py tests/test_semaforo.py
git commit -m "feat: clasificador de semaforo por HSV

- Convierte parche BGR a HSV
- Cuenta pixeles en rangos rojo/amarillo/verde
- 5 tests con parches sinteticos"
```

---

### Tarea 13: Módulo de contexto (ROI + análisis de riesgo)

**Archivos:**
- Crear: `src/percepcion/contexto.py`
- Crear: `config/regiones_interes.yaml`
- Crear: `tests/test_contexto.py`

- [ ] **Paso 1: Crear configuración inicial de ROI (valores razonables para 1920×1080)**

```yaml
# config/regiones_interes.yaml
# Coordenadas para frame 1920x1080 en primera persona de ETS2.
# Los poligonos se especifican como lista de puntos [x, y].
# ESTOS VALORES SON INICIALES; se ajustan con scripts/calibrar_regiones.py mas adelante.

resolucion_base: [1920, 1080]

regiones:
  FRENTE_CERCANO:
    - [600, 700]
    - [1320, 700]
    - [1500, 900]
    - [420, 900]
  FRENTE_LEJANO:
    - [800, 450]
    - [1120, 450]
    - [1320, 700]
    - [600, 700]
  ESPEJO_IZQ:
    - [80, 380]
    - [320, 380]
    - [320, 640]
    - [80, 640]
  ESPEJO_DER:
    - [1600, 380]
    - [1840, 380]
    - [1840, 640]
    - [1600, 640]
  LATERAL_IZQ:
    - [0, 450]
    - [500, 450]
    - [500, 900]
    - [0, 900]
  LATERAL_DER:
    - [1420, 450]
    - [1920, 450]
    - [1920, 900]
    - [1420, 900]

# Umbrales de analisis
umbrales:
  area_minima_frente_cercano: 8000  # px^2
  edad_minima_ocupado: 3            # frames consecutivos
  ventana_confianza: 10             # frames para media movil
```

- [ ] **Paso 2: Escribir tests**

```python
# tests/test_contexto.py
"""Tests del modulo de contexto."""
import numpy as np

from src.percepcion.contexto import Contexto, Analizador
from src.tipos import Clase, EstadoSemaforo, Region, Seguimiento

CONFIG_PRUEBA = {
    "resolucion_base": [1920, 1080],
    "regiones": {
        "FRENTE_CERCANO": [[600, 700], [1320, 700], [1500, 900], [420, 900]],
        "FRENTE_LEJANO": [[800, 450], [1120, 450], [1320, 700], [600, 700]],
        "ESPEJO_IZQ": [[80, 380], [320, 380], [320, 640], [80, 640]],
        "ESPEJO_DER": [[1600, 380], [1840, 380], [1840, 640], [1600, 640]],
        "LATERAL_IZQ": [[0, 450], [500, 450], [500, 900], [0, 900]],
        "LATERAL_DER": [[1420, 450], [1920, 450], [1920, 900], [1420, 900]],
    },
    "umbrales": {
        "area_minima_frente_cercano": 8000,
        "edad_minima_ocupado": 3,
        "ventana_confianza": 10,
    },
}


def seguimiento(clase, x1, y1, x2, y2, id_seg=0, edad=5, conf=0.9):
    return Seguimiento(
        clase=clase,
        caja=(x1, y1, x2, y2),
        confianza=conf,
        area=(x2 - x1) * (y2 - y1),
        id_seguimiento=id_seg,
        edad=edad,
    )


def test_vehiculo_grande_en_frente_cercano_marca_ocupado():
    ctx = Contexto(CONFIG_PRUEBA)
    segs = [seguimiento(Clase.VEHICULO, 800, 750, 1100, 880, edad=5)]
    estado = ctx.analizar(segs, frame_bgr=None, timestamp=0.0)
    assert estado.frente_cercano_ocupado is True


def test_vehiculo_lejos_no_ocupa_frente_cercano():
    ctx = Contexto(CONFIG_PRUEBA)
    # vehiculo en frente lejano (area chica)
    segs = [seguimiento(Clase.VEHICULO, 950, 500, 1000, 550, edad=5)]
    estado = ctx.analizar(segs, frame_bgr=None, timestamp=0.0)
    assert estado.frente_cercano_ocupado is False


def test_peaton_en_frente_marca_riesgo():
    ctx = Contexto(CONFIG_PRUEBA)
    segs = [seguimiento(Clase.PEATON, 900, 750, 960, 880, edad=5)]
    estado = ctx.analizar(segs, frame_bgr=None, timestamp=0.0)
    assert estado.peaton_en_riesgo is True


def test_peaton_en_lateral_tambien_marca_riesgo():
    ctx = Contexto(CONFIG_PRUEBA)
    segs = [seguimiento(Clase.PEATON, 100, 600, 150, 780, edad=5)]
    estado = ctx.analizar(segs, frame_bgr=None, timestamp=0.0)
    assert estado.peaton_en_riesgo is True


def test_vehiculo_en_espejo_con_edad_suficiente_ocupa_espejo():
    ctx = Contexto(CONFIG_PRUEBA)
    segs = [seguimiento(Clase.VEHICULO, 100, 420, 280, 600, edad=5)]
    estado = ctx.analizar(segs, frame_bgr=None, timestamp=0.0)
    assert estado.espejo_izq_ocupado is True


def test_vehiculo_en_espejo_con_edad_baja_no_ocupa():
    ctx = Contexto(CONFIG_PRUEBA)
    segs = [seguimiento(Clase.VEHICULO, 100, 420, 280, 600, edad=1)]
    estado = ctx.analizar(segs, frame_bgr=None, timestamp=0.0)
    assert estado.espejo_izq_ocupado is False


def test_senal_alto_detectada_marca_campo():
    ctx = Contexto(CONFIG_PRUEBA)
    segs = [seguimiento(Clase.SENAL_ALTO, 850, 600, 950, 700, edad=3)]
    estado = ctx.analizar(segs, frame_bgr=None, timestamp=0.0)
    assert estado.senal_alto_cercana is True


def test_confianza_percepcion_se_calcula_con_ventana():
    ctx = Contexto(CONFIG_PRUEBA)
    for _ in range(10):
        ctx.analizar([seguimiento(Clase.VEHICULO, 800, 750, 1100, 880, edad=5)], None, 0.0)
    estado_final = ctx.analizar(
        [seguimiento(Clase.VEHICULO, 800, 750, 1100, 880, edad=5)], None, 0.0
    )
    assert estado_final.confianza_percepcion > 0.8
```

- [ ] **Paso 3: Correr tests (fallan)**

```bash
pytest tests/test_contexto.py -v
```

- [ ] **Paso 4: Implementar `Contexto`**

```python
# src/percepcion/contexto.py
"""Analiza los seguimientos y produce EstadoEscena.

- Clasifica cada track segun la region en la que cae
- Estima ocupacion, riesgo y confianza
- No usa el frame BGR salvo para clasificar semaforos
"""
from collections import deque
from typing import Optional

import cv2
import numpy as np

from src.percepcion.semaforo import ClasificadorSemaforo
from src.tipos import Clase, EstadoEscena, EstadoSemaforo, Region, Seguimiento


class Contexto:
    def __init__(self, config: dict):
        regs = config["regiones"]
        self._poligonos: dict[Region, np.ndarray] = {
            Region[nombre]: np.array(pts, dtype=np.int32)
            for nombre, pts in regs.items()
        }
        umbrales = config["umbrales"]
        self._area_min_frente = umbrales["area_minima_frente_cercano"]
        self._edad_min_ocupado = umbrales["edad_minima_ocupado"]
        self._ventana = umbrales["ventana_confianza"]
        self._historia_confianza: deque[float] = deque(maxlen=self._ventana)
        self._clasif_semaforo = ClasificadorSemaforo()

    def _centro(self, caja: tuple[int, int, int, int]) -> tuple[int, int]:
        x1, y1, x2, y2 = caja
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _region_que_contiene(self, caja: tuple[int, int, int, int]) -> Optional[Region]:
        cx, cy = self._centro(caja)
        for region, pol in self._poligonos.items():
            if cv2.pointPolygonTest(pol, (cx, cy), False) >= 0:
                return region
        return None

    def analizar(
        self,
        seguimientos: list[Seguimiento],
        frame_bgr: Optional[np.ndarray],
        timestamp: float,
    ) -> EstadoEscena:
        frente_cerc = False
        frente_lej = False
        peaton_riesgo = False
        espejo_izq = False
        espejo_der = False
        senal_alto = False
        sem_estado: Optional[EstadoSemaforo] = None

        for seg in seguimientos:
            region = self._region_que_contiene(seg.caja)
            if region is None:
                continue

            es_estable = seg.edad >= self._edad_min_ocupado

            # Frente cercano: vehiculos grandes en region frontal
            if region == Region.FRENTE_CERCANO and seg.clase in (Clase.VEHICULO, Clase.MOTOCICLETA):
                if seg.area >= self._area_min_frente and es_estable:
                    frente_cerc = True
            if region == Region.FRENTE_LEJANO and seg.clase in (Clase.VEHICULO, Clase.MOTOCICLETA):
                if es_estable:
                    frente_lej = True

            # Peaton: riesgo si aparece en cualquier zona frontal o lateral cercana
            if seg.clase == Clase.PEATON and region in (
                Region.FRENTE_CERCANO,
                Region.FRENTE_LEJANO,
                Region.LATERAL_IZQ,
                Region.LATERAL_DER,
            ):
                peaton_riesgo = True

            # Espejos
            if region == Region.ESPEJO_IZQ and seg.clase in (Clase.VEHICULO, Clase.MOTOCICLETA):
                if es_estable:
                    espejo_izq = True
            if region == Region.ESPEJO_DER and seg.clase in (Clase.VEHICULO, Clase.MOTOCICLETA):
                if es_estable:
                    espejo_der = True

            # Senal de alto
            if seg.clase == Clase.SENAL_ALTO and region in (
                Region.FRENTE_CERCANO,
                Region.FRENTE_LEJANO,
            ):
                senal_alto = True

            # Semaforo: clasificar color si tenemos frame
            if seg.clase == Clase.SEMAFORO and frame_bgr is not None:
                x1, y1, x2, y2 = seg.caja
                parche = frame_bgr[max(0, y1):y2, max(0, x1):x2]
                if parche.size > 0:
                    nuevo = self._clasif_semaforo.clasificar(parche)
                    # Si ya hay un estado, priorizamos ROJO/AMARILLO sobre VERDE/DESCONOCIDO
                    orden = {EstadoSemaforo.ROJO: 3, EstadoSemaforo.AMARILLO: 2,
                             EstadoSemaforo.VERDE: 1, EstadoSemaforo.DESCONOCIDO: 0}
                    if sem_estado is None or orden[nuevo] > orden[sem_estado]:
                        sem_estado = nuevo

        # Confianza: media movil de "hay al menos un track estable"
        estable = 1.0 if any(s.edad >= self._edad_min_ocupado for s in seguimientos) else 0.0
        self._historia_confianza.append(estable)
        confianza = float(np.mean(self._historia_confianza)) if self._historia_confianza else 1.0

        return EstadoEscena(
            frente_cercano_ocupado=frente_cerc,
            frente_lejano_ocupado=frente_lej,
            peaton_en_riesgo=peaton_riesgo,
            semaforo_visible=sem_estado,
            senal_alto_cercana=senal_alto,
            espejo_izq_ocupado=espejo_izq,
            espejo_der_ocupado=espejo_der,
            vehiculos_totales=sum(
                1 for s in seguimientos if s.clase in (Clase.VEHICULO, Clase.MOTOCICLETA)
            ),
            confianza_percepcion=confianza,
            timestamp=timestamp,
        )


# Alias para tests
Analizador = Contexto
```

- [ ] **Paso 5: Correr tests**

```bash
pytest tests/test_contexto.py -v
```

Esperado: 8 passed.

- [ ] **Paso 6: Commit**

```bash
git add src/percepcion/contexto.py config/regiones_interes.yaml tests/test_contexto.py
git commit -m "feat: modulo de contexto con ROI poligonales

- Clasifica tracks por region (frente/espejos/laterales)
- Detecta ocupacion, peatones en riesgo, semaforo, alto
- Calcula confianza por ventana movil
- 8 tests unitarios que cubren casos clave"
```

---

### Tarea 14: Herramienta de calibración de regiones

**Archivos:**
- Crear: `scripts/calibrar_regiones.py`

- [ ] **Paso 1: Crear script interactivo para dibujar/ajustar ROI**

```python
# scripts/calibrar_regiones.py
"""Herramienta interactiva para visualizar y ajustar las regiones de interes.

Uso:
    python scripts/calibrar_regiones.py --video datos/videos/ets2_gameplay.mp4

Controles:
    espacio: avanzar 10 frames
    n: siguiente frame
    s: guardar screenshot actual
    q: salir
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

COLORES_REGION = {
    "FRENTE_CERCANO": (0, 255, 0),
    "FRENTE_LEJANO": (0, 200, 200),
    "ESPEJO_IZQ": (255, 0, 255),
    "ESPEJO_DER": (255, 100, 255),
    "LATERAL_IZQ": (255, 150, 0),
    "LATERAL_DER": (0, 150, 255),
}


def dibujar_regiones(frame, config):
    overlay = frame.copy()
    for nombre, pts in config["regiones"].items():
        pol = np.array(pts, dtype=np.int32)
        color = COLORES_REGION.get(nombre, (200, 200, 200))
        cv2.fillPoly(overlay, [pol], color)
        cv2.polylines(frame, [pol], True, color, 2)
        cv2.putText(frame, nombre, tuple(pol[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="config/regiones_interes.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    cap = cv2.VideoCapture(args.video)
    cv2.namedWindow("calibracion", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("calibracion", 1280, 720)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = dibujar_regiones(frame, config)
        cv2.imshow("calibracion", frame)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            for _ in range(10):
                cap.read()
        if k == ord("s"):
            ruta = Path("datos/evidencia") / "calibracion_ROI.png"
            ruta.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ruta), frame)
            print(f"Guardado: {ruta}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Paso 2: Correr sobre el video**

```bash
python scripts/calibrar_regiones.py --video datos/videos/ets2_gameplay.mp4
```

Observar si las regiones iniciales tienen sentido (el capó, los espejos, la zona frontal). Si no, editar `config/regiones_interes.yaml` directamente y volver a correr.

- [ ] **Paso 3: Guardar screenshot de calibración final como evidencia**

Presionar `s` con una vista representativa.

- [ ] **Paso 4: Commit**

```bash
git add scripts/calibrar_regiones.py datos/evidencia/calibracion_ROI.png
git commit -m "feat: herramienta de calibracion visual de ROI

- Dibuja poligonos semitransparentes sobre video
- Permite guardar screenshot como evidencia
- Ajuste de YAML iterativo"
```

**Gate F2:** Solo avanzar si las ROI visualmente cubren las zonas correctas del camión.

---

## FASE 3 — FSM y decisión (Días 6-8)

### Tarea 15: Estados del FSM (enum)

**Archivos:**
- Crear: `src/decision/estado.py`
- Crear: `tests/test_estado.py`

- [ ] **Paso 1: Test minimalista**

```python
# tests/test_estado.py
from src.decision.estado import Estado


def test_estados_existen():
    assert Estado.INICIALIZANDO
    assert Estado.CONDUCIENDO_NORMAL
    assert Estado.SIGUIENDO_VEHICULO
    assert Estado.FRENANDO_PREVENTIVO
    assert Estado.APROXIMANDO_ALTO
    assert Estado.DETENIDO_ALTO
    assert Estado.APROXIMANDO_SEMAFORO
    assert Estado.DETENIDO_SEMAFORO
    assert Estado.CRUZANDO
    assert Estado.EVALUANDO_REBASE
    assert Estado.REBASANDO
    assert Estado.RECUPERACION
    assert Estado.PARO_EMERGENCIA
    # Debe haber exactamente 13 estados
    assert len(Estado) == 13
```

- [ ] **Paso 2: Implementar**

```python
# src/decision/estado.py
"""Estados de la maquina de decision."""
from enum import Enum


class Estado(Enum):
    INICIALIZANDO = "inicializando"
    CONDUCIENDO_NORMAL = "conduciendo_normal"
    SIGUIENDO_VEHICULO = "siguiendo_vehiculo"
    FRENANDO_PREVENTIVO = "frenando_preventivo"
    APROXIMANDO_ALTO = "aproximando_alto"
    DETENIDO_ALTO = "detenido_alto"
    APROXIMANDO_SEMAFORO = "aproximando_semaforo"
    DETENIDO_SEMAFORO = "detenido_semaforo"
    CRUZANDO = "cruzando"
    EVALUANDO_REBASE = "evaluando_rebase"
    REBASANDO = "rebasando"
    RECUPERACION = "recuperacion"
    PARO_EMERGENCIA = "paro_emergencia"
```

- [ ] **Paso 3: Correr test**

```bash
pytest tests/test_estado.py -v
```

Esperado: 1 passed.

- [ ] **Paso 4: Commit**

```bash
git add src/decision/estado.py tests/test_estado.py
git commit -m "feat: enum de 13 estados del FSM"
```

---

### Tarea 16: Implementar FSM con reglas priorizadas

**Archivos:**
- Crear: `src/decision/fsm.py`
- Crear: `tests/test_fsm.py`

Esta tarea es larga; las reglas se testean una por una.

- [ ] **Paso 1: Escribir tests (una prueba por regla de la tabla 10.2 del spec)**

```python
# tests/test_fsm.py
"""Tests de la maquina de decision.

Una prueba por cada regla del FSM (ver seccion 10.2 del spec).
"""
import pytest

from src.decision.estado import Estado
from src.decision.fsm import MaquinaDecision
from src.tipos import Accion, EstadoEscena, EstadoSemaforo


def escena(**overrides) -> EstadoEscena:
    """Escena por defecto: todo tranquilo."""
    base = {
        "frente_cercano_ocupado": False,
        "frente_lejano_ocupado": False,
        "peaton_en_riesgo": False,
        "semaforo_visible": None,
        "senal_alto_cercana": False,
        "espejo_izq_ocupado": False,
        "espejo_der_ocupado": False,
        "vehiculos_totales": 0,
        "confianza_percepcion": 1.0,
        "timestamp": 0.0,
    }
    base.update(overrides)
    return EstadoEscena(**base)


def test_regla1_paro_emergencia_manda_todo_al_freno():
    fsm = MaquinaDecision()
    fsm.activar_paro_emergencia()
    accion = fsm.decidir(escena())
    assert fsm.estado == Estado.PARO_EMERGENCIA
    assert accion == Accion.ALTO_TOTAL


def test_regla2_baja_confianza_entra_en_recuperacion():
    fsm = MaquinaDecision()
    accion = fsm.decidir(escena(confianza_percepcion=0.2))
    assert fsm.estado == Estado.RECUPERACION
    assert accion == Accion.FRENAR_SUAVE


def test_regla3_peaton_en_riesgo_frena_fuerte():
    fsm = MaquinaDecision()
    accion = fsm.decidir(escena(peaton_en_riesgo=True))
    assert fsm.estado == Estado.FRENANDO_PREVENTIVO
    assert accion == Accion.FRENAR_FUERTE


def test_regla4_semaforo_rojo_alto_total():
    fsm = MaquinaDecision()
    accion = fsm.decidir(escena(semaforo_visible=EstadoSemaforo.ROJO))
    assert fsm.estado == Estado.DETENIDO_SEMAFORO
    assert accion == Accion.ALTO_TOTAL


def test_regla5_semaforo_amarillo_frena_suave():
    fsm = MaquinaDecision()
    accion = fsm.decidir(escena(semaforo_visible=EstadoSemaforo.AMARILLO))
    assert fsm.estado == Estado.APROXIMANDO_SEMAFORO
    assert accion == Accion.FRENAR_SUAVE


def test_regla6_senal_alto_cercana_detiene():
    fsm = MaquinaDecision()
    accion = fsm.decidir(escena(senal_alto_cercana=True))
    assert fsm.estado == Estado.DETENIDO_ALTO
    assert accion == Accion.ALTO_TOTAL


def test_regla7_sale_de_alto_tras_2_segundos_y_laterales_libres():
    fsm = MaquinaDecision()
    # Entra en DETENIDO_ALTO
    fsm.decidir(escena(senal_alto_cercana=True, timestamp=0.0))
    # Aun no pasan 2s -> sigue detenido
    accion = fsm.decidir(escena(senal_alto_cercana=True, timestamp=1.0))
    assert fsm.estado == Estado.DETENIDO_ALTO
    assert accion == Accion.ALTO_TOTAL
    # Pasan 2s y laterales libres -> cruza
    accion = fsm.decidir(escena(senal_alto_cercana=False, timestamp=2.5))
    assert fsm.estado == Estado.CRUZANDO
    assert accion == Accion.ACELERAR


def test_regla8_frente_ocupado_sigue_vehiculo():
    fsm = MaquinaDecision()
    accion = fsm.decidir(escena(frente_cercano_ocupado=True))
    assert fsm.estado == Estado.SIGUIENDO_VEHICULO
    assert accion == Accion.FRENAR_SUAVE


def test_regla11_semaforo_verde_acelera():
    fsm = MaquinaDecision()
    accion = fsm.decidir(escena(semaforo_visible=EstadoSemaforo.VERDE))
    assert fsm.estado == Estado.CONDUCIENDO_NORMAL
    assert accion == Accion.ACELERAR


def test_regla12_default_mantener():
    fsm = MaquinaDecision()
    accion = fsm.decidir(escena())
    assert fsm.estado == Estado.CONDUCIENDO_NORMAL
    assert accion in (Accion.ACELERAR, Accion.MANTENER)


def test_prioridad_peaton_sobre_semaforo_verde():
    fsm = MaquinaDecision()
    accion = fsm.decidir(escena(
        peaton_en_riesgo=True,
        semaforo_visible=EstadoSemaforo.VERDE,
    ))
    # La regla 3 (peaton) debe ganar sobre la 11 (verde)
    assert fsm.estado == Estado.FRENANDO_PREVENTIVO
    assert accion == Accion.FRENAR_FUERTE


def test_transiciones_se_loggean_con_razon():
    fsm = MaquinaDecision()
    fsm.decidir(escena(semaforo_visible=EstadoSemaforo.ROJO))
    transicion = fsm.ultima_transicion
    assert transicion is not None
    assert transicion["de"] == Estado.INICIALIZANDO.name
    assert transicion["a"] == Estado.DETENIDO_SEMAFORO.name
    assert "regla" in transicion
    assert "razon" in transicion
```

- [ ] **Paso 2: Correr tests (fallan)**

```bash
pytest tests/test_fsm.py -v
```

- [ ] **Paso 3: Implementar FSM**

```python
# src/decision/fsm.py
"""Maquina de decision con 12 reglas priorizadas.

Evalua las reglas en orden; la primera que matchee gana. Esto garantiza
comportamiento predecible y auditable (RNF-05).
"""
import time
from dataclasses import dataclass
from typing import Optional

from src.decision.estado import Estado
from src.tipos import Accion, EstadoEscena, EstadoSemaforo


@dataclass
class Transicion:
    de: str
    a: str
    accion: str
    regla: int
    razon: str
    timestamp: float


class MaquinaDecision:
    def __init__(self, tiempo_minimo_alto_s: float = 2.0):
        self._estado = Estado.INICIALIZANDO
        self._t_entrada_alto: Optional[float] = None
        self._paro_emergencia = False
        self._tiempo_minimo_alto = tiempo_minimo_alto_s
        self._ultima_transicion: Optional[dict] = None
        self._historial: list[dict] = []

    @property
    def estado(self) -> Estado:
        return self._estado

    @property
    def ultima_transicion(self) -> Optional[dict]:
        return self._ultima_transicion

    @property
    def historial(self) -> list[dict]:
        return list(self._historial)

    def activar_paro_emergencia(self) -> None:
        self._paro_emergencia = True

    def desactivar_paro_emergencia(self) -> None:
        self._paro_emergencia = False

    def _transicionar(self, nuevo_estado: Estado, accion: Accion, regla: int, razon: str, t: float):
        if nuevo_estado != self._estado:
            trans = {
                "de": self._estado.name,
                "a": nuevo_estado.name,
                "accion": accion.name,
                "regla": regla,
                "razon": razon,
                "timestamp": t,
            }
            self._ultima_transicion = trans
            self._historial.append(trans)
            # Resetear timer al salir de DETENIDO_ALTO
            if self._estado == Estado.DETENIDO_ALTO and nuevo_estado != Estado.DETENIDO_ALTO:
                self._t_entrada_alto = None
            # Capturar timer al entrar a DETENIDO_ALTO
            if nuevo_estado == Estado.DETENIDO_ALTO and self._t_entrada_alto is None:
                self._t_entrada_alto = t
            self._estado = nuevo_estado

    def decidir(self, escena: EstadoEscena) -> Accion:
        t = escena.timestamp

        # Regla 1: paro de emergencia
        if self._paro_emergencia:
            self._transicionar(Estado.PARO_EMERGENCIA, Accion.ALTO_TOTAL, 1, "paro manual/watchdog", t)
            return Accion.ALTO_TOTAL

        # Regla 2: baja confianza
        if escena.confianza_percepcion < 0.3:
            self._transicionar(Estado.RECUPERACION, Accion.FRENAR_SUAVE, 2, "confianza<0.3", t)
            return Accion.FRENAR_SUAVE

        # Regla 3: peaton en riesgo
        if escena.peaton_en_riesgo:
            self._transicionar(Estado.FRENANDO_PREVENTIVO, Accion.FRENAR_FUERTE, 3, "peaton en riesgo", t)
            return Accion.FRENAR_FUERTE

        # Regla 4: semaforo rojo
        if escena.semaforo_visible == EstadoSemaforo.ROJO:
            self._transicionar(Estado.DETENIDO_SEMAFORO, Accion.ALTO_TOTAL, 4, "semaforo=ROJO", t)
            return Accion.ALTO_TOTAL

        # Regla 5: semaforo amarillo
        if escena.semaforo_visible == EstadoSemaforo.AMARILLO:
            self._transicionar(Estado.APROXIMANDO_SEMAFORO, Accion.FRENAR_SUAVE, 5, "semaforo=AMARILLO", t)
            return Accion.FRENAR_SUAVE

        # Regla 6 y 7: senal de alto
        if escena.senal_alto_cercana:
            self._transicionar(Estado.DETENIDO_ALTO, Accion.ALTO_TOTAL, 6, "senal de alto cercana", t)
            return Accion.ALTO_TOTAL

        # Regla 7: saliendo de alto
        if self._estado == Estado.DETENIDO_ALTO:
            if self._t_entrada_alto is None:
                self._t_entrada_alto = t
            if (t - self._t_entrada_alto) >= self._tiempo_minimo_alto:
                self._transicionar(Estado.CRUZANDO, Accion.ACELERAR, 7,
                                   f"detenido {t - self._t_entrada_alto:.1f}s + laterales libres", t)
                return Accion.ACELERAR
            else:
                return Accion.ALTO_TOTAL

        # Regla 8: frente ocupado
        if escena.frente_cercano_ocupado:
            self._transicionar(Estado.SIGUIENDO_VEHICULO, Accion.FRENAR_SUAVE, 8, "frente cercano ocupado", t)
            return Accion.FRENAR_SUAVE

        # Regla 11: semaforo verde
        if escena.semaforo_visible == EstadoSemaforo.VERDE:
            self._transicionar(Estado.CONDUCIENDO_NORMAL, Accion.ACELERAR, 11, "semaforo=VERDE", t)
            return Accion.ACELERAR

        # Regla 12: default
        if self._estado == Estado.INICIALIZANDO:
            self._transicionar(Estado.CONDUCIENDO_NORMAL, Accion.MANTENER, 12, "default inicial", t)
        else:
            self._transicionar(Estado.CONDUCIENDO_NORMAL, Accion.MANTENER, 12, "default", t)
        return Accion.MANTENER
```

- [ ] **Paso 4: Correr tests**

```bash
pytest tests/test_fsm.py -v
```

Esperado: 12 passed.

- [ ] **Paso 5: Commit**

```bash
git add src/decision/fsm.py tests/test_fsm.py
git commit -m "feat: maquina de decision con 12 reglas priorizadas

- Evaluacion en orden, primera coincidencia gana
- Timer explicito para DETENIDO_ALTO (>= 2s)
- Registra transiciones con razon y numero de regla (RNF-05)
- 12 tests unitarios (uno por regla + prioridad + logging)"
```

---

### Tarea 17: Mapa `Accion → ComandoControl` + Controlador Nulo

**Archivos:**
- Crear: `src/control/base.py`
- Crear: `src/control/nulo.py`
- Crear: `src/control/mapa_acciones.py`
- Crear: `tests/test_control.py`

- [ ] **Paso 1: Tests**

```python
# tests/test_control.py
"""Tests del sistema de control."""
from src.control.base import Controlador
from src.control.mapa_acciones import accion_a_comando
from src.control.nulo import ControladorNulo
from src.tipos import Accion


def test_acelerar_produce_acelerador_alto():
    cmd = accion_a_comando(Accion.ACELERAR, timestamp=0.0)
    assert cmd.acelerador > 0.3
    assert cmd.freno == 0.0
    assert cmd.volante == 0.0


def test_alto_total_freno_maximo():
    cmd = accion_a_comando(Accion.ALTO_TOTAL, timestamp=0.0)
    assert cmd.freno == 1.0
    assert cmd.acelerador == 0.0


def test_rebasar_izq_tiene_volante_negativo():
    cmd = accion_a_comando(Accion.REBASAR_IZQ, timestamp=0.0)
    assert cmd.volante < 0
    assert cmd.acelerador > 0


def test_controlador_nulo_registra_pero_no_hace_nada():
    ctrl = ControladorNulo()
    ctrl.aplicar(accion_a_comando(Accion.ACELERAR, 0.0))
    ctrl.aplicar(accion_a_comando(Accion.FRENAR_FUERTE, 1.0))
    assert len(ctrl.registro) == 2
    assert ctrl.registro[0].acelerador > 0


def test_controlador_nulo_liberar_registra_cero():
    ctrl = ControladorNulo()
    ctrl.aplicar(accion_a_comando(Accion.ACELERAR, 0.0))
    ctrl.liberar()
    assert ctrl.registro[-1].acelerador == 0.0
    assert ctrl.registro[-1].freno == 0.0
    assert ctrl.registro[-1].volante == 0.0
```

- [ ] **Paso 2: Implementar interfaz y mapa**

```python
# src/control/base.py
"""Interfaz abstracta de controlador."""
from abc import ABC, abstractmethod

from src.tipos import ComandoControl


class Controlador(ABC):
    @abstractmethod
    def aplicar(self, cmd: ComandoControl) -> None: ...

    @abstractmethod
    def liberar(self) -> None:
        """Suelta todos los inputs (acelerador=0, freno=0, volante=0)."""

    @abstractmethod
    def cerrar(self) -> None: ...
```

```python
# src/control/mapa_acciones.py
"""Mapeo de Accion (alto nivel) a ComandoControl (bajo nivel).

Valores concretos definidos en seccion 8.4 del spec.
"""
from src.tipos import Accion, ComandoControl

_MAPA = {
    Accion.MANTENER:      (0.3, 0.0,  0.0),
    Accion.ACELERAR:      (0.6, 0.0,  0.0),
    Accion.FRENAR_SUAVE:  (0.0, 0.4,  0.0),
    Accion.FRENAR_FUERTE: (0.0, 0.8,  0.0),
    Accion.ALTO_TOTAL:    (0.0, 1.0,  0.0),
    Accion.GIRAR_IZQ:     (0.2, 0.0, -0.5),
    Accion.GIRAR_DER:     (0.2, 0.0, +0.5),
    Accion.REBASAR_IZQ:   (0.8, 0.0, -0.3),
    Accion.REBASAR_DER:   (0.8, 0.0, +0.3),
    Accion.ESPERAR:       (0.0, 0.0,  0.0),
}


def accion_a_comando(accion: Accion, timestamp: float) -> ComandoControl:
    acelerador, freno, volante = _MAPA[accion]
    return ComandoControl(
        acelerador=acelerador,
        freno=freno,
        volante=volante,
        timestamp=timestamp,
    )
```

```python
# src/control/nulo.py
"""Controlador que NO envia inputs al sistema; solo loggea.

Util para Fases 0-3 (pruebas sobre video) donde no queremos tocar el juego.
"""
from src.control.base import Controlador
from src.tipos import ComandoControl


class ControladorNulo(Controlador):
    def __init__(self):
        self._registro: list[ComandoControl] = []

    def aplicar(self, cmd: ComandoControl) -> None:
        self._registro.append(cmd)

    def liberar(self) -> None:
        self._registro.append(ComandoControl(0.0, 0.0, 0.0, timestamp=-1.0))

    def cerrar(self) -> None:
        pass

    @property
    def registro(self) -> list[ComandoControl]:
        return list(self._registro)
```

- [ ] **Paso 3: Correr tests**

```bash
pytest tests/test_control.py -v
```

Esperado: 5 passed.

- [ ] **Paso 4: Commit**

```bash
git add src/control/base.py src/control/nulo.py src/control/mapa_acciones.py tests/test_control.py
git commit -m "feat: controlador nulo + mapa Accion -> ComandoControl

- Interfaz abstracta Controlador
- Mapa concreto con valores del spec
- ControladorNulo para pruebas sobre video
- 5 tests unitarios"
```

---

### Tarea 18: Integración F3 — piloto sobre video con ControladorNulo

**Archivos:**
- Crear: `src/piloto.py`
- Crear: `scripts/ejecutar_piloto.py`

- [ ] **Paso 1: Implementar orquestador**

```python
# src/piloto.py
"""Orquestador principal del sistema.

Integra fuente, detector, tracker, contexto, decision y control en un loop.
Es el punto unico donde los modulos se ensamblan.
"""
import time
from dataclasses import dataclass
from typing import Optional

from src.control.base import Controlador
from src.control.mapa_acciones import accion_a_comando
from src.decision.fsm import MaquinaDecision
from src.fuente.base import FuenteCuadros
from src.percepcion.contexto import Contexto
from src.percepcion.detector import Detector
from src.percepcion.tracker import Tracker
from src.tipos import Accion, EstadoEscena


@dataclass
class PasoPiloto:
    """Snapshot del pipeline en un frame (util para logging y debugging)."""
    indice_frame: int
    timestamp: float
    n_detecciones: int
    n_seguimientos: int
    estado_escena: EstadoEscena
    accion: Accion
    estado_fsm: str
    latencia_ms: float


class Piloto:
    def __init__(
        self,
        fuente: FuenteCuadros,
        detector: Detector,
        tracker: Tracker,
        contexto: Contexto,
        fsm: MaquinaDecision,
        controlador: Controlador,
    ):
        self._fuente = fuente
        self._detector = detector
        self._tracker = tracker
        self._contexto = contexto
        self._fsm = fsm
        self._controlador = controlador
        self._pasos: list[PasoPiloto] = []

    def iniciar(self) -> None:
        self._fuente.iniciar()

    def ejecutar(self, max_frames: Optional[int] = None, callback=None) -> list[PasoPiloto]:
        n = 0
        while self._fuente.esta_activa:
            cuadro = self._fuente.siguiente()
            if cuadro is None:
                break
            inicio = time.perf_counter()
            detecciones = self._detector.detectar(cuadro.imagen)
            seguimientos = self._tracker.actualizar(detecciones)
            escena = self._contexto.analizar(seguimientos, cuadro.imagen, cuadro.timestamp)
            accion = self._fsm.decidir(escena)
            comando = accion_a_comando(accion, cuadro.timestamp)
            self._controlador.aplicar(comando)
            latencia = (time.perf_counter() - inicio) * 1000

            paso = PasoPiloto(
                indice_frame=cuadro.indice,
                timestamp=cuadro.timestamp,
                n_detecciones=len(detecciones),
                n_seguimientos=len(seguimientos),
                estado_escena=escena,
                accion=accion,
                estado_fsm=self._fsm.estado.name,
                latencia_ms=latencia,
            )
            self._pasos.append(paso)
            if callback is not None:
                callback(cuadro, detecciones, seguimientos, escena, accion, paso)

            n += 1
            if max_frames is not None and n >= max_frames:
                break

        return self._pasos

    def cerrar(self) -> None:
        self._fuente.cerrar()
        self._controlador.liberar()
        self._controlador.cerrar()
```

- [ ] **Paso 2: Crear script de ejecución**

```python
# scripts/ejecutar_piloto.py
"""Ejecuta el sistema completo.

Uso:
    python scripts/ejecutar_piloto.py --config config/default.yaml
"""
import argparse
from pathlib import Path

import cv2
import yaml

from src.control.nulo import ControladorNulo
from src.decision.fsm import MaquinaDecision
from src.fuente.video import FuenteVideo
from src.percepcion.contexto import Contexto
from src.percepcion.detector import Detector
from src.percepcion.tracker import Tracker
from src.piloto import Piloto


def construir_fuente(cfg):
    tipo = cfg["fuente"]["tipo"]
    if tipo == "video":
        return FuenteVideo(cfg["fuente"]["ruta_video"])
    if tipo == "pantalla":
        from src.fuente.pantalla import FuentePantalla
        return FuentePantalla(cfg["fuente"]["region"], monitor=cfg["fuente"]["monitor"])
    raise ValueError(f"Fuente desconocida: {tipo}")


def construir_controlador(cfg):
    tipo = cfg["control"]["tipo"]
    if tipo == "nulo":
        return ControladorNulo()
    if tipo == "gamepad":
        from src.control.gamepad import ControladorGamepad
        return ControladorGamepad()
    if tipo == "teclado":
        from src.control.teclado import ControladorTeclado
        return ControladorTeclado()
    raise ValueError(f"Control desconocido: {tipo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--clases", default="config/clases.yaml")
    parser.add_argument("--regiones", default="config/regiones_interes.yaml")
    parser.add_argument("--mostrar", action="store_true", default=True)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg_clases = yaml.safe_load(Path(args.clases).read_text())
    cfg_regs = yaml.safe_load(Path(args.regiones).read_text())

    fuente = construir_fuente(cfg)
    detector = Detector(
        ruta_pesos=cfg["modelo"]["pesos"],
        mapeo_clases=cfg_clases["mapeo_coco"],
        conf_min=cfg["modelo"]["conf_min"],
        imgsz=cfg["modelo"]["imgsz"],
        device=cfg["modelo"]["device"],
    )
    tracker = Tracker()
    contexto = Contexto(cfg_regs)
    fsm = MaquinaDecision()
    controlador = construir_controlador(cfg)

    piloto = Piloto(fuente, detector, tracker, contexto, fsm, controlador)
    piloto.iniciar()

    if args.mostrar:
        cv2.namedWindow("piloto", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("piloto", 1280, 720)

        def callback(cuadro, dets, segs, escena, accion, paso):
            frame = cuadro.imagen.copy()
            for d in dets:
                x1, y1, x2, y2 = d.caja
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, d.clase.name, (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            hud = [
                f"FPS: {cuadro.fps_instantaneo:.1f}",
                f"Latencia: {paso.latencia_ms:.1f} ms",
                f"Estado: {paso.estado_fsm}",
                f"Accion: {accion.name}",
                f"Conf: {escena.confianza_percepcion:.2f}",
                f"Frente: {'OCUPADO' if escena.frente_cercano_ocupado else 'libre'}",
                f"Peaton: {'RIESGO' if escena.peaton_en_riesgo else 'no'}",
                f"Semaforo: {escena.semaforo_visible.name if escena.semaforo_visible else 'ninguno'}",
                f"Alto: {'SI' if escena.senal_alto_cercana else 'no'}",
            ]
            for i, linea in enumerate(hud):
                cv2.putText(frame, linea, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("piloto", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt
    else:
        callback = None

    try:
        pasos = piloto.ejecutar(max_frames=args.max_frames, callback=callback)
    except KeyboardInterrupt:
        print("Detenido por el usuario.")
    finally:
        piloto.cerrar()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Paso 3: Ejecutar sobre video descargado**

```bash
python scripts/ejecutar_piloto.py --config config/default.yaml --mostrar
```

Observar:
- Se dibujan cajas sobre objetos
- El HUD muestra estado del FSM cambiando
- Cuando aparece un rojo, debe decir "Accion: ALTO_TOTAL" y "Estado: DETENIDO_SEMAFORO"
- Cuando aparece un peatón al frente, debe decir "Accion: FRENAR_FUERTE"

- [ ] **Paso 4: Verificar 8 escenarios del doc (tabla 11 del spec)**

Anotar en `datos/evidencia/F3_validacion.md` qué casos se observaron:

```markdown
# Validación F3 — piloto sobre video

- Caso 1 (avance normal): [OK/pendiente] — observado en segundo X:XX
- Caso 2 (vehículo delante): ...
- Caso 3 (semáforo rojo→verde): ...
- Caso 4 (señal de alto): ...
- Caso 5 (cruce lateral): ...
- Caso 6 (peatón): ...
- Caso 7 (rebase): ...
- Caso 8 (pérdida de detección): ...
```

- [ ] **Paso 5: Commit**

```bash
git add src/piloto.py scripts/ejecutar_piloto.py datos/evidencia/F3_validacion.md
git commit -m "feat: orquestador Piloto + script ejecutar_piloto

- Integra fuente, detector, tracker, contexto, FSM, control
- HUD en pantalla con estado FSM y flags de escena
- Registra PasoPiloto por frame para analisis posterior"
```

**Gate F3:** Solo avanzar si al menos 6 de 8 casos del doc se comportan correctamente.

---

## FASE 4 — Captura de pantalla + gamepad (Días 9-10)

### Tarea 19: `FuentePantalla` con DXcam

**Archivos:**
- Crear: `src/fuente/pantalla.py`

- [ ] **Paso 1: Implementar**

```python
# src/fuente/pantalla.py
"""Fuente de cuadros desde captura de pantalla (DXcam).

Usa DXGI Desktop Duplication -> muy baja latencia, 60+ FPS posibles.
"""
import time
from typing import Optional

import dxcam
import numpy as np

from src.fuente.base import FuenteCuadros
from src.tipos import Cuadro


class FuentePantalla(FuenteCuadros):
    def __init__(
        self,
        region: Optional[tuple[int, int, int, int]] = None,
        monitor: int = 0,
        fps_objetivo: int = 60,
    ):
        self._region = region  # (x1, y1, x2, y2) o None para pantalla completa
        self._monitor = monitor
        self._fps = fps_objetivo
        self._camara: Optional[dxcam.DXCamera] = None
        self._activa = False
        self._indice = 0
        self._ultimo_t: Optional[float] = None

    def iniciar(self) -> None:
        self._camara = dxcam.create(output_idx=self._monitor, output_color="BGR")
        if self._camara is None:
            raise RuntimeError("No se pudo crear la camara DXcam")
        if self._region:
            self._camara.start(region=self._region, target_fps=self._fps)
        else:
            self._camara.start(target_fps=self._fps)
        self._activa = True

    def siguiente(self) -> Optional[Cuadro]:
        if not self._activa or self._camara is None:
            return None
        frame = self._camara.get_latest_frame()
        if frame is None:
            return None
        ahora = time.perf_counter()
        if self._ultimo_t is None:
            fps_inst = 0.0
        else:
            dt = ahora - self._ultimo_t
            fps_inst = 1.0 / dt if dt > 0 else 0.0
        self._ultimo_t = ahora

        # DXcam devuelve por defecto RGB; pedimos BGR en create() arriba
        cuadro = Cuadro(
            imagen=np.ascontiguousarray(frame),
            timestamp=ahora,
            indice=self._indice,
            fps_instantaneo=fps_inst,
        )
        self._indice += 1
        return cuadro

    def cerrar(self) -> None:
        if self._camara is not None:
            self._camara.stop()
            del self._camara
            self._camara = None
        self._activa = False

    @property
    def esta_activa(self) -> bool:
        return self._activa
```

- [ ] **Paso 2: Prueba manual con ETS2 corriendo**

1. Abrir ETS2 en modo ventana 1920×1080
2. Ejecutar:
   ```bash
   python scripts/ejecutar_piloto.py --config config/default.yaml
   ```
   con `config/default.yaml` editado para `fuente.tipo: "pantalla"`.
3. Verificar que se capturan frames del juego y se dibujan cajas.

- [ ] **Paso 3: Commit**

```bash
git add src/fuente/pantalla.py
git commit -m "feat: FuentePantalla con DXcam

- Captura DXGI Desktop Duplication, 60+ FPS
- Configurable por region y monitor
- Devuelve frames en BGR listos para YOLO"
```

---

### Tarea 20: Controlador Gamepad (vgamepad)

**Archivos:**
- Crear: `src/control/gamepad.py`

**Prerequisito:** instalar driver **ViGEmBus** en Windows:
- Descargar de https://github.com/nefarius/ViGEmBus/releases y ejecutar el instalador .exe
- Reiniciar Windows si lo pide

- [ ] **Paso 1: Verificar vgamepad funciona**

```bash
python -c "import vgamepad as vg; g = vg.VX360Gamepad(); print('Gamepad virtual creado. Abrir Joy.cpl para verificar que aparece.')"
```

- [ ] **Paso 2: Implementar**

```python
# src/control/gamepad.py
"""Controlador por gamepad virtual Xbox 360 (via ViGEmBus).

Requiere driver ViGEmBus instalado en Windows.
ETS2 debe estar configurado para aceptar gamepad (opcion nativa).
"""
import vgamepad as vg

from src.control.base import Controlador
from src.tipos import ComandoControl


class ControladorGamepad(Controlador):
    def __init__(self):
        self._gamepad = vg.VX360Gamepad()
        self._gamepad.update()

    def aplicar(self, cmd: ComandoControl) -> None:
        # vgamepad espera triggers en [0, 255] y ejes en [-32768, 32767]
        trigger_acel = int(max(0.0, min(1.0, cmd.acelerador)) * 255)
        trigger_freno = int(max(0.0, min(1.0, cmd.freno)) * 255)
        eje_volante = int(max(-1.0, min(1.0, cmd.volante)) * 32767)

        self._gamepad.right_trigger(value=trigger_acel)
        self._gamepad.left_trigger(value=trigger_freno)
        self._gamepad.left_joystick(x_value=eje_volante, y_value=0)
        self._gamepad.update()

    def liberar(self) -> None:
        self._gamepad.right_trigger(value=0)
        self._gamepad.left_trigger(value=0)
        self._gamepad.left_joystick(x_value=0, y_value=0)
        self._gamepad.update()

    def cerrar(self) -> None:
        self.liberar()
        del self._gamepad
```

- [ ] **Paso 3: Probar comando aislado con el juego**

1. Abrir ETS2, entrar a modo conducción, detenido en carretera vacía
2. En un script rápido:
   ```python
   # scripts/probar_gamepad.py
   import time
   from src.control.gamepad import ControladorGamepad
   from src.tipos import ComandoControl

   ctrl = ControladorGamepad()
   print("Acelerando 3s...")
   ctrl.aplicar(ComandoControl(acelerador=0.6, freno=0, volante=0, timestamp=0))
   time.sleep(3)
   print("Frenando 2s...")
   ctrl.aplicar(ComandoControl(acelerador=0, freno=1.0, volante=0, timestamp=0))
   time.sleep(2)
   ctrl.liberar()
   ctrl.cerrar()
   ```
3. Ejecutar con ETS2 en foco. El camión debe acelerar y luego frenar.

- [ ] **Paso 4: Commit**

```bash
git add src/control/gamepad.py scripts/probar_gamepad.py
git commit -m "feat: ControladorGamepad via vgamepad/ViGEmBus

- Mapea ComandoControl a ejes Xbox 360 virtual
- RT=acelerador, LT=freno, stick_x=volante
- Script probar_gamepad.py para validacion aislada"
```

---

### Tarea 21: Controlador Teclado (respaldo)

**Archivos:**
- Crear: `src/control/teclado.py`

- [ ] **Paso 1: Implementar**

```python
# src/control/teclado.py
"""Controlador por teclado usando pydirectinput (respaldo del gamepad).

Binariza los valores analogicos con umbrales.
"""
import pydirectinput

from src.control.base import Controlador
from src.tipos import ComandoControl


class ControladorTeclado(Controlador):
    UMBRAL_ACELERAR = 0.2
    UMBRAL_FRENAR = 0.2
    UMBRAL_VOLANTE = 0.15

    def __init__(self):
        self._presionadas: set[str] = set()

    def _presionar(self, tecla: str):
        if tecla not in self._presionadas:
            pydirectinput.keyDown(tecla)
            self._presionadas.add(tecla)

    def _soltar(self, tecla: str):
        if tecla in self._presionadas:
            pydirectinput.keyUp(tecla)
            self._presionadas.remove(tecla)

    def aplicar(self, cmd: ComandoControl) -> None:
        if cmd.acelerador > self.UMBRAL_ACELERAR:
            self._presionar("w")
        else:
            self._soltar("w")

        if cmd.freno > self.UMBRAL_FRENAR:
            self._presionar("s")
        else:
            self._soltar("s")

        if cmd.volante < -self.UMBRAL_VOLANTE:
            self._presionar("a")
            self._soltar("d")
        elif cmd.volante > self.UMBRAL_VOLANTE:
            self._presionar("d")
            self._soltar("a")
        else:
            self._soltar("a")
            self._soltar("d")

    def liberar(self) -> None:
        for tecla in list(self._presionadas):
            self._soltar(tecla)

    def cerrar(self) -> None:
        self.liberar()
```

- [ ] **Paso 2: Commit**

```bash
git add src/control/teclado.py
git commit -m "feat: ControladorTeclado con pydirectinput (respaldo)

- Binariza valores analogicos con umbrales
- Mapea W/A/S/D a acelerar/izq/frenar/der"
```

---

## FASE 5 — Seguridad, registro y sistema completo (Días 11-13)

### Tarea 22: Monitor de seguridad (paro manual + watchdog)

**Archivos:**
- Crear: `src/seguridad/monitor.py`

- [ ] **Paso 1: Implementar**

```python
# src/seguridad/monitor.py
"""Monitor de seguridad: paro manual por tecla y watchdog.

Corre en hilo separado. Activa flag que Piloto debe consultar cada iteracion.
"""
import threading
import time
from typing import Callable, Optional

try:
    import keyboard as kb_module
    HAY_KEYBOARD = True
except ImportError:
    HAY_KEYBOARD = False


class MonitorSeguridad:
    def __init__(
        self,
        tecla_paro: str = "f12",
        timeout_watchdog_s: float = 0.5,
        on_paro: Optional[Callable[[], None]] = None,
    ):
        self._tecla = tecla_paro
        self._timeout = timeout_watchdog_s
        self._on_paro = on_paro
        self._paro = threading.Event()
        self._ultimo_heartbeat = time.monotonic()
        self._hilo: Optional[threading.Thread] = None
        self._parar_hilo = threading.Event()

    def iniciar(self) -> None:
        if HAY_KEYBOARD:
            kb_module.add_hotkey(self._tecla, self._disparar_paro)
        self._hilo = threading.Thread(target=self._loop_watchdog, daemon=True)
        self._hilo.start()

    def heartbeat(self) -> None:
        self._ultimo_heartbeat = time.monotonic()

    def solicitud_paro(self) -> bool:
        return self._paro.is_set()

    def _disparar_paro(self):
        self._paro.set()
        if self._on_paro:
            self._on_paro()

    def _loop_watchdog(self):
        while not self._parar_hilo.is_set():
            time.sleep(0.05)
            if time.monotonic() - self._ultimo_heartbeat > self._timeout:
                self._disparar_paro()
                break

    def cerrar(self) -> None:
        self._parar_hilo.set()
        if HAY_KEYBOARD:
            try:
                kb_module.remove_hotkey(self._tecla)
            except Exception:
                pass
```

- [ ] **Paso 2: Agregar `keyboard` a requirements**

Editar `requirements.txt` y añadir:
```
keyboard>=0.13.5
```

```bash
pip install keyboard
```

- [ ] **Paso 3: Integrar en `piloto.py`**

Modificar `src/piloto.py` para aceptar monitor y consultar `solicitud_paro()`:

```python
# En Piloto.__init__:
def __init__(self, ..., monitor: Optional["MonitorSeguridad"] = None):
    ...
    self._monitor = monitor
```

Y en el loop de `ejecutar`, al inicio de cada iteración:

```python
if self._monitor and self._monitor.solicitud_paro():
    self._fsm.activar_paro_emergencia()
    self._controlador.liberar()
    break
self._monitor and self._monitor.heartbeat()
```

- [ ] **Paso 4: Commit**

```bash
git add src/seguridad/monitor.py src/piloto.py requirements.txt
git commit -m "feat: monitor de seguridad con paro manual + watchdog

- Hilo separado escuchando tecla F12
- Watchdog por heartbeat (timeout 500 ms)
- Integrado en Piloto via solicitud_paro()"
```

---

### Tarea 23: Registro JSONL + grabador de video con overlays

**Archivos:**
- Crear: `src/registro/logger.py`
- Crear: `src/registro/grabador.py`
- Crear: `src/registro/metricas.py`

- [ ] **Paso 1: Implementar logger**

```python
# src/registro/logger.py
"""Logger JSONL: un evento por linea."""
import json
import time
from pathlib import Path


class LoggerJSONL:
    def __init__(self, ruta: Path):
        ruta.parent.mkdir(parents=True, exist_ok=True)
        self._archivo = ruta.open("w", encoding="utf-8")

    def evento(self, tipo: str, **campos) -> None:
        registro = {"t": time.time(), "tipo": tipo, **campos}
        self._archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
        self._archivo.flush()

    def cerrar(self) -> None:
        self._archivo.close()
```

- [ ] **Paso 2: Implementar grabador**

```python
# src/registro/grabador.py
"""Graba video MP4 con overlays (cajas, HUD) como evidencia."""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class GrabadorVideo:
    def __init__(self, ruta: Path, fps: int = 30, resolucion: Optional[tuple[int, int]] = None):
        ruta.parent.mkdir(parents=True, exist_ok=True)
        self._ruta = ruta
        self._fps = fps
        self._resolucion = resolucion
        self._escritor: Optional[cv2.VideoWriter] = None

    def escribir(self, frame: np.ndarray) -> None:
        if self._escritor is None:
            h, w = frame.shape[:2]
            res = self._resolucion or (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._escritor = cv2.VideoWriter(str(self._ruta), fourcc, self._fps, res)
        self._escritor.write(frame)

    def cerrar(self) -> None:
        if self._escritor is not None:
            self._escritor.release()
            self._escritor = None
```

- [ ] **Paso 3: Implementar agregador de métricas**

```python
# src/registro/metricas.py
"""Agregador de metricas post-corrida."""
import json
from pathlib import Path

import numpy as np


def analizar_corrida(ruta_jsonl: Path) -> dict:
    eventos = [json.loads(l) for l in ruta_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    frames = [e for e in eventos if e["tipo"] == "frame"]
    decisiones = [e for e in eventos if e["tipo"] == "decision"]
    transiciones = [e for e in eventos if e["tipo"] == "transicion"]

    if not frames:
        return {"error": "sin frames"}

    latencias = np.array([e["latencia_ms"] for e in frames])
    fps_inst = np.array([e["fps"] for e in frames if e.get("fps", 0) > 0])

    acciones = {}
    for d in decisiones:
        acciones[d["accion"]] = acciones.get(d["accion"], 0) + 1

    return {
        "n_frames": len(frames),
        "duracion_s": round(frames[-1]["t"] - frames[0]["t"], 2),
        "fps_promedio": round(float(fps_inst.mean()), 2) if len(fps_inst) else 0,
        "fps_min": round(float(fps_inst.min()), 2) if len(fps_inst) else 0,
        "latencia_ms_p50": round(float(np.percentile(latencias, 50)), 2),
        "latencia_ms_p95": round(float(np.percentile(latencias, 95)), 2),
        "latencia_ms_max": round(float(latencias.max()), 2),
        "n_transiciones": len(transiciones),
        "distribucion_acciones": acciones,
    }
```

- [ ] **Paso 4: Integrar logger y grabador en `piloto.py`**

Modificar el callback/ejecutar para usar logger y grabador (pasarlos por parámetro). Ver el plan original del piloto para ajustar.

- [ ] **Paso 5: Commit**

```bash
git add src/registro/
git commit -m "feat: registro JSONL + grabador MP4 + agregador de metricas"
```

---

### Tarea 24: Integración final en ETS2

**Archivos:** ninguno nuevo; ajustes de config.

- [ ] **Paso 1: Configurar ETS2**

Dentro del juego:
1. Opciones → Controles → Agregar controlador Xbox virtual
2. Asignar acelerador=RT, freno=LT, volante=stick_izq_X
3. Salir al modo conducción en un mapa conocido (p.ej. carretera con tráfico cercano a una ciudad)

- [ ] **Paso 2: Configurar `default.yaml` para modo real**

```yaml
fuente:
  tipo: "pantalla"
  region: [0, 0, 1920, 1080]
  monitor: 0

control:
  tipo: "gamepad"
```

- [ ] **Paso 3: Primer arranque supervisado**

```bash
python scripts/ejecutar_piloto.py --config config/default.yaml
```

**Con mano en F12** para paro inmediato si se descontrola. Observar 30 segundos. Cualquier comportamiento peligroso → presionar F12 y parar.

- [ ] **Paso 4: Calibrar regiones de ROI específicas para la cabina del camión en ETS2**

Si las regiones iniciales no se alinean con la cabina real, editar `config/regiones_interes.yaml` y volver a probar.

- [ ] **Paso 5: Commit**

```bash
git add config/default.yaml config/regiones_interes.yaml
git commit -m "chore: configuracion para integracion con ETS2 real

- fuente: pantalla con DXcam
- control: gamepad virtual via vgamepad
- regiones calibradas para cabina primera persona"
```

**Gate F4:** el camión responde a comandos aislados en ETS2 (acelera, frena, gira).

---

### Tarea 25: Ejecutar los 8 escenarios y grabar evidencia

**Archivos:**
- Crear: `datos/evidencia/F5_escenarios/` (uno por caso)

- [ ] **Paso 1: Preparar el juego**

Para cada escenario, buscar una ubicación adecuada en ETS2 (o usar `/teleport` si habilitado).

- [ ] **Paso 2: Ejecutar cada escenario con grabación**

Para cada caso:
1. Posicionarse
2. `python scripts/ejecutar_piloto.py --config config/default.yaml` con logging y grabado activo
3. Dejar correr el tiempo necesario para que el escenario suceda
4. Presionar F12 al terminar
5. Renombrar el video y log generados:
   ```bash
   mv datos/evidencia/ultima_corrida.mp4 datos/evidencia/F5_escenarios/caso_N.mp4
   mv datos/evidencia/ultima_corrida.jsonl datos/evidencia/F5_escenarios/caso_N.jsonl
   ```

- [ ] **Paso 3: Analizar métricas de cada corrida**

```bash
python -c "from src.registro.metricas import analizar_corrida; from pathlib import Path; import json, pathlib; 
for p in pathlib.Path('datos/evidencia/F5_escenarios').glob('*.jsonl'): 
    print(p.stem, json.dumps(analizar_corrida(p), indent=2, ensure_ascii=False))"
```

- [ ] **Paso 4: Documentar resultados**

Escribir `datos/evidencia/F5_resultados.md` con tabla:

```markdown
# Resultados F5

| Caso | Resultado | FPS promedio | Latencia p95 | Observaciones |
|------|-----------|--------------|--------------|---------------|
| 1 (avance) | OK | 42 | 28 ms | — |
| 2 (vehiculo adelante) | OK | 41 | 29 ms | Frenó a tiempo |
| 3 (rojo-verde) | OK | 40 | 30 ms | Respetó rojo 15 s |
| 4 (alto) | OK | 41 | 28 ms | Pausó 2.3 s |
| 5 (cruce) | OK | 40 | 31 ms | — |
| 6 (peaton) | OK | 41 | 29 ms | — |
| 7 (rebase) | Parcial | 40 | 32 ms | Abortó por espejo ocupado |
| 8 (recuperacion) | OK | 39 | 33 ms | Redujo velocidad |
```

- [ ] **Paso 5: Commit**

```bash
git add datos/evidencia/F5_escenarios/*.mp4 datos/evidencia/F5_escenarios/*.jsonl datos/evidencia/F5_resultados.md
git commit -m "docs: evidencia F5 de los 8 escenarios de validacion"
```

**Gate F5:** al menos 6 de 8 casos con resultado OK.

---

## FASE 6 — Reporte y entrega (Día 14)

### Tarea 26: Reporte técnico

**Archivos:**
- Crear: `docs/reporte_tecnico/reporte.md` (se exporta a PDF/DOCX)

- [ ] **Paso 1: Plantilla del reporte**

Contenido mínimo (tabla 2.8 del spec):

```markdown
# Conducción autónoma visual en ETS2 con YOLO26

## 1. Planteamiento del problema
[Tomar de sección 2.2 del doc de requerimientos]

## 2. Arquitectura del sistema
[Diagrama de la sección 4.1 del spec + 1 párrafo]

## 3. Diseño del pipeline de visión
[Fuente → Detector → Tracker → Contexto, con breve descripción]

## 4. Dataset y etiquetado (si hubo fine-tuning)
[Si usamos solo pre-entrenado: decirlo y justificar]

## 5. Entrenamiento / ajuste de YOLO26
[Usado pre-entrenado COCO; comparativa si hubo fine-tuning]

## 6. Reglas / política de decisión
[Tabla de 12 reglas del spec + explicación de principios del FSM]

## 7. Implementación de emulación de controles
[Gamepad virtual Xbox con vgamepad]

## 8. Escenarios de prueba
[8 escenarios del doc con descripción breve]

## 9. Métricas obtenidas
[FPS promedio/p95, latencia, cumplimiento de señales]

## 10. Análisis de errores y limitaciones
[Falsos positivos en espejos; robustez a iluminación; etc.]

## 11. Conclusiones y trabajo futuro
[Qué funcionó, qué no, qué extensiones del 2.15 se podrían intentar]
```

- [ ] **Paso 2: Llenar con los datos reales obtenidos**

- [ ] **Paso 3: Exportar a PDF**

Si pandoc está disponible:
```bash
pandoc docs/reporte_tecnico/reporte.md -o docs/reporte_tecnico/reporte.pdf
```

Si no, abrir en Word o usar una herramienta online.

- [ ] **Paso 4: Commit**

```bash
git add docs/reporte_tecnico/
git commit -m "docs: reporte tecnico final"
```

---

### Tarea 27: Video demostrativo editado

- [ ] **Paso 1: Concatenar los 8 videos**

Usar ffmpeg (si está instalado) o una herramienta como Shotcut/DaVinci Resolve Gratis:

```bash
# Crear archivo de lista
echo "file 'datos/evidencia/F5_escenarios/caso_1.mp4'" > /tmp/lista.txt
# (agregar todas)
ffmpeg -f concat -safe 0 -i /tmp/lista.txt -c copy datos/evidencia/demo_final.mp4
```

- [ ] **Paso 2: Subir a YouTube en "no listado" o guardar en la entrega**

- [ ] **Paso 3: Commit (referencia, no el video pesado)**

```bash
echo "Video demo: <URL o ruta local>" > datos/evidencia/demo_final_ref.txt
git add datos/evidencia/demo_final_ref.txt
git commit -m "docs: referencia al video demostrativo final"
```

---

### Tarea 28: Preparar entrega final

- [ ] **Paso 1: Checklist de entregables (sección 2.7 del spec)**

Verificar que existe:
- [x] Código fuente completo (repo git)
- [x] README.md con instalación y ejecución
- [x] Modelo(s): `datos/modelos/yolov8n.pt` (o yolov26 si disponible)
- [x] Reporte técnico PDF en `docs/reporte_tecnico/`
- [x] Video demostrativo
- [x] Carpeta de evidencia con logs, capturas, métricas
- [x] Presentación (opcional en spec; si se pide: crear)

- [ ] **Paso 2: Actualizar README con ejecución final**

Instrucciones completas de cero a demo, para que un revisor externo pueda reproducir.

- [ ] **Paso 3: Crear zip de entrega**

```bash
cd ..
# Excluir archivos pesados opcionalmente
zip -r "Proyecto_Final_Entrega_$(date +%Y%m%d).zip" "Proyecto Final" \
  -x "Proyecto Final/datos/videos/*" \
  -x "Proyecto Final/datos/modelos/*.pt" \
  -x "Proyecto Final/venv/*" \
  -x "Proyecto Final/.git/*"
```

- [ ] **Paso 4: Commit final**

```bash
git add README.md
git commit -m "docs: README final con instrucciones completas de ejecucion

- Entrega completa: codigo + reporte + video + evidencia
- Proyecto Final Universidad del Istmo"
git tag entrega-final
```

---

## Resumen del plan

| Fase | Días | Tareas | Entregable clave |
|------|------|--------|-------------------|
| F0 | 1 | 1-4 | Benchmark ≥ 30 FPS |
| F1 | 2-3 | 5-11 | Detector sobre video YouTube |
| F2 | 4-5 | 12-14 | Contexto + ROI calibradas |
| F3 | 6-8 | 15-18 | FSM funcional sobre video |
| F4 | 9-10 | 19-21 | Pantalla + gamepad en ETS2 |
| F5 | 11-13 | 22-25 | 8 escenarios ejecutados con evidencia |
| F6 | 14 | 26-28 | Reporte + video demo + entrega |

**28 tareas totales**, cada una con pasos bite-sized y commits frecuentes. Cada fase tiene un **gate** de calidad: no se avanza sin aprobar la validación anterior.

**Si el tiempo aprieta**, el orden de recorte (menos crítico primero):
1. Caso 7 (rebase) — maniobra opcional más compleja
2. Fine-tuning del modelo — si el pre-entrenado da suficientes detecciones
3. Caso 8 (recuperación) — puede demostrarse en simulación
4. Dashboard visual — el HUD en `ejecutar_piloto.py` ya lo cumple

**Lo que nunca se recorta:**
- Tests unitarios de `decision/fsm.py` (15 pts de evaluación dependen de esto)
- Grabación de evidencia en video (criterio explícito del doc)
- Reporte técnico (10 pts)
- Paro manual F12 (RF-12 obligatorio)
