# Conducción autónoma visual en ETS2 con YOLO26

Proyecto Final — Universidad del Istmo (Guatemala)

## Requisitos previos

| Requisito | Versión mínima | Notas |
| --------- | ------------- | ----- |
| Python | 3.11+ | Probado en 3.14 |
| GPU NVIDIA | — | CUDA 12.6+ recomendado |
| Driver NVIDIA | 525+ | Driver 596+ recomendado |
| [ViGEmBus](https://github.com/nefarius/ViGEmBus/releases/latest) | 1.22+ | **Requerido para gamepad virtual** — instalar antes que vgamepad |
| Euro Truck Simulator 2 | — | Resolución 1920×1080 recomendada |

> **ViGEmBus** es un driver de Windows para gamepads virtuales. Sin él `vgamepad` falla al iniciar.
> Descargar e instalar `ViGEmBus_Setup_x64.msi` desde el link de arriba, luego reiniciar.

## Instalación

```bat
:: 1. Clonar y entrar al directorio
git clone <repo>
cd YoloCamion

:: 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate

:: 3. Instalar PyTorch con CUDA 12.6 (primero, antes que requirements.txt)
pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu126

:: 4. Instalar el resto de dependencias
pip install -r requirements.txt
```

> Si tu driver soporta otra versión de CUDA, reemplaza `cu126` por `cu124`, `cu128`, etc.
> Verifica con: `nvidia-smi` → busca "CUDA Version" en la esquina superior derecha.

### Verificar instalación

```bat
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

Debe mostrar algo como: `2.11.0+cu126 | CUDA: True`

## Pesos del modelo

### YOLO (detección de objetos)

Colocar el archivo de pesos en `datos/modelos/yolo26n.pt`.
La ruta se configura en `config/default.yaml` bajo `modelo.pesos`.

### YOLOP (carriles y área manejable)

Se descarga automáticamente desde Torch Hub en la primera ejecución (~500 MB).
Se guarda en `%USERPROFILE%\.cache\torch\hub\`.

## Ejecutar

```bat
:: Activar entorno (si no está activo)
venv\Scripts\activate

:: Ejecución básica (captura la pantalla con dxcam/mss)
python scripts/ejecutar_piloto.py

:: Con countdown de 5 segundos para cambiar al juego
python scripts/ejecutar_piloto.py --delay 5

:: Sin grabar video (más rápido, útil en pruebas)
python scripts/ejecutar_piloto.py --delay 5 --sin-video
```

### Opciones de debug

```bat
:: Imprimir error de carril por consola cada 30 frames
python scripts/ejecutar_piloto.py --debug-carril

:: Guardar imagen de máscaras YOLOP cada 60 frames en datos/evidencia/
python scripts/ejecutar_piloto.py --debug-carril-img

:: Guardar imagen compuesta: captura original | entrada del modelo | máscaras + look-ahead
python scripts/ejecutar_piloto.py --debug-yolop

:: Todo junto con delay
python scripts/ejecutar_piloto.py --delay 5 --sin-video --debug-yolop
```

Las imágenes de debug se guardan en `datos/evidencia/debug_modelo_XXXXXX.jpg`.

### Cambiar fuente de video

```bat
:: Usar archivo de video en lugar de captura en vivo
python scripts/ejecutar_piloto.py --fuente video

:: Captura de pantalla completa (en vez de ventana específica)
python scripts/ejecutar_piloto.py --fuente pantalla

:: Captura por ventana aunque ETS2 quede tapado (puede lavar contraste en DirectX)
python scripts/ejecutar_piloto.py --fuente ventana
```

### Otras opciones

```bat
python scripts/ejecutar_piloto.py --help
```

## Tests

```bat
venv\Scripts\activate
pytest
```

## Estructura

```text
YoloCamion/
├── config/
│   ├── default.yaml          # Configuración principal (modelo, fuente, control)
│   └── regiones_interes.yaml # ROI calibradas para detección
├── datos/
│   ├── modelos/              # Pesos YOLO (yolo26n.pt)
│   ├── videos/               # Videos de prueba
│   └── evidencia/            # Salida: video grabado + imágenes de debug
├── scripts/
│   └── ejecutar_piloto.py    # Punto de entrada principal
├── src/
│   ├── control/              # PID, Pure Pursuit, gamepad
│   ├── decision/             # FSM de conducción
│   ├── fuente/               # Captura de pantalla/video
│   ├── percepcion/           # YOLO, YOLOP, tracker, flujo óptico
│   └── registro/             # Logger, grabador de video
└── tests/
```
