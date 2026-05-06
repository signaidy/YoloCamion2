# Conducción autónoma visual en ETS2

Proyecto Final — Universidad del Istmo (Guatemala)

Este repositorio contiene un piloto visual para **Euro Truck Simulator 2** basado en:

- **YOLO11n** para detección y tracking de objetos.
- **YOLOP** para segmentación de carriles y área manejable.
- **Pure Pursuit visual** para seguimiento de carril.
- **OCR del velocímetro del HUD** con OpenCV y un banco de prototipos reales.
- **FSM + PID sobre gamepad virtual** para acelerar, frenar y girar.

## Estado actual

Actualmente el sistema puede:

- mantener carril en autopista usando `ll_mask` de YOLOP con fallback a `da_mask`
- seguir vehículos y frenar según TTC visual
- respetar semáforos y señales de alto con histéresis
- leer la velocidad desde el HUD del camión sin depender de telemetría interna
- capturar desde video, pantalla completa o ventana específica
- generar artefactos de debug para analizar carril, OCR y decisiones

Limitaciones actuales:

- **no** sigue todavía la ruta completa del GPS/minimapa
- está calibrado principalmente para **ETS2 en 1920x1080**
- las maniobras complejas de salidas, intersecciones y cambios de carril guiados por ruta siguen pendientes
- la lectura del velocímetro depende del HUD visible y del banco de prototipos configurado

## Requisitos previos

| Requisito | Versión mínima | Notas |
| --------- | ------------- | ----- |
| Python | 3.11+ | Probado en 3.14 |
| GPU NVIDIA | — | CUDA recomendado para inferencia en vivo |
| Driver NVIDIA | 525+ | 596+ recomendado |
| [ViGEmBus](https://github.com/nefarius/ViGEmBus/releases/latest) | 1.22+ | Requerido para gamepad virtual |
| Euro Truck Simulator 2 | — | Resolución 1920×1080 recomendada |

> Sin `ViGEmBus`, `vgamepad` no puede exponer el control virtual al juego.

## Instalación

```bat
:: 1. Clonar y entrar al repo
git clone <repo>
cd YoloCamion

:: 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate

:: 3. Instalar dependencias
pip install -r requirements.txt
```

`requirements.txt` ya incluye el índice de PyTorch para `cu126`. Si tu entorno usa otra versión de CUDA, puedes instalar `torch` y `torchvision` manualmente antes de `requirements.txt`.

### Verificar instalación

```bat
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

Lo importante es que `CUDA: True` aparezca en una máquina con GPU configurada.

## Modelos y archivos esperados

### Detección de objetos

La configuración por defecto espera los pesos en:

```text
datos/modelos/yolo11n.pt
```

La ruta se controla desde `config/default.yaml`.

### Segmentación de carril

YOLOP se descarga automáticamente desde `torch.hub` en la primera ejecución.

### OCR del velocímetro

El lector usa un banco de prototipos reales por defecto:

```text
config/velocidad_dashboard_prototypes.json
```

Ese banco se puede extender con muestras nuevas tomadas de `datos/evidencia/velocidad_componentes/`.

## Ejecución

```bat
:: Activar entorno
venv\Scripts\activate

:: Ejecución básica
python scripts/ejecutar_piloto.py

:: Countdown para cambiar al juego
python scripts/ejecutar_piloto.py --delay 5

:: Sin grabación MP4
python scripts/ejecutar_piloto.py --delay 5 --sin-video
```

### Fuentes soportadas

```bat
:: Video grabado
python scripts/ejecutar_piloto.py --fuente video

:: Pantalla completa / región de monitor
python scripts/ejecutar_piloto.py --fuente pantalla

:: Ventana de ETS2
python scripts/ejecutar_piloto.py --fuente ventana
```

`pantalla` usa `dxcam` con fallback a `mss`. `ventana` usa captura por ventana y puede ser útil cuando el juego no está en primer plano.

### Control

En `config/default.yaml` el modo por defecto es:

```yaml
control:
  tipo: "gamepad"
```

También existen `nulo` y `teclado` para pruebas.

## Debug y evidencia

### Flags útiles

```bat
:: Log de carril cada 30 frames
python scripts/ejecutar_piloto.py --debug-carril

:: Guardar overlay simple de YOLOP
python scripts/ejecutar_piloto.py --debug-carril-img

:: Guardar panel compuesto del modelo + ROI del velocímetro + dumps OCR
python scripts/ejecutar_piloto.py --debug-yolop

:: Guardar clasificación de carriles ego / mismo sentido / contrario
python scripts/ejecutar_piloto.py --debug-clasif-carriles
```

### Archivos generados

En `datos/evidencia/` el sistema puede producir:

- `sesion_*.jsonl`: log estructurado de frames, decisiones y eventos
- `debug_yolop_*.jpg`: overlay simple de máscaras sobre el frame
- `debug_modelo_*.jpg`: panel compuesto con captura, entrada del modelo y máscaras
- `debug_vel_roi_*.jpg`: ROI ampliada del velocímetro con componentes detectados
- `debug_carriles_*.jpg`: visualización de clasificación/offset de carril
- `grabacion_*.mp4`: grabación completa si `registro.grabar_video=true`

Con `--debug-yolop` y `velocidad_dashboard.dump_componentes_dir` habilitado, también se guardan crops de dígitos en:

```text
datos/evidencia/velocidad_componentes/
```

Estos artefactos están ignorados por git para mantener el repo limpio.

## Banco de prototipos del velocímetro

Para extender el OCR con muestras reales:

```bat
python scripts/registrar_prototipos_velocidad.py ^
  --output config/velocidad_dashboard_prototypes.json ^
  --sample 3=datos/evidencia/velocidad_componentes/vel_comp_000060_0.png ^
  --sample 1=datos/evidencia/velocidad_componentes/vel_comp_000180_0.png
```

Cada muestra debe ser un crop monocromático del dígito ya segmentado.

## Tests

```bat
venv\Scripts\activate
pytest
```

Si quieres correr solo los tests más ligados al estado actual del piloto:

```bat
pytest tests/test_pure_pursuit.py ^
       tests/test_velocidad_dashboard.py ^
       tests/test_carril_speed_policy.py ^
       tests/test_carril_steering_policy.py ^
       tests/test_velocidad_feedback_policy.py
```

## Estructura

```text
YoloCamion/
├── config/
│   ├── default.yaml
│   ├── regiones_interes.yaml
│   └── velocidad_dashboard_prototypes.json
├── datos/
│   ├── modelos/
│   ├── videos/
│   └── evidencia/
├── scripts/
│   ├── ejecutar_piloto.py
│   └── registrar_prototipos_velocidad.py
├── src/
│   ├── control/
│   ├── decision/
│   ├── fuente/
│   ├── percepcion/
│   ├── registro/
│   └── seguridad/
└── tests/
```

## Siguiente paso lógico

La siguiente capacidad grande pendiente es una capa de navegación por ruta:

- leer minimapa / GPS del HUD
- decidir carril objetivo antes de salidas o cruces
- ejecutar maniobras guiadas por ruta, no solo seguimiento local de carril

Hoy el piloto está más cerca de un **autopilot visual de autopista** que de un conductor completo de ruta.
