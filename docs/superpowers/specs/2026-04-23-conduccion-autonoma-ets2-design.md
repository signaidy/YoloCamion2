# Conducción autónoma visual en ETS2 con YOLO26 — Documento de diseño

- **Autor:** Equipo de Proyecto Final (Universidad del Istmo)
- **Fecha:** 2026-04-23
- **Estado:** Aprobado (pendiente de revisión escrita)
- **Plazo de entrega:** 2 semanas desde 2026-04-23

---

## 1. Resumen ejecutivo

Sistema que observa la salida visual de *Euro Truck Simulator 2* (ETS2), detecta objetos de interés con **YOLO26** (Ultralytics), interpreta la escena con reglas explicables y emula controles externos para conducir el camión. El juego no se modifica ni se inyecta código en su proceso; toda la información de decisión proviene de visión por computadora.

Implementación en **Python 3.11** con pipeline optimizado GPU-first sobre una **RTX 3070 Ti Laptop**. Estrategia de desarrollo por fases: primero validación sobre video de YouTube descargado, luego captura en vivo del juego con emulación de gamepad virtual.

---

## 2. Objetivos y no-objetivos

### 2.1 Objetivos

- Cumplir los 12 requerimientos funcionales (RF-01…RF-12) del documento de requerimientos oficial.
- Cumplir los 7 requerimientos no funcionales (RNF-01…RNF-07).
- Demostrar los 8 escenarios de validación obligatorios (sección 2.9 del documento de requerimientos).
- Producir los 7 entregables exigidos (sección 2.7).
- Mantener la lógica de decisión **explicable** (sin cajas negras de tipo red neuronal en la etapa de decisión).

### 2.2 No-objetivos (fuera de alcance deliberado)

- Entrenamiento de YOLO26 desde cero: usaremos pre-entrenado COCO + fine-tuning dirigido si es necesario.
- Planificación de rutas globales: solo maniobras locales.
- Aprendizaje por refuerzo, control predictivo avanzado o redes neuronales para decisión.
- Estimación métrica exacta de distancias: usamos heurísticas por área/posición relativa.
- Portabilidad a otros simuladores.

---

## 3. Trazabilidad a requerimientos

### 3.1 Requerimientos funcionales

| ID | Requerimiento (resumen) | Cómo lo cumple el diseño |
|----|-------------------------|--------------------------|
| RF-01 | Capturar imagen del juego en tiempo real | Módulo `fuente/pantalla.py` con DXcam |
| RF-02 | Operar con cámara primera persona | Configuración fija del juego, documentada en `config/default.yaml` |
| RF-03 | Usar YOLO26 como núcleo de detección | Módulo `percepcion/detector.py` con Ultralytics |
| RF-04 | Detectar vehículos, peatones, semáforos, alto | Enum `Clase` + mapeo desde COCO |
| RF-05 | Considerar vista frontal + retrovisores + laterales | `percepcion/contexto.py` con ROI por `Region` |
| RF-06 | Decidir acelerar/frenar | Reglas 8, 11, 12 del FSM |
| RF-07 | Detenerse ante rojo y alto | Reglas 4 y 6 del FSM |
| RF-08 | Decidir cruce vs. esperar | Regla 7 del FSM (cruce tras pausa + laterales libres) |
| RF-09 | Maniobra de rebase con verificación lateral | Reglas 9-10 del FSM (doble señal positiva requerida) |
| RF-10 | Emulación externa de controles | Módulo `control/gamepad.py` (vgamepad) |
| RF-11 | Logs, capturas o video como evidencia | Módulo `registro/` (JSONL + grabador de video) |
| RF-12 | Mecanismo de paro manual inmediato | Módulo `seguridad/monitor.py` (tecla F12 + watchdog) |

### 3.2 Requerimientos no funcionales

| ID | Requerimiento (resumen) | Cómo lo cumple el diseño |
|----|-------------------------|--------------------------|
| RNF-01 | Reproducibilidad | `config/` en YAML, `requirements.txt`, README documentado |
| RNF-02 | Latencia medida y reportada | `registro/metricas.py` registra por ciclo |
| RNF-03 | Módulos claramente separados | 7 paquetes con interfaces abstractas |
| RNF-04 | Instrucciones de instalación y ejecución | README + scripts ejecutables |
| RNF-05 | Decisión explicable, no caja negra | FSM con reglas priorizadas + log de razón |
| RNF-06 | Sin modificar el juego | Solo captura de pantalla + entrada externa |
| RNF-07 | Fuente principal de decisión es visual | Toda la lógica consume solo `EstadoEscena`, que proviene de visión |

---

## 4. Arquitectura

### 4.1 Vista de alto nivel

```
┌─────────────────┐
│  FUENTE CUADROS │  ← video YouTube (Fase 0) o pantalla ETS2 (Fase 2+)
└────────┬────────┘
         ↓ Cuadro
┌─────────────────┐
│    DETECTOR     │  ← YOLO26 pre-entrenado (COCO) con mapeo a clases propias
└────────┬────────┘
         ↓ list[Deteccion]
┌─────────────────┐
│     TRACKER     │  ← ByteTrack integrado en Ultralytics
└────────┬────────┘
         ↓ list[Seguimiento]
┌─────────────────┐
│    CONTEXTO     │  ← ROI (frente/espejos/laterales) + análisis de riesgo
└────────┬────────┘
         ↓ EstadoEscena
┌─────────────────┐
│    DECISIÓN     │  ← máquina de estados con 12 reglas priorizadas
└────────┬────────┘
         ↓ Accion
┌─────────────────┐
│    CONTROL      │  ← vgamepad / teclado / nulo (para pruebas de video)
└─────────────────┘

Paralelo: SEGURIDAD (F12 + watchdog), REGISTRO (JSONL + video)
```

### 4.2 Principios de diseño

1. **Separación estricta por responsabilidad**: cada módulo tiene una interfaz abstracta y puede probarse aislado.
2. **Datos planos entre módulos**: solo dataclasses (no objetos con comportamiento compartido).
3. **Adaptadores intercambiables** para fuente y control: mismo código corre sobre video o pantalla; igual con gamepad/teclado/nulo.
4. **Conservador por defecto**: ante duda, frenar.
5. **Todo loggeable con razón**: cada transición del FSM emite por qué ocurrió.

---

## 5. Estructura de carpetas

```
Proyecto Final/
├── README.md
├── requirements.txt
├── pyproject.toml
├── config/
│   ├── default.yaml
│   ├── regiones_interes.yaml
│   └── clases.yaml
├── src/
│   ├── tipos.py
│   ├── piloto.py
│   ├── fuente/        { base.py, video.py, pantalla.py }
│   ├── percepcion/    { detector.py, tracker.py, semaforo.py, contexto.py }
│   ├── decision/      { estado.py, reglas.py, fsm.py }
│   ├── control/       { base.py, gamepad.py, teclado.py, nulo.py }
│   ├── seguridad/     { monitor.py }
│   └── registro/      { logger.py, metricas.py, grabador.py }
├── scripts/
│   ├── descargar_video.py
│   ├── benchmark_fps.py
│   ├── probar_deteccion.py
│   ├── calibrar_regiones.py
│   └── ejecutar_piloto.py
├── datos/
│   ├── videos/           (gitignore)
│   ├── modelos/          (gitignore si son grandes)
│   └── evidencia/
├── tests/
│   ├── test_contexto.py
│   ├── test_decision.py
│   ├── test_control.py
│   └── fixtures/
└── docs/
    ├── superpowers/specs/
    └── reporte_tecnico/
```

---

## 6. Stack técnico

| Componente | Librería | Versión mínima | Justificación |
|------------|----------|----------------|---------------|
| Lenguaje | Python | 3.11 | Compatibilidad Ultralytics; rapidez de desarrollo |
| Detector | `ultralytics` | última con YOLO26 | Requerimiento explícito (RF-03) |
| Captura pantalla | `dxcam` (o `bettercam`) | última | DXGI Desktop Duplication, ~240 FPS |
| Captura video | `opencv-python` (cv2.VideoCapture) | 4.9+ | Para Fase 0 sobre video de YouTube |
| Descarga YouTube | `yt-dlp` | última | Datos de prueba reproducibles |
| Preprocesado | `opencv-python` | 4.9+ | Estándar, C++ nativo |
| Emulación gamepad | `vgamepad` (requiere ViGEmBus driver) | última | Control analógico real, compatible con ETS2 |
| Emulación teclado | `pydirectinput` | última | Respaldo sin driver adicional |
| Métricas/análisis | `pandas`, `numpy` | última | Estándar |
| Tests | `pytest` | última | Estándar |
| Config | `pyyaml` | última | YAML legible, versionable |

---

## 7. Contratos de datos (`src/tipos.py`)

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

@dataclass
class Cuadro:
    imagen: np.ndarray            # BGR (H, W, 3)
    timestamp: float
    indice: int
    fps_instantaneo: float

class Clase(Enum):
    VEHICULO = "vehiculo"         # agrupa COCO: car + truck + bus
    MOTOCICLETA = "motocicleta"
    PEATON = "peaton"
    SEMAFORO = "semaforo"
    SENAL_ALTO = "senal_alto"
    DESCONOCIDO = "desconocido"

@dataclass
class Deteccion:
    clase: Clase
    caja: tuple[int, int, int, int]   # (x1, y1, x2, y2) en px
    confianza: float
    area: int

@dataclass
class Seguimiento(Deteccion):
    id_seguimiento: int
    edad: int                     # frames consecutivos visible

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
    frente_cercano_ocupado: bool
    frente_lejano_ocupado: bool
    peaton_en_riesgo: bool
    semaforo_visible: Optional[EstadoSemaforo]
    senal_alto_cercana: bool
    espejo_izq_ocupado: bool
    espejo_der_ocupado: bool
    vehiculos_totales: int
    confianza_percepcion: float   # 0-1
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
    acelerador: float             # 0.0 - 1.0
    freno: float                  # 0.0 - 1.0
    volante: float                # -1.0 a +1.0
    timestamp: float
```

---

## 8. Detalle por módulo

### 8.1 `fuente/` — captura intercambiable

**Interfaz `FuenteCuadros`:**
```python
class FuenteCuadros(ABC):
    def iniciar(self) -> None: ...
    def siguiente(self) -> Optional[Cuadro]: ...
    def cerrar(self) -> None: ...
    @property
    def esta_activa(self) -> bool: ...
```

- **`FuenteVideo`**: `cv2.VideoCapture` sobre archivo local. Permite seek y loop.
- **`FuentePantalla`**: `dxcam.create()` con región configurable. FPS objetivo ≥ 30.

### 8.2 `percepcion/`

**`detector.py`** — carga `yolov26n.pt` (o `yolov26s.pt`), llama `model(frame, conf=0.35, device='cuda')`, mapea IDs COCO → `Clase`.

**`tracker.py`** — usa `model.track(persist=True)` de Ultralytics (ByteTrack integrado). Añade `id_seguimiento` y `edad`.

**`semaforo.py`** — recibe caja de un semáforo detectado, recorta ROI, convierte a HSV, cuenta píxeles dominantes en rangos rojo/amarillo/verde, devuelve `EstadoSemaforo`. Sin modelo adicional (análisis de color puro).

**`contexto.py`** — consume `list[Seguimiento]` y configuración de ROI (polígonos), calcula intersecciones, aplica heurísticas:
- `frente_cercano_ocupado`: vehículo/peatón en ROI frontal con área ≥ umbral o `y2` muy bajo en frame
- `peaton_en_riesgo`: peatón en zona frontal o lateral cercana
- `espejo_X_ocupado`: vehículo en ROI del espejo con `edad ≥ 3` (anti-parpadeo)
- `confianza_percepcion`: media móvil de estabilidad de tracks

Devuelve `EstadoEscena`.

### 8.3 `decision/`

Máquina de estados finita con 13 estados y 12 reglas priorizadas (ver Sección 10).

### 8.4 `control/`

**Interfaz `Controlador`:**
```python
class Controlador(ABC):
    def aplicar(self, cmd: ComandoControl) -> None: ...
    def liberar(self) -> None: ...   # soltar todo (emergencia)
    def cerrar(self) -> None: ...
```

- **`ControladorGamepad`** (vgamepad): mapea cmd a ejes Xbox 360 virtual: RT=acelerador, LT=freno, stick_x=volante.
- **`ControladorTeclado`** (pydirectinput): binariza a W/A/S/D con umbrales.
- **`ControladorNulo`**: solo loggea; default en Fases 0-3 para seguridad.

**Mapa acción → comando:**

| Acción | acelerador | freno | volante |
|--------|-----------|-------|---------|
| `MANTENER` | 0.3 | 0 | 0 |
| `ACELERAR` | 0.6 | 0 | 0 |
| `FRENAR_SUAVE` | 0 | 0.4 | 0 |
| `FRENAR_FUERTE` | 0 | 0.8 | 0 |
| `ALTO_TOTAL` | 0 | 1.0 | 0 |
| `GIRAR_IZQ` | 0.2 | 0 | -0.5 |
| `GIRAR_DER` | 0.2 | 0 | +0.5 |
| `REBASAR_IZQ` | 0.8 | 0 | -0.3 |
| `REBASAR_DER` | 0.8 | 0 | +0.3 |
| `ESPERAR` | 0 | 0 | 0 |

### 8.5 `seguridad/monitor.py`

- Hilo separado escuchando `F12` (paro duro) y `ESC` (paro suave).
- Watchdog: si el loop principal no emite heartbeat en 500 ms, invoca `controlador.liberar()` y termina.
- Hook `atexit` libera controles aunque el proceso muera.

### 8.6 `registro/`

- **`logger.py`**: JSONL, un evento por línea. Tipos: `frame`, `decision`, `transicion`, `error`, `seguridad`.
- **`metricas.py`**: agrega por sesión → FPS promedio/mínimo/p95, latencia percepción-acción, cumplimiento de señales, eventos de seguridad.
- **`grabador.py`**: MP4 con overlays (cajas dibujadas, acción actual, estado FSM, FPS) — evidencia audiovisual para entregables.

---

## 9. Configuración (`config/`)

### 9.1 `default.yaml`

```yaml
fuente:
  tipo: "video"                   # "video" | "pantalla"
  ruta_video: "datos/videos/ets2_gameplay.mp4"
  monitor: 1
  region: [0, 0, 1920, 1080]

modelo:
  pesos: "datos/modelos/yolov26n.pt"
  imgsz: 640
  conf_min: 0.35
  device: "cuda"

control:
  tipo: "nulo"                    # "nulo" | "gamepad" | "teclado"

seguridad:
  tecla_paro: "f12"
  timeout_watchdog_ms: 500

registro:
  ruta_base: "datos/evidencia"
  grabar_video: true
  fps_objetivo: 30
```

### 9.2 `regiones_interes.yaml`

Polígonos en coordenadas de frame 1920×1080. Calibrables por escenario con `scripts/calibrar_regiones.py`.

### 9.3 `clases.yaml`

Mapeo `id_coco → Clase`:

```yaml
2: VEHICULO        # car
7: VEHICULO        # truck
5: VEHICULO        # bus
3: MOTOCICLETA     # motorcycle
0: PEATON          # person
9: SEMAFORO        # traffic light
11: SENAL_ALTO     # stop sign
```

---

## 10. Máquina de estados de decisión

### 10.1 Estados

1. `INICIALIZANDO`
2. `CONDUCIENDO_NORMAL`
3. `SIGUIENDO_VEHICULO`
4. `FRENANDO_PREVENTIVO`
5. `APROXIMANDO_ALTO`
6. `DETENIDO_ALTO`
7. `APROXIMANDO_SEMAFORO`
8. `DETENIDO_SEMAFORO`
9. `CRUZANDO`
10. `EVALUANDO_REBASE`
11. `REBASANDO`
12. `RECUPERACION`
13. `PARO_EMERGENCIA`

### 10.2 Reglas de decisión (orden de prioridad)

Se evalúan en orden; la primera coincidencia gana.

| # | Condición | Estado destino | Acción | Ref. doc. |
|---|-----------|----------------|--------|-----------|
| 1 | Paro manual o watchdog | `PARO_EMERGENCIA` | `ALTO_TOTAL` | RF-12 |
| 2 | `confianza_percepcion < 0.3` | `RECUPERACION` | `FRENAR_SUAVE` | §1.11 |
| 3 | `peaton_en_riesgo` | `FRENANDO_PREVENTIVO` | `FRENAR_FUERTE` | §1.11 |
| 4 | `semaforo = ROJO` | `DETENIDO_SEMAFORO` | `ALTO_TOTAL` | RF-07 |
| 5 | `semaforo = AMARILLO` con distancia suficiente | `APROXIMANDO_SEMAFORO` | `FRENAR_SUAVE` | §1.11 |
| 6 | `senal_alto_cercana` y `timer_alto < 2s` | `DETENIDO_ALTO` | `ALTO_TOTAL` | RF-07 |
| 7 | Saliendo de alto (`timer_alto ≥ 2s`) y laterales libres | `CRUZANDO` | `ACELERAR` | RF-08 |
| 8 | `frente_cercano_ocupado` y no en rebase | `SIGUIENDO_VEHICULO` | `FRENAR_SUAVE` | §1.11 |
| 9 | `SIGUIENDO_VEHICULO` estable ≥ 3s + espejo_izq libre + lateral_izq libre | `EVALUANDO_REBASE` → `REBASANDO` | `REBASAR_IZQ` | RF-09 |
| 10 | Conflicto lateral durante rebase | abortar → `SIGUIENDO_VEHICULO` | `FRENAR_SUAVE` | §1.11 |
| 11 | `semaforo = VERDE` y frente libre | `CONDUCIENDO_NORMAL` | `ACELERAR` | RF-06 |
| 12 | Default (nada especial) | `CONDUCIENDO_NORMAL` | `MANTENER` | §1.11 |

### 10.3 Principios del FSM

- **Histéresis**: transición requiere N frames consecutivos cumpliendo condición (3 para "ocupado", 5 para "libre").
- **Timers explícitos** con `time.monotonic()`, nunca `sleep`.
- **Rebase exige doble señal**: frente bloqueado + todo lateral libre (5+ frames).
- **Conservador por defecto**: duda → frenar.
- **Logging obligatorio** de transiciones con razón y número de regla.

---

## 11. Estrategia de pruebas por fases

| Fase | Día | Producto | Criterio de aprobación |
|------|-----|----------|------------------------|
| F0: Benchmark | 1 | `scripts/benchmark_fps.py` | ≥ 30 FPS end-to-end con YOLO26n sobre 1920×1080 |
| F1: Detector+Tracker | 2-3 | `probar_deteccion.py` sobre video YouTube | Detecciones visibles y estables >70% frames relevantes |
| F2: Contexto | 4-5 | Overlay con ROI + ocupación | Tests unitarios pasan + inspección visual |
| F3: FSM con `ControladorNulo` | 6-8 | Corrida completa sobre video | 8 escenarios del doc dan acción esperada |
| F4: Captura pantalla + gamepad en ETS2 | 9-10 | Comando aislado en juego real | Acelerar/frenar/girar responden correctamente |
| F5: Sistema completo | 11-13 | Escenarios 1-8 del doc ejecutados en ETS2 | Video de cada caso + métricas |
| F6: Entrega | 14 | Reporte + video demo + presentación | Checklist 2.7 del doc completo |

### 11.1 Tests unitarios mínimos

- `test_contexto.py`: intersección con ROI, umbrales de ocupación, cálculo de confianza.
- `test_decision.py`: una prueba por cada regla (12+ pruebas).
- `test_control.py`: mapa acción→comando, liberar() deja todo en cero.
- `test_semaforo.py`: clasificación correcta de parches HSV conocidos.

### 11.2 Evidencia experimental

- Video MP4 con overlays por cada escenario validado.
- CSV/JSONL de métricas por corrida (FPS, latencia, eventos).
- Análisis comparativo entre `yolov26n` y `yolov26s` (si tiempo lo permite, para sección 2.15).

---

## 12. Riesgos técnicos y mitigaciones

| Riesgo | Prob. | Impacto | Mitigación |
|--------|-------|---------|------------|
| FPS insuficiente | Media | Alto | F0 valida al día 1; si falla, bajar a imgsz=480 o usar nano |
| Modelo pre-entrenado no detecta bien algo del juego | Media | Medio | Fine-tuning dirigido con 200-500 frames etiquetados (día 3-4 si aplica) |
| Clasificación de semáforo falla (color engañoso) | Media | Alto | HSV + umbrales calibrables + fallback a estado DESCONOCIDO |
| `vgamepad` no se instala (driver ViGEmBus) | Baja | Medio | Fallback a `pydirectinput` con teclado |
| ETS2 no reconoce entradas en segundo plano | Baja | Alto | Documentar config de "ventana con foco" en README |
| Falsos positivos en espejos (reflejos) | Alta | Medio | Histéresis + ROI conservadores + `edad ≥ 3` |
| Oscilación en controles | Media | Medio | Suavizado por EMA en `ComandoControl` |
| Pérdida de detección por oclusión | Media | Alto | Estado `RECUPERACION` + `FRENAR_SUAVE` hasta recuperar |
| No terminar a tiempo | Alta | Crítico | Fases con entregables parciales válidos; si F5 no completa, F3 (video) ya es defendible |

---

## 13. Entregables finales (trazables a sección 2.7 del doc)

1. **Código fuente completo** — repositorio local (git) + zip.
2. **README.md** — instalación, configuración, ejecución, troubleshooting.
3. **Modelo(s) entrenado(s)** — `datos/modelos/yolov26n.pt` + fine-tuned si aplica, enlace de descarga.
4. **Reporte técnico** (PDF/Word) — con secciones según 2.8 del documento.
5. **Video demostrativo** — edición de los 8 escenarios con overlays.
6. **Carpeta evidencia** — logs JSONL, CSV de métricas, capturas, videos por corrida.
7. **Presentación final** — slides + demo en vivo (opcional).

---

## 14. Cronograma de 14 días

```
Día 1   F0: setup + benchmark FPS
Día 2-3 F1: detector + tracker sobre video YouTube
Día 4-5 F2: módulo contexto + ROI calibradas
Día 6-8 F3: FSM completo probado con ControladorNulo sobre video
Día 9-10 F4: captura pantalla + gamepad en ETS2 (pruebas aisladas)
Día 11-13 F5: integración total en ETS2 + escenarios 1-8
Día 14  F6: reporte técnico + video demo + presentación
```

Cada fase tiene un "entregable defendible" aunque la siguiente no se complete — si el proyecto se atrasa, F3 (video) ya demuestra funcionalidad.

---

## 15. Criterios de éxito (sección 1.17 del doc)

El proyecto será exitoso si:
1. ✅ Conduce en al menos 6 de 8 escenarios definidos.
2. ✅ Respeta semáforos rojos y señales de alto con ≥ 90 % de cumplimiento.
3. ✅ Reacciona correctamente a vehículos y peatones (sin colisiones evitables).
4. ✅ No modifica el juego.
5. ✅ La lógica de decisión es explicable (trazada en logs).
6. ✅ Entrega evidencia cuantitativa (FPS, latencia, métricas).

---

## 16. Referencias

- Documento de requerimientos oficial: `Proyecto_ETS2_YOLO26_Requerimientos.docx`
- Ultralytics YOLO26 docs: https://docs.ultralytics.com
- DXcam: https://github.com/ra1nty/DXcam
- ViGEmBus + vgamepad: https://github.com/yannbouteiller/vgamepad
- Groover, M. P. — *Automation, Production Systems, and Computer-Integrated Manufacturing*, 4th ed.
- SCS Modding Wiki — Telemetry SDK (solo instrumental, nunca decisorio)
