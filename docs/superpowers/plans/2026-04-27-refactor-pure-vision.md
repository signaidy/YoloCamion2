# Refactor a Arquitectura Pure-Vision de 4 Capas

> **Estado:** PROPUESTA — requiere confirmación del usuario antes de ejecutar.
> Fecha: 2026-04-27. Reemplaza al rechazado `2026-04-27-arquitectura-hibrida-autonoma.RECHAZADO.md`.

**Objetivo:** Llevar el sistema actual (captura + YOLO + FSM + gamepad/teclado) a la arquitectura de 4 capas Pure-Vision: Percepción Visual con TTC, Planificación FSM, Ejecución PID/vgamepad, y Evaluación post-ejecución totalmente sandboxed.

**Restricción dura (RNF-06, RNF-07, regla del catedrático):**
- La lógica de conducción se deriva **exclusivamente de píxeles**.
- Prohibido modificar archivos del juego (descarta `scs-sdk-plugin` DLL).
- La telemetría — si llegara a usarse — solo puede vivir en la Capa 4 como evaluador post-mortem y nunca alimentar al loop.

---

## Estado actual del código (auditoría)

```
Capa 1 Percepción
  ✅ src/fuente/{pantalla,ventana,buffer}.py     dxcam→mss fallback OK
  ✅ src/percepcion/tracker.py                   YOLO.track con ByteTrack (IDs persistentes + edad)
  ✅ src/percepcion/contexto.py                  6 ROI cargadas de regiones_interes.yaml
  ⚠️ src/percepcion/carriles.py                  brillo+Hough día/noche con CLAHE (A DEPRECAR/REEMPLAZAR)
  ✅ src/percepcion/semaforo.py                  clasificador color
  ❌ Inferencia YOLOP / Segmentación             NO EXISTE  ← punto de partida del plan
  ❌ TTC / Optical Flow / Bbox-scaling           NO EXISTE

Capa 2 Planificación
  ✅ src/decision/fsm.py                         12 reglas, histéresis, timers, prioridades
  ✅ src/decision/estado.py                      13 estados FSM (incluye REBASANDO, RECUPERACION, PARO)
  ❌ Disparadores basados en TTC                 NO EXISTE — R3/R8 usan area/intersección, no proximidad temporal

Capa 3 Ejecución
  ✅ src/control/gamepad.py                      vgamepad pasthrough sin PID
  ⚠️ src/control/teclado.py                      pydirectinput PWM — A DEPRECAR
  ❌ src/control/pid.py                          NO EXISTE
  ❌ src/control/pure_pursuit.py                 NO EXISTE
  ❌ Map Acción→ComandoControl                   _MAPA_ACCION en ejecutar_piloto.py con valores fijos
                                                 (debe ser setpoint para PID, no salida directa)

Capa 4 Evaluación
  ✅ src/registro/{logger,metricas,grabador}.py  log JSONL + métricas + video con overlay
  ❌ Wrapper SCS SDK sandbox                     decisión D1 pendiente (ver Fase 0)
```

---

## FASE 0 — Decisiones tomadas (2026-04-27)

### D1 — RESUELTA: D1.a (recomendada)
Capa 4 evaluación = OCR del velocímetro sobre video grabado + JSONL. Sin SCS SDK, sin DLL, sin tocar archivos del juego. 100 % visión coherente con RNF-07.

### D2 — RESUELTA: mantener `imgsz = 640` por ahora
Esperar al benchmark FPS de la Tarea 1.6 antes de evaluar subir a 960.

### D3 — RESUELTA: NO borrar `src/control/teclado.py`
Conservar como fallback. En la Tarea 3.5 solo se marca como DEPRECATED (docstring + WARN al iniciar), sin eliminar el archivo.

### D4 — RESUELTA (2026-04-28): Migrar Percepción a YOLOP + Volante a Pure Pursuit
El control del camión en curvas se mejorará adoptando un modelo panóptico (YOLOP) para obtener la segmentación del área manejable (Drivable Area) y líneas de carril, reemplazando el OpenCV clásico. Esto se combinará con un control *Pure Pursuit* (mirada a futuro) para un giro suave del volante.

---

## FASE 1 — Capa de Percepción: YOLOP y Física Visual (TTC + Flujo Óptico)

Punto de partida modificado a petición del usuario para incluir anticipación en curvas.

### Tarea 1.0 — Inferencia Panóptica con YOLOP

**Archivos nuevos:**
- `src/percepcion/yolop_inference.py`
- `tests/test_yolop.py`

**Lógica:**
- Descargar o cargar el modelo pre-entrenado de YOLOP (ONNX recomendado por velocidad).
- Procesar cada frame para extraer las tres salidas: `vehiculos` (Bounding Boxes), `area_manejable` (segmentación) y `mascara_carriles`.
- Reemplazar el uso del detector clásico de OpenCV por las salidas de YOLOP en el pipeline de percepción.

### Tarea 1.1 — Tipos para Física Visual

**Archivos:**
- Modificar: `src/tipos.py`
- Modificar: `tests/test_tipos.py`

**Cambios:**
- Nuevo dataclass `FisicaVisual`:
  ```
  velocidad_relativa_px_s: float   # crecimiento del bbox o flujo en píxeles/seg
  ttc_segundos: float              # tiempo a colisión visual; +inf si separación
  area_px: int
  area_anterior_px: int
  centroide: tuple[int, int]
  vector_flujo: tuple[float, float]  # promedio del flujo dentro del bbox
  ```
- Nuevo campo opcional en `Seguimiento`: `fisica: Optional[FisicaVisual] = None`.
- Nuevos campos en `EstadoEscena`:
  ```
  ttc_minimo_frente_s: float = float("inf")
  vehiculo_critico_id: Optional[int] = None
  ```

**Tests (TDD):** valores por defecto, no rompe construcción de `EstadoEscena` existente.

### Tarea 1.2 — Estimador TTC por escalado de bounding box

**Archivos nuevos:**
- `src/percepcion/fisica.py`
- `tests/test_fisica.py`

**Lógica (resumen):**
- Para cada `Seguimiento` con `id_seguimiento` ya conocido, almacenar serie de `(timestamp, area_px, centroide)` con `deque(maxlen=N)` (N≈8).
- Velocidad relativa de aproximación visual ≈ `d(area)/dt` normalizada por el área media.
- TTC ≈ `area / max(d(area)/dt, ε)` cuando crece; `+inf` cuando decrece (se aleja).
- Filtrar saltos espurios (cambio de id, oclusión) descartando muestras con `dt > 0.5 s` o `Δarea/area > 1.5`.

**Tests (TDD):**
- Bbox que duplica su área en 1 s → TTC ≈ 1 s.
- Bbox que se mantiene constante → TTC = +inf.
- Bbox que decrece → TTC = +inf.
- ID nuevo sin historial → TTC = +inf, no lanza excepción.

### Tarea 1.3 — Flujo óptico denso restringido al ROI frontal

**Archivos nuevos:**
- `src/percepcion/flujo_optico.py`
- `tests/test_flujo_optico.py`

**Lógica:**
- Capturar `gris_anterior` del frame previo.
- Calcular `cv2.calcOpticalFlowFarneback` o `DISOpticalFlow` (más rápido) **solo dentro del ROI frente_cercano + frente_lejano** para no comer FPS.
- Para cada `Seguimiento` cuya bbox cae en esos ROI, promediar el vector de flujo dentro de la bbox → `FisicaVisual.vector_flujo`.
- Combinar con la estimación por área: si `vector_flujo` apunta hacia el centro de la imagen (o sea, el objeto se queda quieto en pantalla mientras el ego avanza) → confianza extra de "se aleja". Si crece y el flujo apunta hacia afuera del centro → "se aproxima".
- Decisión final: `velocidad_relativa = max(velocidad_por_area, velocidad_por_flujo)`.

**Tests (TDD):** con frames sintéticos (un cuadrado que crece y se mueve hacia abajo), comprobar signo y magnitud aproximada.

**Performance:** medir el costo del cálculo en `scripts/benchmark_fps.py` antes y después de integrarlo. Presupuesto: ≤ 10 ms por frame en GPU integrada típica. Si excede, bajar a flujo disperso (`cv2.calcOpticalFlowPyrLK`) sobre puntos Shi-Tomasi dentro de cada bbox.

### Tarea 1.4 — Integrar TTC en `AnalizadorContexto`

**Archivos:**
- Modificar: `src/percepcion/contexto.py`
- Modificar: `tests/test_contexto.py`

**Cambios:**
- `AnalizadorContexto.__init__` recibe `estimador_fisica: EstimadorFisicaVisual`.
- En `analizar()`, después de procesar seguimientos:
  - Llamar `estimador_fisica.actualizar(seguimientos, timestamp, imagen_gris)`.
  - Para cada `seg.fisica`, si está en ROI frente_cercano o frente_lejano y TTC < `ttc_minimo`, actualizar `ttc_minimo_frente_s` y `vehiculo_critico_id`.
- Retornar `EstadoEscena` con los nuevos campos poblados.

**Tests (TDD):**
- Escena con un solo vehículo en frente_cercano cuya área crece de 5000→8000 px² en 0.3 s → `ttc_minimo_frente_s` debe estar entre 0.4 y 1.0 s.
- Escena vacía → `ttc_minimo_frente_s = +inf`, `vehiculo_critico_id = None`.

### Tarea 1.5 — Smoke test integrado en video grabado

**Archivos:**
- Nuevo: `scripts/probar_ttc.py`
- Reutilizar: video existente `datos/videos/ets2_volvo_fh16.f299.mp4` y/o sesiones grabadas.

Recorre un video, dibuja sobre cada bbox: `id`, `area`, `TTC`, vector flujo. Genera salida MP4 anotada en `datos/evidencia/ttc_<sesion>.mp4` para validación visual sin necesidad de ETS2 abierto.

### Tarea 1.6 — Benchmark FPS con TTC activo

Correr `scripts/benchmark_fps.py` antes y después. Aceptable si la pérdida de FPS es ≤ 25 %; si no, optimizar (Tarea 1.3 nota de performance).

---

## FASE 2 — Capa de Planificación: FSM con disparadores TTC

### Tarea 2.1 — Nuevas constantes y reglas TTC

**Archivos:**
- Modificar: `src/decision/fsm.py`
- Modificar: `tests/test_decision.py`

**Cambios:**
- Constantes nuevas (calibrables luego):
  ```
  _TTC_FRENO_FUERTE = 1.5   # s — TTC < 1.5s en frente_cercano → frenar fuerte
  _TTC_FRENO_SUAVE  = 3.0   # s — TTC entre 1.5 y 3.0 → frenar suave
  _TTC_REBASE_OK    = 4.0   # s — TTC frontal > 4s + condiciones → permitir rebase
  ```
- Nuevas reglas, insertadas con prioridades correctas dentro del orden actual:
  - **R3.5** (entre peatón y semáforo): si `escena.ttc_minimo_frente_s < _TTC_FRENO_FUERTE` → `FRENAR_FUERTE` + estado `FRENANDO_PREVENTIVO`.
  - **R8b** (refina R8): si `frente_cercano_ocupado` Y `ttc_minimo_frente_s < _TTC_FRENO_SUAVE` → `FRENAR_SUAVE`. Si TTC ≥ umbral pero está ocupado → `MANTENER` (siguiendo a velocidad similar).
  - **R9b** (refina R9 rebase): añade condición `ttc_minimo_frente_s < _TTC_REBASE_OK` al gating actual; un TTC alto significa que el frente avanza igual o más rápido — no rebasar.

**Tests (TDD):** cada regla nueva tiene su test con `EstadoEscena` mockeado.

### Tarea 2.2 — Salida del FSM como setpoint, no como acción rígida

**Archivos:**
- Modificar: `src/decision/fsm.py` → `ResultadoDecision` añade `setpoint: SetpointControl`.
- Nuevo dataclass en `src/tipos.py`: `SetpointControl(velocidad_objetivo_norm: float, freno_objetivo: float, desviacion_volante: float)`.
- Modificar: `scripts/ejecutar_piloto.py` → eliminar `_MAPA_ACCION` rígido, generar `SetpointControl` a partir de la `Accion` (o la propia `ResultadoDecision` lo trae ya construido).

**Razón:** la Capa 3 (PID) necesita un objetivo continuo, no un valor instantáneo.

---

## FASE 3 — Capa de Ejecución: PID + vgamepad analógico

### Tarea 3.1 — `PIDController` genérico con anti-windup

**Archivos nuevos:**
- `src/control/pid.py`
- `tests/test_pid.py`

API mínima:
```
PIDController(kp, ki, kd, limite=1.0)
.calcular(setpoint, medicion, dt) -> float ∈ [-limite, +limite]
.reset()
```

Tests TDD: respuesta proporcional pura, saturación min/max, anti-windup por clamping, reset.

### Tarea 3.1.5 — Controlador `PurePursuit` (Punto de Anticipación)

**Archivos nuevos:**
- `src/control/pure_pursuit.py`
- `tests/test_pure_pursuit.py`

**Lógica:**
- Toma la máscara de `area_manejable` o `mascara_carriles` de YOLOP.
- Calcula el "Look-ahead point" (Punto de anticipación) a cierta distancia frente al camión.
- Calcula el ángulo de giro necesario para que la nariz del camión apunte a ese punto.

### Tarea 3.2 — `ControladorGamepadPID` (reemplaza el actual `gamepad.py`)

**Archivos:**
- Modificar: `src/control/gamepad.py` → renombrar internamente o crear `src/control/gamepad_pid.py` y mantener el viejo como fallback `gamepad_directo.py`.
- Test: `tests/test_gamepad_pid.py` con `vgamepad.VX360Gamepad` mockeado.

Tres PIDs / Controladores:
- `control_volante`: Usa el `PurePursuit` controller integrado con un PID suave para generar el comando del joystick izquierdo hacia el punto de fuga. Salida → `left_joystick_float.x_value_float`.
- `pid_velocidad`: setpoint = `velocidad_objetivo_norm` (0–1), medición = **velocidad estimada visualmente** (ver Tarea 3.3). Salida positiva → `right_trigger`; negativa → `left_trigger` con coeficiente menor (freno suave).
- Freno de emergencia (`freno_objetivo ≥ 0.9`): bypass PID, `left_trigger` directo a 255 y reset de los integrales.

**Importante:** la "medición de velocidad" del PID de velocidad **no puede venir de telemetría** (RNF-07). Tarea 3.3 la deriva visualmente.

### Tarea 3.3 — Estimador de velocidad propia por flujo óptico (visual)

**Archivos nuevos:**
- `src/percepcion/velocidad_propia.py`
- `tests/test_velocidad_propia.py`

**Lógica:**
- Tomar el flujo óptico ya calculado en Tarea 1.3, restringido a una franja inferior central de la imagen (asfalto del propio carril, fuera del capó).
- La magnitud media del flujo en esa franja es proporcional a la velocidad del ego respecto al asfalto.
- Calibrar el factor multiplicador con dos pasadas conocidas en ETS2 (40 km/h y 80 km/h en recta) → guardar en `config/default.yaml`.
- Salida normalizada [0–1] donde 1 ≡ velocidad máxima fijada (p. ej. 90 km/h).

**Justificación pure-vision:** un humano ve el asfalto pasar y juzga la velocidad — esto es exactamente eso.

**Tests:** con frames sintéticos donde el flujo crece linealmente, la salida también crece linealmente.

### Tarea 3.4 — Integrar PID + velocidad propia en `ejecutar_piloto.py`

**Archivos:**
- Modificar: `scripts/ejecutar_piloto.py`.

Cambios:
- Construir `EstimadorFlujoOptico`, `EstimadorVelocidadPropia`, `EstimadorFisicaVisual` y conectarlos al loop.
- Cada frame: medir velocidad propia visual → pasarla al `ControladorGamepadPID.actualizar_velocidad_actual()`.
- Para `--control gamepad`, instanciar `ControladorGamepadPID`. Para `--control teclado` mantener actual con WARN de DEPRECATED.

### Tarea 3.5 — Marcar `ControladorTeclado` como DEPRECATED

Añadir docstring de cabecera + WARN al `iniciar()`. Sigue funcional como fallback (decisión D3).

---

## FASE 4 — Capa de Evaluación (resuelta por D1)

### Si D1.a (recomendado, 100 % visión)

**Archivos nuevos:**
- `scripts/evaluar_sesion.py` — toma `sesion_<ts>.jsonl` + `sesion_<ts>.mp4`, recorre el video con OCR (`pytesseract` o `easyocr`) sobre el ROI del velocímetro y odómetro del HUD, cruza con los timestamps del JSONL.
- `src/evaluacion/ocr_velocimetro.py` — clasifica los dígitos del HUD del Volvo FH16.
- `tests/test_ocr_velocimetro.py` — con frames recortados conocidos.

**Salida:** `datos/evidencia/reporte_<sesion>.csv` con columnas `t, regla_fsm, accion, velocidad_kmh_ocr, evento_seguridad`.

**Coherencia con RNF-07:** el ground-truth se obtiene también por visión (OCR del HUD), nunca por API interna.

### Si D1.b (no recomendado)

Entonces se construye `src/evaluacion/sandbox_telemetria.py` que abre la shared memory desde un proceso **separado** que escribe a archivo, sin retroalimentar el loop. Pero requiere instalar la DLL en una copia del juego — confirmación explícita del usuario obligatoria.

---

## FASE 5 — Validación end-to-end y calibración

### Tarea 5.1 — Suite de regresión completa

`pytest tests/ -v` debe seguir pasando. Documentar nuevo conteo esperado (≈ tests actuales + 25–30).

### Tarea 5.2 — Sesiones de calibración PID

Protocolo en pista recta + intersección + tráfico ligero, sin video grabado para máximo FPS:
1. Ajustar `kp` del PID de volante hasta que siga la línea sin oscilar.
2. Ajustar `kd` para amortiguar.
3. Ajustar `kp/ki` del PID de velocidad para alcanzar el setpoint sin sobreimpulso.
4. Guardar valores en `config/default.yaml` sección `pid:`.

### Tarea 5.3 — Sesiones de validación de los 8 escenarios obligatorios del pliego (sección 2.9)

Una sesión por escenario, video + JSONL guardados como evidencia.

---

## Mapa de cumplimiento RF/RNF

| Requisito | Cubierto por |
|---|---|
| RF-01 captura tiempo real | `src/fuente/pantalla.py` (sin cambios) |
| RF-02 cámara primera persona | configuración en juego (operacional) |
| RF-03 YOLO26 | **Migrado 2026-04-27**: `datos/modelos/yolo26n.pt`, ultralytics 8.4.41. Mismas 80 clases COCO → mapeo `_COCO_A_CLASE` no cambia. |
| RF-04 vehículos/peatones/semáforos/altos | tracker + clasificador semáforo (sin cambios) |
| RF-05 frente + retrovisores + laterales | ROI ya particionadas; TTC se evalúa en frente |
| RF-06 acelerar/frenar | PID velocidad (Fase 3) |
| RF-07 alto/rojo | reglas R4/R6 ya existen |
| RF-08 cruce con verificación lateral | R7 ya existe; refinar con TTC en Fase 2 |
| RF-09 rebase con seguridad lateral | R9b refinada con TTC (Fase 2.1) |
| RF-10 emulación externa | vgamepad (Fase 3) |
| RF-11 logs + video | logger/grabador existentes (sin cambios) |
| RF-12 paro manual | `MonitorSeguridad` ya existe (sin cambios) |
| RNF-06 sin modificar juego | garantizado por elección D1.a |
| RNF-07 fuente principal visual | velocidad propia visual (Fase 3.3), TTC visual (Fase 1), evaluación visual (Fase 4 D1.a) |

---

## Resumen de archivos tocados

**Nuevos:**
- `src/percepcion/fisica.py`, `src/percepcion/flujo_optico.py`, `src/percepcion/velocidad_propia.py`
- `src/control/pid.py`, `src/control/gamepad_pid.py`
- `src/evaluacion/__init__.py`, `src/evaluacion/ocr_velocimetro.py`
- `scripts/probar_ttc.py`, `scripts/evaluar_sesion.py`
- Tests correspondientes en `tests/`.

**Modificados:**
- `src/tipos.py` (FisicaVisual, SetpointControl, campos en Seguimiento/EstadoEscena)
- `src/percepcion/contexto.py` (integra estimador de física)
- `src/decision/fsm.py` (R3.5, R8b, R9b)
- `scripts/ejecutar_piloto.py` (instancia nuevos componentes, ya no usa `_MAPA_ACCION` rígido)
- `config/default.yaml` (sección `pid`, `ttc`, `velocidad_propia.calibracion`)

**Deprecados (mantener):**
- `src/control/teclado.py` (marca DEPRECATED)
- `src/control/gamepad.py` (renombrar a `gamepad_directo.py` o mantener como modo fallback sin PID)

---

## Orden de ejecución recomendado

```
Fase 0 confirmar → Fase 1 (1.1 → 1.6) → Fase 2 (2.1 → 2.2) → Fase 3 (3.1 → 3.5) → Fase 4 → Fase 5
```

Cada tarea respeta TDD: test que falla → implementación → test pasa → commit. Nada se commitea sin que `pytest tests/` quede verde.
