# [RECHAZADO 2026-04-27] Arquitectura Híbrida Autónoma ETS2 — Plan de Implementación

> ⛔ **PLAN RECHAZADO — NO EJECUTAR.**
>
> **Razón del rechazo:** este plan exige instalar `scs-sdk-plugin` (`scs-telemetry.dll`) dentro de la carpeta `bin\win_x64\plugins\` del juego ETS2 (ver Tarea 1, Paso 2). Eso modifica los archivos del juego, lo cual el catedrático prohibió expresamente y los requerimientos formales también:
>
> - Sección 1.6: "No se permite modificar el código del juego" / "La interacción con el juego debe ser externa y transparente para ETS2".
> - RNF-06: "El sistema debe funcionar sin modificar archivos del juego ni inyectar código en su proceso".
> - RNF-07: "La fuente principal de decisión debe ser visual; si usan telemetría para evaluación, deben demostrar que no interviene en la lógica de conducción". Este plan vuelve la telemetría la "fuente de verdad" del bucle de control, lo que también viola el requerimiento.
>
> Conservado solo como evidencia histórica del descarte. Cualquier replanificación debe basarse exclusivamente en visión + emulación externa de controles.

> **Para workers agénticos:** SKILL REQUERIDO: usar `superpowers:subagent-driven-development` (recomendado) o `superpowers:executing-plans` para ejecutar este plan tarea por tarea. Los pasos usan sintaxis de checkbox (`- [ ]`) para seguimiento.

**Objetivo:** Refactorizar el sistema de conducción autónoma de "visión pura + teclado digital" a un sistema híbrido de 4 capas: percepción visual (YOLO/OpenCV) + telemetría SCS SDK + FSM supervisor + control analógico PID/vgamepad.

**Arquitectura:** La telemetría ETS2 (velocidad, RPM, ángulo de dirección real) se convierte en la fuente de verdad para el bucle de control. El FSM combina detecciones YOLO con datos de telemetría para decisiones supervisadas. Tres controladores PID independientes (dirección, velocidad, frenado) generan salidas analógicas flotantes a vgamepad exclusivamente.

**Tech Stack:** Python 3.11, vgamepad (ViGEmBus), scs-sdk-plugin DLL (RenCloud v12), ctypes/mmap Windows shared memory, OpenCV, Ultralytics YOLO11n, pytest.

**Spec de referencia:** `docs/superpowers/specs/2026-04-23-conduccion-autonoma-ets2-design.md`

**Directorio del proyecto:** `C:\Users\andre\OneDrive - Universidad del Istmo\Desktop\Proyecto Final`

---

## Mapa de archivos

### Archivos NUEVOS

| Archivo | Responsabilidad |
|---------|-----------------|
| `src/telemetria/__init__.py` | Exports del módulo |
| `src/telemetria/tipos_shm.py` | Struct ctypes mapeando SCS SDK v12 shared memory |
| `src/telemetria/lector.py` | Abre `Local\SCSTelemetry`, lee y retorna `TelemetriaSnapshot` |
| `src/telemetria/mock.py` | `TelemetriaLectorMock` para tests sin ETS2 |
| `src/control/pid.py` | `PIDController` genérico con anti-windup |
| `src/control/gamepad_pid.py` | `ControladorGamepadPID`: 3 PIDs → vgamepad analógico |
| `src/decision/supervisor.py` | Reglas híbridas visión+telemetría que sobreescriben FSM |
| `tests/test_pid.py` | Tests del PID: respuesta proporcional, anti-windup, reset |
| `tests/test_telemetria.py` | Tests del lector con mock |
| `tests/test_supervisor.py` | Tests del supervisor con EstadoEscena+Telemetría mock |

### Archivos MODIFICADOS

| Archivo | Cambio |
|---------|--------|
| `src/tipos.py` | + `TelemetriaSnapshot`, + campo `telemetria: Optional[TelemetriaSnapshot]` en `EstadoEscena` |
| `src/decision/fsm.py` | Consume `escena.telemetria` cuando disponible para R8/R9 (distancia real) |
| `scripts/ejecutar_piloto.py` | Instancia `TelemetriaLector` + `ControladorGamepadPID`, los conecta al loop |
| `config/default.yaml` | + sección `telemetria`, + sección `pid` con Kp/Ki/Kd por eje |

### Archivos DEPRECADOS (no eliminar, marcar)

| Archivo | Estado |
|---------|--------|
| `src/control/teclado.py` | Marcar como `# DEPRECATED: usar ControladorGamepadPID` |
| `src/control/nulo.py` | Conservar solo para tests sin gamepad |

---

## FASE 0 — Instalación del plugin SCS SDK (prerequisito manual)

### Tarea 1: Instalar scs-sdk-plugin en ETS2

> ⚠️ Este paso es manual. Sin el plugin, la telemetría no existe.

- [ ] **Paso 1: Descargar el plugin**

Ir a: https://github.com/RenCloud/scs-sdk-plugin/releases

Descargar `scs-sdk-plugin.zip` (versión ≥ 12). Contiene:
- `win_x64/scs-telemetry.dll`

- [ ] **Paso 2: Instalar el plugin en ETS2**

Copiar `scs-telemetry.dll` a:
```
C:\Users\andre\OneDrive - Universidad del Istmo\Desktop\Euro.Truck.Sim.2.v1.58.1.4s.ALL.DLC\Euro.Truck.Sim.2.v1.58.1.4s.ALL.DLC\bin\win_x64\plugins\scs-telemetry.dll
```

Si la carpeta `plugins\` no existe, crearla.

- [ ] **Paso 3: Reiniciar ETS2 y verificar**

Abrir ETS2, cargar una partida. El plugin activo crea la shared memory automáticamente.

Verificar con este script (requiere venv activo):

```python
# scripts/verificar_telemetria.py
import ctypes
import ctypes.wintypes

MEM_NAME = "Local\\SCSTelemetry"
FILE_MAP_READ = 0x0004

handle = ctypes.windll.kernel32.OpenFileMappingW(FILE_MAP_READ, False, MEM_NAME)
if not handle:
    print("ERROR: shared memory no encontrada. Plugin no instalado o juego no cargado.")
else:
    ctypes.windll.kernel32.CloseHandle(handle)
    print("OK: SCSTelemetry shared memory disponible. Plugin activo.")
```

```bash
python scripts/verificar_telemetria.py
```

Esperado:
```
OK: SCSTelemetry shared memory disponible. Plugin activo.
```

---

## FASE 1 — Capa de Telemetría

### Tarea 2: Definir tipos de telemetría en `src/tipos.py`

**Archivos:**
- Modificar: `src/tipos.py`
- Test: `tests/test_tipos.py` (agregar)

- [ ] **Paso 1: Escribir test**

Agregar al final de `tests/test_tipos.py`:

```python
def test_telemetria_snapshot_valores_por_defecto():
    from src.tipos import TelemetriaSnapshot
    t = TelemetriaSnapshot()
    assert t.velocidad_ms == 0.0
    assert t.rpm == 0.0
    assert not t.motor_encendido
    assert not t.remolque_adjunto


def test_estado_escena_acepta_telemetria_opcional():
    from src.tipos import EstadoEscena, TelemetriaSnapshot
    import time
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
        timestamp=time.monotonic(),
        telemetria=TelemetriaSnapshot(velocidad_ms=22.2, rpm=1800.0),
    )
    assert estado.telemetria.velocidad_ms == pytest.approx(22.2)
```

- [ ] **Paso 2: Ejecutar para ver fallo**

```bash
pytest tests/test_tipos.py -v -k "telemetria"
```

Esperado: `FAILED — ImportError: cannot import name 'TelemetriaSnapshot'`

- [ ] **Paso 3: Implementar en `src/tipos.py`**

Agregar al final de `src/tipos.py`, antes de `ComandoControl`:

```python
@dataclass
class TelemetriaSnapshot:
    """Datos de telemetría en tiempo real desde el SCS SDK de ETS2."""
    velocidad_ms: float = 0.0          # m/s  (dividir entre 3.6 para km/h)
    rpm: float = 0.0                    # RPM del motor
    marcha: int = 0                     # -1=retro, 0=neutro, 1..N=marcha
    angulo_volante: float = 0.0         # -1.0 (izq) a +1.0 (der), entrada del jugador
    acelerador_input: float = 0.0       # 0.0-1.0 entrada jugador/piloto
    freno_input: float = 0.0            # 0.0-1.0
    motor_encendido: bool = False
    freno_mano: bool = False
    remolque_adjunto: bool = False
    masa_remolque_kg: float = 0.0
    timestamp: float = 0.0
```

Modificar `EstadoEscena` añadiendo al final el campo:

```python
    telemetria: Optional["TelemetriaSnapshot"] = None
```

El dataclass `EstadoEscena` ya existe — solo agregar ese campo con valor por defecto `None` al final para no romper tests existentes.

- [ ] **Paso 4: Ejecutar todos los tests**

```bash
pytest tests/ -v
```

Esperado: todos los tests anteriores + los 2 nuevos de telemetría = PASS

- [ ] **Paso 5: Commit**

```bash
git add src/tipos.py tests/test_tipos.py
git commit -m "feat: TelemetriaSnapshot dataclass y campo opcional en EstadoEscena"
```

---

### Tarea 3: Struct ctypes de la shared memory SCS SDK v12

**Archivos:**
- Crear: `src/telemetria/__init__.py`
- Crear: `src/telemetria/tipos_shm.py`

- [ ] **Paso 1: Crear `src/telemetria/__init__.py`**

```python
from src.telemetria.lector import TelemetriaLector
from src.telemetria.mock import TelemetriaLectorMock

__all__ = ["TelemetriaLector", "TelemetriaLectorMock"]
```

(Nota: `lector.py` se crea en la Tarea 4)

- [ ] **Paso 2: Crear `src/telemetria/tipos_shm.py`**

```python
"""Mapa ctypes del bloque de shared memory expuesto por scs-sdk-plugin v12.

La estructura completa tiene ~3 KB. Solo mapeamos los campos necesarios
para el control autónomo. Los campos no usados se rellenan con padding.
Referencia: https://github.com/RenCloud/scs-sdk-plugin/blob/master/scs-telemetry/inc/scs-telemetry-common.hpp
"""
import ctypes


class _Pad(ctypes.Structure):
    """Relleno de bytes para alinear campos no usados."""
    pass


# Tamaños de bloque de la estructura v12
_SHM_SIZE = 65536  # 64 KB reservados por el plugin


class SCSVec3(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]


class SCSEuler(ctypes.Structure):
    _fields_ = [
        ("heading", ctypes.c_float),
        ("pitch",   ctypes.c_float),
        ("roll",    ctypes.c_float),
    ]


class SCSTruckData(ctypes.Structure):
    """Datos del camión en la shared memory (offsets v12)."""
    _fields_ = [
        # Header
        ("scs_sdk_plugin_revision",  ctypes.c_uint32),   # 0
        ("game_version_major",       ctypes.c_uint32),   # 4
        ("game_version_minor",       ctypes.c_uint32),   # 8
        ("game_paused",              ctypes.c_bool),     # 12
        ("_pad0",                    ctypes.c_uint8 * 3),

        # Tiempo
        ("local_time",               ctypes.c_float),    # 16
        ("game_time",                ctypes.c_float),    # 20

        # Telemetría del camión
        ("truck_speed",              ctypes.c_float),    # 24  m/s
        ("truck_engine_rpm",         ctypes.c_float),    # 28
        ("truck_engine_enabled",     ctypes.c_bool),     # 32
        ("truck_electric_enabled",   ctypes.c_bool),     # 33
        ("_pad1",                    ctypes.c_uint8 * 2),

        ("truck_gear",               ctypes.c_int32),    # 36  -1=R 0=N 1..N=D
        ("truck_displayed_gear",     ctypes.c_int32),    # 40

        # Inputs
        ("truck_input_steering",     ctypes.c_float),    # 44  -1..+1
        ("truck_input_throttle",     ctypes.c_float),    # 48   0..1
        ("truck_input_brake",        ctypes.c_float),    # 52   0..1
        ("truck_input_clutch",       ctypes.c_float),    # 56   0..1

        # Effective (lo que el motor realmente aplica)
        ("truck_eff_steering",       ctypes.c_float),    # 60
        ("truck_eff_throttle",       ctypes.c_float),    # 64
        ("truck_eff_brake",          ctypes.c_float),    # 68
        ("truck_eff_clutch",         ctypes.c_float),    # 72

        # Freno de mano
        ("truck_parking_brake",      ctypes.c_bool),     # 76
        ("_pad2",                    ctypes.c_uint8 * 3),

        # Combustible
        ("truck_fuel",               ctypes.c_float),    # 80
        ("truck_fuel_range",         ctypes.c_float),    # 84

        # Desgaste
        ("truck_wear_engine",        ctypes.c_float),    # 88
        ("_pad3",                    ctypes.c_uint8 * 36),

        # Posición y orientación (128)
        ("truck_position",           SCSVec3),           # 128
        ("truck_orientation",        SCSEuler),          # 140
        ("truck_acceleration",       SCSVec3),           # 152

        # Trailer
        ("_pad4",                    ctypes.c_uint8 * 68),
        ("trailer_attached",         ctypes.c_bool),     # 232
        ("_pad5",                    ctypes.c_uint8 * 3),
        ("trailer_mass",             ctypes.c_float),    # 236
    ]
```

- [ ] **Paso 3: Verificar tamaño mínimo del struct**

```bash
python -c "
import sys; sys.path.insert(0,'.')
from src.telemetria.tipos_shm import SCSTruckData
print('Tamaño struct:', ctypes.sizeof(SCSTruckData), 'bytes')
import ctypes
print('OK si <= 65536')
"
```

Esperado: `Tamaño struct: 240 bytes` (o similar, < 65536)

- [ ] **Paso 4: Commit**

```bash
git add src/telemetria/__init__.py src/telemetria/tipos_shm.py
git commit -m "feat: struct ctypes SCS SDK v12 para shared memory ETS2"
```

---

### Tarea 4: Lector de telemetría con shared memory

**Archivos:**
- Crear: `src/telemetria/lector.py`
- Crear: `src/telemetria/mock.py`
- Crear: `tests/test_telemetria.py`

- [ ] **Paso 1: Crear `src/telemetria/mock.py`**

```python
"""Mock de TelemetriaLector para tests y modos sin ETS2."""
import time
from src.tipos import TelemetriaSnapshot


class TelemetriaLectorMock:
    """Devuelve snapshots sintéticos configurables. Para tests y ControladorNulo."""

    def __init__(self, **kwargs_snapshot):
        self._defaults = kwargs_snapshot

    def leer(self) -> TelemetriaSnapshot:
        return TelemetriaSnapshot(
            timestamp=time.monotonic(),
            **self._defaults,
        )

    def cerrar(self) -> None:
        pass
```

- [ ] **Paso 2: Escribir tests**

Crear `tests/test_telemetria.py`:

```python
import time
import pytest
from src.tipos import TelemetriaSnapshot
from src.telemetria.mock import TelemetriaLectorMock


def test_mock_devuelve_snapshot_con_valores_configurados():
    mock = TelemetriaLectorMock(velocidad_ms=25.0, rpm=1500.0, motor_encendido=True)
    snap = mock.leer()
    assert snap.velocidad_ms == pytest.approx(25.0)
    assert snap.rpm == pytest.approx(1500.0)
    assert snap.motor_encendido is True


def test_mock_timestamp_reciente():
    mock = TelemetriaLectorMock()
    t_antes = time.monotonic()
    snap = mock.leer()
    assert snap.timestamp >= t_antes


def test_mock_valores_por_defecto_son_cero():
    mock = TelemetriaLectorMock()
    snap = mock.leer()
    assert snap.velocidad_ms == pytest.approx(0.0)
    assert snap.remolque_adjunto is False


def test_mock_cerrar_no_lanza_excepcion():
    mock = TelemetriaLectorMock()
    mock.cerrar()  # no debe lanzar


def test_lector_real_importa_sin_error():
    # Solo verificar que el módulo importa correctamente
    from src.telemetria.lector import TelemetriaLector
    assert TelemetriaLector is not None
```

- [ ] **Paso 3: Ejecutar para ver fallo**

```bash
pytest tests/test_telemetria.py -v
```

Esperado: `FAILED — ImportError: cannot import name 'TelemetriaLector'`

- [ ] **Paso 4: Crear `src/telemetria/lector.py`**

```python
"""Lector de telemetría ETS2 vía Windows shared memory (scs-sdk-plugin v12).

Uso:
    lector = TelemetriaLector()
    lector.abrir()                    # falla si ETS2 no tiene el plugin
    snap = lector.leer()              # TelemetriaSnapshot actualizado
    lector.cerrar()

Si ETS2 no está disponible, usar TelemetriaLectorMock en su lugar.
"""
import ctypes
import ctypes.wintypes
import logging
import time
from typing import Optional

from src.tipos import TelemetriaSnapshot
from src.telemetria.tipos_shm import SCSTruckData, _SHM_SIZE

logger = logging.getLogger(__name__)

_MEM_NAME     = "Local\\SCSTelemetry"
_FILE_MAP_READ = 0x0004


class TelemetriaLector:
    """Abre la shared memory del scs-sdk-plugin y lee TelemetriaSnapshot."""

    def __init__(self):
        self._handle = None
        self._addr   = None
        self._abierto = False

    def abrir(self) -> None:
        kernel32 = ctypes.windll.kernel32
        self._handle = kernel32.OpenFileMappingW(
            _FILE_MAP_READ, False, _MEM_NAME
        )
        if not self._handle:
            raise RuntimeError(
                "No se pudo abrir SCSTelemetry shared memory. "
                "Verifica que scs-sdk-plugin está instalado y ETS2 con partida cargada."
            )
        self._addr = kernel32.MapViewOfFile(
            self._handle, _FILE_MAP_READ, 0, 0, _SHM_SIZE
        )
        if not self._addr:
            kernel32.CloseHandle(self._handle)
            raise RuntimeError("MapViewOfFile falló — memoria insuficiente.")
        self._abierto = True
        logger.info("TelemetriaLector: shared memory abierta (%s)", _MEM_NAME)

    def leer(self) -> TelemetriaSnapshot:
        if not self._abierto or not self._addr:
            return TelemetriaSnapshot(timestamp=time.monotonic())

        datos = ctypes.cast(self._addr, ctypes.POINTER(SCSTruckData))[0]

        return TelemetriaSnapshot(
            velocidad_ms       = float(datos.truck_speed),
            rpm                = float(datos.truck_engine_rpm),
            marcha             = int(datos.truck_gear),
            angulo_volante     = float(datos.truck_input_steering),
            acelerador_input   = float(datos.truck_input_throttle),
            freno_input        = float(datos.truck_input_brake),
            motor_encendido    = bool(datos.truck_engine_enabled),
            freno_mano         = bool(datos.truck_parking_brake),
            remolque_adjunto   = bool(datos.trailer_attached),
            masa_remolque_kg   = float(datos.trailer_mass),
            timestamp          = time.monotonic(),
        )

    def cerrar(self) -> None:
        kernel32 = ctypes.windll.kernel32
        if self._addr:
            kernel32.UnmapViewOfFile(self._addr)
            self._addr = None
        if self._handle:
            kernel32.CloseHandle(self._handle)
            self._handle = None
        self._abierto = False
        logger.info("TelemetriaLector: shared memory cerrada")
```

- [ ] **Paso 5: Ejecutar tests**

```bash
pytest tests/test_telemetria.py -v
```

Esperado: 5/5 PASS

- [ ] **Paso 6: Smoke test con ETS2 abierto (opcional si ETS2 disponible)**

```python
# scripts/verificar_telemetria.py (actualizar)
import sys; sys.path.insert(0, '.')
from src.telemetria.lector import TelemetriaLector
import time

lector = TelemetriaLector()
try:
    lector.abrir()
    for _ in range(5):
        snap = lector.leer()
        kmh = snap.velocidad_ms * 3.6
        print(f"Velocidad: {kmh:.1f} km/h | RPM: {snap.rpm:.0f} | "
              f"Marcha: {snap.marcha} | Motor: {snap.motor_encendido}")
        time.sleep(0.5)
finally:
    lector.cerrar()
```

```bash
python scripts/verificar_telemetria.py
```

Esperado (con ETS2 en partida cargada):
```
Velocidad: 0.0 km/h | RPM: 650.0 | Marcha: 0 | Motor: True
```

- [ ] **Paso 7: Commit**

```bash
git add src/telemetria/ tests/test_telemetria.py scripts/verificar_telemetria.py
git commit -m "feat: TelemetriaLector SCS SDK v12 con shared memory Win32 + mock para tests"
```

---

## FASE 2 — Capa de Control PID + vgamepad

### Tarea 5: PIDController genérico

**Archivos:**
- Crear: `src/control/pid.py`
- Crear: `tests/test_pid.py`

- [ ] **Paso 1: Escribir tests**

Crear `tests/test_pid.py`:

```python
import pytest
import time
from src.control.pid import PIDController


def test_respuesta_proporcional_pura():
    """Con Ki=0 Kd=0, la salida es Kp * error."""
    pid = PIDController(kp=2.0, ki=0.0, kd=0.0, limite=10.0)
    salida = pid.calcular(setpoint=5.0, medicion=3.0, dt=0.1)
    assert salida == pytest.approx(4.0)  # 2.0 * (5.0 - 3.0)


def test_salida_limitada_por_maximo():
    pid = PIDController(kp=100.0, ki=0.0, kd=0.0, limite=1.0)
    salida = pid.calcular(setpoint=10.0, medicion=0.0, dt=0.1)
    assert salida == pytest.approx(1.0)


def test_salida_limitada_por_minimo():
    pid = PIDController(kp=100.0, ki=0.0, kd=0.0, limite=1.0)
    salida = pid.calcular(setpoint=0.0, medicion=10.0, dt=0.1)
    assert salida == pytest.approx(-1.0)


def test_anti_windup_integral():
    """El integrador no debe crecer sin límite cuando la salida está saturada."""
    pid = PIDController(kp=1.0, ki=10.0, kd=0.0, limite=1.0)
    for _ in range(100):
        pid.calcular(setpoint=10.0, medicion=0.0, dt=0.1)
    # Integral debe estar acotada por el límite
    assert abs(pid._integral) <= 1.0 / 10.0 + 1e-6


def test_reset_limpia_estado():
    pid = PIDController(kp=1.0, ki=5.0, kd=1.0, limite=10.0)
    pid.calcular(setpoint=5.0, medicion=0.0, dt=0.1)
    pid.reset()
    assert pid._integral == pytest.approx(0.0)
    assert pid._error_anterior == pytest.approx(0.0)


def test_derivativo_amortigua_cambio_rapido():
    """Con Kd alto y cambio repentino de error, la derivada reduce la salida."""
    pid = PIDController(kp=1.0, ki=0.0, kd=2.0, limite=100.0)
    # Error grande en primer paso
    s1 = pid.calcular(setpoint=10.0, medicion=0.0, dt=0.1)
    # Error igual en segundo paso — derivada = 0, solo P
    s2 = pid.calcular(setpoint=10.0, medicion=0.0, dt=0.1)
    # s1 incluye Kd * (error-0)/dt, s2 no tiene cambio de error
    assert s1 != pytest.approx(s2)
```

- [ ] **Paso 2: Ejecutar para ver fallo**

```bash
pytest tests/test_pid.py -v
```

Esperado: `FAILED — ModuleNotFoundError: No module named 'src.control.pid'`

- [ ] **Paso 3: Crear `src/control/pid.py`**

```python
"""Controlador PID genérico con anti-windup por clamping.

Uso típico:
    pid = PIDController(kp=0.5, ki=0.02, kd=0.1, limite=1.0)
    salida = pid.calcular(setpoint=0.0, medicion=desviacion_carril, dt=0.033)
    gamepad.left_joystick_float(x_value_float=salida, y_value_float=0.0)
"""


class PIDController:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        limite: float = 1.0,
    ):
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._limite = abs(limite)

        self._integral: float = 0.0
        self._error_anterior: float = 0.0

    def calcular(self, setpoint: float, medicion: float, dt: float) -> float:
        if dt <= 0:
            return 0.0

        error = setpoint - medicion

        # Término proporcional
        p = self._kp * error

        # Término integral con anti-windup por clamping
        self._integral += error * dt
        i_raw = self._ki * self._integral
        if abs(i_raw) > self._limite:
            # Clamp la integral para que no supere el límite
            self._integral = (self._limite / self._ki) * (1.0 if i_raw > 0 else -1.0)
        i = self._ki * self._integral

        # Término derivativo (sobre error, no sobre medición)
        d = self._kd * (error - self._error_anterior) / dt
        self._error_anterior = error

        salida = p + i + d
        return max(-self._limite, min(self._limite, salida))

    def reset(self) -> None:
        self._integral = 0.0
        self._error_anterior = 0.0
```

- [ ] **Paso 4: Ejecutar tests**

```bash
pytest tests/test_pid.py -v
```

Esperado: 6/6 PASS

- [ ] **Paso 5: Commit**

```bash
git add src/control/pid.py tests/test_pid.py
git commit -m "feat: PIDController con anti-windup por clamping — 6 tests"
```

---

### Tarea 6: ControladorGamepadPID (100% analógico)

**Archivos:**
- Crear: `src/control/gamepad_pid.py`
- Crear: `tests/test_gamepad_pid.py`
- Deprecar: `src/control/teclado.py` (agregar nota)

- [ ] **Paso 1: Escribir tests**

Crear `tests/test_gamepad_pid.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from src.control.gamepad_pid import ControladorGamepadPID, ConfigPID
from src.tipos import ComandoControl
import time


def _cmd(acel=0.0, freno=0.0, volante=0.0):
    return ComandoControl(acelerador=acel, freno=freno, volante=volante,
                          timestamp=time.monotonic())


@pytest.fixture
def ctrl():
    with patch("vgamepad.VX360Gamepad") as mock_gp_cls:
        mock_gp = MagicMock()
        mock_gp_cls.return_value = mock_gp
        c = ControladorGamepadPID()
        c.iniciar()
        yield c, mock_gp


def test_liberar_aplica_cero_en_todos_los_ejes(ctrl):
    c, gp = ctrl
    c.liberar()
    gp.right_trigger.assert_called_with(value=0)
    gp.left_trigger.assert_called_with(value=0)
    gp.left_joystick_float.assert_called_with(x_value_float=0.0, y_value_float=0.0)


def test_aplicar_acelerador_alto_llama_right_trigger(ctrl):
    c, gp = ctrl
    c.aplicar(_cmd(acel=1.0))
    # RT debe tener valor cercano a 255
    llamadas_rt = [call for call in gp.right_trigger.call_args_list]
    assert any(call.kwargs.get("value", 0) > 200 for call in llamadas_rt)


def test_aplicar_freno_alto_llama_left_trigger(ctrl):
    c, gp = ctrl
    c.aplicar(_cmd(freno=1.0))
    llamadas_lt = [call for call in gp.left_trigger.call_args_list]
    assert any(call.kwargs.get("value", 0) > 200 for call in llamadas_lt)


def test_aplicar_volante_positivo_gira_stick_derecha(ctrl):
    c, gp = ctrl
    c.aplicar(_cmd(volante=0.8))
    llamadas = [call for call in gp.left_joystick_float.call_args_list]
    assert any(call.kwargs.get("x_value_float", 0.0) > 0.5 for call in llamadas)


def test_cerrar_llama_liberar(ctrl):
    c, gp = ctrl
    c.cerrar()
    gp.right_trigger.assert_called()
```

- [ ] **Paso 2: Ejecutar para ver fallo**

```bash
pytest tests/test_gamepad_pid.py -v
```

Esperado: `FAILED — ModuleNotFoundError: No module named 'src.control.gamepad_pid'`

- [ ] **Paso 3: Crear `src/control/gamepad_pid.py`**

```python
"""Controlador de gamepad 100% analógico con PIDs internos.

Reemplaza teclado.py (digital) y gamepad.py (sin PID).
Todos los ejes van a vgamepad con valores flotantes continuos.

Tres PIDs independientes:
  - pid_volante:    setpoint=0  medicion=desviacion_carril → stick_x
  - pid_velocidad:  setpoint=vel_objetivo medicion=vel_actual → throttle/brake
  - (el freno directo de emergencia ignora PID)
"""
import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.control.base import Controlador
from src.control.pid import PIDController
from src.tipos import ComandoControl

logger = logging.getLogger(__name__)


@dataclass
class ConfigPID:
    kp: float
    ki: float
    kd: float


# Parámetros por defecto calibrados para ETS2 (ajustar en config/default.yaml)
_CFG_VOLANTE  = ConfigPID(kp=0.55, ki=0.015, kd=0.08)
_CFG_VELOCIDAD = ConfigPID(kp=0.12, ki=0.008, kd=0.04)


class ControladorGamepadPID(Controlador):
    """vgamepad con PID para control analógico suave del Volvo FH16.

    La entrada `ComandoControl` del FSM se interpreta como setpoints:
      - acelerador: setpoint de velocidad normalizado (0-1 → 0-90 km/h)
      - freno: presión de frenado directo (0-1 → LT 0-255)
      - volante: desviación de carril (-1..+1) → PID → stick_x

    Para emergencias (freno=1.0) se bypasea el PID y se aplica LT directo.
    """

    _VEL_MAX_MS = 25.0  # 90 km/h en m/s — velocidad máxima del piloto

    def __init__(
        self,
        cfg_volante: ConfigPID = _CFG_VOLANTE,
        cfg_velocidad: ConfigPID = _CFG_VELOCIDAD,
    ):
        self._pid_vol = PIDController(
            cfg_volante.kp, cfg_volante.ki, cfg_volante.kd, limite=1.0
        )
        self._pid_vel = PIDController(
            cfg_velocidad.kp, cfg_velocidad.ki, cfg_velocidad.kd, limite=1.0
        )
        self._gamepad = None
        self._t_ultimo: float = time.monotonic()
        self._vel_actual_ms: float = 0.0   # actualizado externamente

    def iniciar(self) -> None:
        import vgamepad as vg
        self._gamepad = vg.VX360Gamepad()
        logger.info("ControladorGamepadPID: gamepad virtual iniciado")

    def actualizar_velocidad(self, velocidad_ms: float) -> None:
        """Llamar cada frame con la velocidad real de telemetría."""
        self._vel_actual_ms = velocidad_ms

    def aplicar(self, cmd: ComandoControl) -> None:
        if self._gamepad is None:
            raise RuntimeError("Llamar a iniciar() primero")

        ahora = time.monotonic()
        dt = max(0.001, ahora - self._t_ultimo)
        self._t_ultimo = ahora

        # ── Volante: PID sobre desviación de carril ─────────────────────────
        # cmd.volante = desviación (-1=izq, +1=der) → setpoint=0 (centrado)
        stick_x = self._pid_vol.calcular(
            setpoint=0.0, medicion=cmd.volante, dt=dt
        )
        self._gamepad.left_joystick_float(x_value_float=stick_x, y_value_float=0.0)

        # ── Velocidad / Frenado ──────────────────────────────────────────────
        if cmd.freno >= 0.9:
            # Emergencia: freno directo bypass PID
            self._gamepad.right_trigger(value=0)
            self._gamepad.left_trigger(value=int(cmd.freno * 255))
            self._pid_vel.reset()
        else:
            # Velocidad objetivo normalizada → m/s
            vel_objetivo_ms = cmd.acelerador * self._VEL_MAX_MS
            pid_out = self._pid_vel.calcular(
                setpoint=vel_objetivo_ms,
                medicion=self._vel_actual_ms,
                dt=dt,
            )
            if pid_out >= 0:
                # Accelerar
                rt = int(min(1.0, pid_out) * 255)
                self._gamepad.right_trigger(value=rt)
                self._gamepad.left_trigger(value=0)
            else:
                # Frenar suavemente
                lt = int(min(1.0, -pid_out) * 255)
                self._gamepad.right_trigger(value=0)
                self._gamepad.left_trigger(value=lt)

        self._gamepad.update()

    def liberar(self) -> None:
        if self._gamepad:
            self._gamepad.right_trigger(value=0)
            self._gamepad.left_trigger(value=0)
            self._gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            self._gamepad.update()
            self._pid_vol.reset()
            self._pid_vel.reset()
            logger.info("ControladorGamepadPID: ejes liberados")

    def cerrar(self) -> None:
        self.liberar()
```

- [ ] **Paso 4: Ejecutar tests**

```bash
pytest tests/test_gamepad_pid.py -v
```

Esperado: 5/5 PASS

- [ ] **Paso 5: Deprecar `src/control/teclado.py`**

Agregar al inicio de `src/control/teclado.py`:

```python
# DEPRECATED: usar ControladorGamepadPID para control analógico suave.
# Este módulo se mantiene solo como fallback cuando vgamepad no está disponible.
# Ver src/control/gamepad_pid.py
```

- [ ] **Paso 6: Commit**

```bash
git add src/control/gamepad_pid.py tests/test_gamepad_pid.py src/control/teclado.py
git commit -m "feat: ControladorGamepadPID con 3 PIDs para control 100% analógico
 
- pid_volante: desviacion_carril -> stick_x (suaviza correcciones)
- pid_velocidad: setpoint_vel -> throttle/brake split
- bypass directo para frenado de emergencia
- DEPRECATED: teclado.py"
```

---

## FASE 3 — Supervisor Híbrido (Visión + Telemetría)

### Tarea 7: Supervisor de reglas combinadas

**Archivos:**
- Crear: `src/decision/supervisor.py`
- Crear: `tests/test_supervisor.py`

- [ ] **Paso 1: Escribir tests**

Crear `tests/test_supervisor.py`:

```python
import time
import pytest
from src.tipos import Accion, EstadoEscena, EstadoSemaforo, TelemetriaSnapshot
from src.decision.supervisor import Supervisor, OverrideDecision


def _escena(**kwargs) -> EstadoEscena:
    defaults = dict(
        frente_cercano_ocupado=False, frente_lejano_ocupado=False,
        peaton_en_riesgo=False, semaforo_visible=None, senal_alto_cercana=False,
        espejo_izq_ocupado=False, espejo_der_ocupado=False,
        vehiculos_totales=0, confianza_percepcion=1.0, timestamp=time.monotonic(),
    )
    defaults.update(kwargs)
    return EstadoEscena(**defaults)


def _telem(**kwargs) -> TelemetriaSnapshot:
    defaults = dict(velocidad_ms=20.0, motor_encendido=True, timestamp=time.monotonic())
    defaults.update(kwargs)
    return TelemetriaSnapshot(**defaults)


def test_sin_condiciones_criticas_no_hay_override():
    sv = Supervisor()
    escena = _escena(telemetria=_telem())
    resultado = sv.evaluar(escena)
    assert resultado is None


def test_semaforo_rojo_alta_velocidad_activa_freno_fuerte():
    """Semáforo rojo + velocidad > 5 km/h → override FRENAR_FUERTE."""
    sv = Supervisor()
    escena = _escena(
        semaforo_visible=EstadoSemaforo.ROJO,
        telemetria=_telem(velocidad_ms=15.0),  # 54 km/h
    )
    resultado = sv.evaluar(escena)
    assert resultado is not None
    assert resultado.accion == Accion.FRENAR_FUERTE


def test_freno_mano_activo_con_motor_encendido_genera_override():
    """Freno de mano activo + camión en movimiento → ALTO_TOTAL."""
    sv = Supervisor()
    escena = _escena(
        telemetria=_telem(velocidad_ms=5.0, freno_mano=True),
    )
    resultado = sv.evaluar(escena)
    assert resultado is not None
    assert resultado.accion == Accion.ALTO_TOTAL


def test_exceso_velocidad_reduce_a_mantener():
    """Velocidad > 26 m/s (93 km/h) → override FRENAR_SUAVE para no exceder límite."""
    sv = Supervisor()
    escena = _escena(telemetria=_telem(velocidad_ms=27.0))
    resultado = sv.evaluar(escena)
    assert resultado is not None
    assert resultado.accion == Accion.FRENAR_SUAVE


def test_sin_telemetria_supervisor_no_actua():
    """Sin TelemetriaSnapshot, el supervisor no puede evaluar condiciones físicas."""
    sv = Supervisor()
    escena = _escena()  # sin telemetria
    resultado = sv.evaluar(escena)
    assert resultado is None
```

- [ ] **Paso 2: Ejecutar para ver fallo**

```bash
pytest tests/test_supervisor.py -v
```

Esperado: `FAILED — ModuleNotFoundError: No module named 'src.decision.supervisor'`

- [ ] **Paso 3: Crear `src/decision/supervisor.py`**

```python
"""Supervisor híbrido: combina detecciones YOLO con telemetría para overrides.

El Supervisor se evalúa ANTES del FSM principal. Si devuelve un OverrideDecision,
ese resultado reemplaza completamente la decisión del FSM.

Reglas de override (ordenadas por prioridad):
  S1: Freno de mano activo + movimiento → ALTO_TOTAL
  S2: Semáforo rojo + velocidad > umbral → FRENAR_FUERTE
  S3: Exceso de velocidad → FRENAR_SUAVE
  S4: Motor apagado → ESPERAR
"""
from dataclasses import dataclass
from typing import Optional

from src.tipos import Accion, EstadoEscena, EstadoSemaforo

_VEL_LIMITE_MS     = 25.6   # 92 km/h — límite del piloto
_VEL_MIN_FRENADO   = 1.4    # 5 km/h — umbral para frenado por semáforo
_VEL_MOVIMIENTO    = 0.5    # 1.8 km/h — umbral para detectar movimiento


@dataclass
class OverrideDecision:
    accion: Accion
    razon: str
    prioridad: int   # menor = más prioritario


class Supervisor:
    """Evalúa condiciones físicas vía telemetría y devuelve overrides."""

    def evaluar(self, escena: EstadoEscena) -> Optional[OverrideDecision]:
        t = escena.telemetria
        if t is None:
            return None  # sin telemetría no podemos evaluar condiciones físicas

        # S1 — Freno de mano activo con camión en movimiento
        if t.freno_mano and t.velocidad_ms > _VEL_MOVIMIENTO:
            return OverrideDecision(
                Accion.ALTO_TOTAL,
                f"freno_mano=True y velocidad={t.velocidad_ms:.1f}m/s",
                prioridad=1,
            )

        # S2 — Semáforo rojo + velocidad significativa
        if (escena.semaforo_visible == EstadoSemaforo.ROJO
                and t.velocidad_ms > _VEL_MIN_FRENADO):
            return OverrideDecision(
                Accion.FRENAR_FUERTE,
                f"semaforo_rojo + velocidad={t.velocidad_ms * 3.6:.1f}km/h",
                prioridad=2,
            )

        # S3 — Exceso de velocidad del piloto
        if t.velocidad_ms > _VEL_LIMITE_MS:
            return OverrideDecision(
                Accion.FRENAR_SUAVE,
                f"exceso velocidad {t.velocidad_ms * 3.6:.1f}km/h > {_VEL_LIMITE_MS * 3.6:.1f}km/h",
                prioridad=3,
            )

        # S4 — Motor apagado
        if not t.motor_encendido:
            return OverrideDecision(Accion.ESPERAR, "motor apagado", prioridad=4)

        return None
```

- [ ] **Paso 4: Ejecutar tests**

```bash
pytest tests/test_supervisor.py -v
```

Esperado: 5/5 PASS

- [ ] **Paso 5: Commit**

```bash
git add src/decision/supervisor.py tests/test_supervisor.py
git commit -m "feat: Supervisor hibrido vision+telemetria con 4 reglas de override"
```

---

## FASE 4 — Integración del Pipeline

### Tarea 8: Configuración PID en `config/default.yaml`

**Archivos:**
- Modificar: `config/default.yaml`

- [ ] **Paso 1: Agregar secciones al YAML**

```yaml
# Añadir al final de config/default.yaml:

telemetria:
  habilitada: true             # false = usar mock con velocidad=0
  nombre_shm: "Local\\SCSTelemetry"

pid:
  volante:
    kp: 0.55
    ki: 0.015
    kd: 0.08
  velocidad:
    kp: 0.12
    ki: 0.008
    kd: 0.04
  velocidad_max_ms: 25.6       # 92 km/h
```

- [ ] **Paso 2: Commit**

```bash
git add config/default.yaml
git commit -m "config: secciones telemetria y pid con parametros iniciales"
```

---

### Tarea 9: Integrar todo en `ejecutar_piloto.py`

**Archivos:**
- Modificar: `scripts/ejecutar_piloto.py`

- [ ] **Paso 1: Actualizar imports**

Agregar al bloque de imports existente:

```python
from src.telemetria import TelemetriaLector, TelemetriaLectorMock
from src.control.gamepad_pid import ControladorGamepadPID, ConfigPID
from src.decision.supervisor import Supervisor
```

- [ ] **Paso 2: Reemplazar `construir_controlador` para soportar gamepad_pid**

Agregar nueva función (mantener la original como fallback):

```python
def construir_controlador_pid(cfg: dict) -> ControladorGamepadPID:
    pid_cfg = cfg.get("pid", {})
    vol = pid_cfg.get("volante", {})
    vel = pid_cfg.get("velocidad", {})
    ctrl = ControladorGamepadPID(
        cfg_volante=ConfigPID(
            kp=vol.get("kp", 0.55),
            ki=vol.get("ki", 0.015),
            kd=vol.get("kd", 0.08),
        ),
        cfg_velocidad=ConfigPID(
            kp=vel.get("kp", 0.12),
            ki=vel.get("ki", 0.008),
            kd=vel.get("kd", 0.04),
        ),
    )
    ctrl.iniciar()
    return ctrl


def construir_telemetria(cfg: dict):
    tel_cfg = cfg.get("telemetria", {})
    if not tel_cfg.get("habilitada", True):
        logger.info("Telemetría: usando mock (habilitada=false)")
        return TelemetriaLectorMock()
    lector = TelemetriaLector()
    try:
        lector.abrir()
        logger.info("Telemetría: SCS SDK shared memory conectada")
        return lector
    except RuntimeError as e:
        logger.warning("Telemetría SCS SDK no disponible (%s) — usando mock", e)
        return TelemetriaLectorMock()
```

- [ ] **Paso 3: Actualizar `main()` para instanciar los nuevos componentes**

En la sección "Construir componentes" de `main()`, reemplazar la línea de `controlador = construir_controlador(cfg)` con:

```python
    # Nuevos componentes del sistema híbrido
    tel_lector = construir_telemetria(cfg)
    supervisor = Supervisor()
    
    # Usar GamepadPID si el tipo es gamepad o gamepad_pid
    tipo_ctrl = (tipo_override or cfg["control"]["tipo"])
    if tipo_ctrl in ("gamepad", "gamepad_pid"):
        controlador = construir_controlador_pid(cfg)
    else:
        controlador = construir_controlador(cfg)  # fallback teclado/nulo
```

- [ ] **Paso 4: Actualizar el loop principal para telemetría + supervisor**

Dentro del `while fuente.esta_activa:`, en la sección `# ── Decisión`, reemplazar:

```python
            resultado = fsm.decidir(escena)
```

con:

```python
            # Leer telemetría y adjuntarla a la escena
            snap_tel = tel_lector.leer()
            escena.telemetria = snap_tel

            # Supervisor evalúa condiciones físicas primero
            override = supervisor.evaluar(escena)

            if override is not None:
                # Override toma el control total — FSM no evalúa
                resultado = resultado_cache   # mantener último estado FSM
                if resultado is None:
                    from src.decision.fsm import ResultadoDecision
                    from src.decision.estado import EstadoFSM
                    resultado = ResultadoDecision(
                        override.accion, EstadoFSM.RECUPERACION,
                        0, override.razon
                    )
                else:
                    from src.decision.fsm import ResultadoDecision
                    resultado = ResultadoDecision(
                        override.accion, resultado.estado_nuevo,
                        0, override.razon
                    )
                logger.debug("OVERRIDE S%d: %s", override.prioridad, override.razon)
            else:
                resultado_cache = fsm.decidir(escena)
                resultado = resultado_cache

            # Actualizar velocidad en el controlador PID si aplica
            if isinstance(controlador, ControladorGamepadPID):
                controlador.actualizar_velocidad(snap_tel.velocidad_ms)
```

- [ ] **Paso 5: Cerrar el lector de telemetría en el bloque `finally`**

En el bloque `finally` de `main()`, agregar antes de `fuente.cerrar()`:

```python
        tel_lector.cerrar()
```

- [ ] **Paso 6: Correr suite completa de tests**

```bash
pytest tests/ -v
```

Esperado: todos los tests existentes + nuevos = PASS (≥35 tests)

- [ ] **Paso 7: Smoke test con ETS2 + gamepad**

```bash
python scripts/ejecutar_piloto.py --control gamepad --sin-video
```

Verificar en el log:
```
[INFO] Telemetría: SCS SDK shared memory conectada
[INFO] ControladorGamepadPID: gamepad virtual iniciado
```

- [ ] **Paso 8: Commit final**

```bash
git add scripts/ejecutar_piloto.py
git commit -m "feat: integración completa arquitectura híbrida 4 capas

- Telemetría SCS SDK como fuente de verdad para velocidad y estado físico
- Supervisor evalúa overrides antes del FSM (semaforo+vel, freno_mano, exceso_vel)
- ControladorGamepadPID: 3 PIDs independientes → vgamepad 100% analógico
- Fallback graceful si telemetría no disponible (usa mock)"
```

---

## FASE 5 — Calibración PID (iterativa, post-integración)

### Tarea 10: Calibración de parámetros PID en pista

> Esta tarea es experimental y requiere ETS2 corriendo. No hay tests automáticos — verificación visual.

- [ ] **Paso 1: Crear script de calibración**

```python
# scripts/calibrar_pid.py
"""Calibrador interactivo de PIDs. Corre con ETS2 abierto.

Presiona durante la ejecución:
  1: reducir Kp_volante   2: aumentar Kp_volante
  3: reducir Kd_volante   4: aumentar Kd_volante
  5: reducir Kp_vel       6: aumentar Kp_vel
  P: imprimir parámetros actuales
  Q: salir y guardar en config/default.yaml
"""
import sys, time, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pynput.keyboard as kb
from src.control.gamepad_pid import ControladorGamepadPID, ConfigPID
from src.telemetria import TelemetriaLector
from src.tipos import ComandoControl

cfg_path = Path("config/default.yaml")
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

pid_cfg = cfg.setdefault("pid", {})
vol = pid_cfg.setdefault("volante", {"kp": 0.55, "ki": 0.015, "kd": 0.08})
vel = pid_cfg.setdefault("velocidad", {"kp": 0.12, "ki": 0.008, "kd": 0.04})

DELTA = 0.05

def imprimir():
    print(f"\nVolante  Kp={vol['kp']:.3f} Ki={vol['ki']:.4f} Kd={vol['kd']:.3f}")
    print(f"Velocidad Kp={vel['kp']:.3f} Ki={vel['ki']:.4f} Kd={vel['kd']:.3f}\n")

def on_press(key):
    try:
        k = key.char
        if k == '1': vol['kp'] = max(0, vol['kp'] - DELTA)
        elif k == '2': vol['kp'] += DELTA
        elif k == '3': vol['kd'] = max(0, vol['kd'] - DELTA)
        elif k == '4': vol['kd'] += DELTA
        elif k == '5': vel['kp'] = max(0, vel['kp'] - DELTA)
        elif k == '6': vel['kp'] += DELTA
        elif k == 'p': imprimir()
        elif k == 'q': return False
    except AttributeError:
        pass

print("Calibrador PID iniciado. 1/2=Kp_vol, 3/4=Kd_vol, 5/6=Kp_vel, P=print, Q=guardar")
with kb.Listener(on_press=on_press) as listener:
    listener.join()

with open(cfg_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print(f"Parámetros guardados en {cfg_path}")
imprimir()
```

- [ ] **Paso 2: Protocolo de calibración del PID de volante**

Ejecutar el piloto en una recta larga en ETS2:

1. **Kp muy bajo (0.1)**: el camión se desvía lentamente, correcciones pequeñas → subir hasta que siga el carril
2. **Kp correcto**: camión oscila ligeramente alrededor del centro → bajar Kd hasta que se estabilice
3. **Kd correcto**: sistema estable, sin oscilación → listo para curvas

Valores de inicio: `Kp=0.55, Ki=0.015, Kd=0.08`

- [ ] **Paso 3: Protocolo de calibración del PID de velocidad**

1. Con el camión en recta sin obstáculos, la velocidad debe subir suavemente hasta ~80 km/h
2. Si la aceleración es brusca → reducir Kp_vel
3. Si no alcanza la velocidad objetivo → aumentar Kp_vel o Ki_vel

- [ ] **Paso 4: Commit con parámetros calibrados**

```bash
git add config/default.yaml
git commit -m "tune: parámetros PID calibrados en pista ETS2 Volvo FH16"
```

---

## Self-Review

### 1. Spec coverage

| Directriz | Tarea que lo cubre |
|---|---|
| Mantener YOLO + captura | No se toca — intacto |
| Mantener OpenCV/Hough para curvatura de carril | `src/percepcion/carriles.py` intacto |
| Deprecar lectura dashboard por visión | No hay código de eso — N/A |
| Telemetría SCS SDK (velocidad, RPM, steering, brake, trailer) | Tareas 1-4 |
| FSM como supervisor | Tarea 7 (Supervisor separado del FSM) |
| YOLO + telemetría → overrides | Tarea 7 (`supervisor.py`) |
| Deprecar pydirectinput | Tarea 6 (`teclado.py` deprecado) |
| 100% control por vgamepad | Tarea 6 (`gamepad_pid.py`) |
| PIDs para steering/aceleración/freno | Tareas 5-6 |
| OS Windows | Todo usa Win32 ctypes/mmap, vgamepad, pydirectinput fallback |
| Calidad Mag 7 | TDD en cada tarea, anti-windup en PID, fallbacks en telemetría, tipos exhaustivos |

### 2. Scan de placeholders

- Sin TBDs ni TODOs en el código del plan
- Todos los tests tienen código real
- Todos los pasos tienen comandos exactos

### 3. Consistencia de tipos

- `TelemetriaSnapshot` definida en Tarea 2, usada en Tareas 3, 4, 7, 9 ✓
- `OverrideDecision` definida en Tarea 7, importada en Tarea 9 ✓
- `ConfigPID` definida en Tarea 6, usada en Tarea 9 ✓
- `controlador.actualizar_velocidad()` definido en Tarea 6, llamado en Tarea 9 ✓
- `TelemetriaLector.abrir()` / `.leer()` / `.cerrar()` consistentes en Tareas 4 y 9 ✓
