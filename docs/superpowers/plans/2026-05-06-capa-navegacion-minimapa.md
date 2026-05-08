# Capa de Navegación por Minimapa — Plan de Implementación (Fase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introducir una capa de percepción del minimapa y del límite de velocidad del HUD que lea la maniobra de ruta y la velocidad permitida, y las registre en el piloto sin tocar todavía el control lateral ni la FSM.

**Architecture:** Se crea un parser determinista del minimapa en `src/percepcion/minimapa.py` y un lector del límite del HUD en `src/percepcion/limite_velocidad_hud.py`; se agregan tipos nuevos en `src/tipos.py`, y `scripts/ejecutar_piloto.py` los ejecuta en modo observación para emitir `estado_ruta` y `estado_limite_velocidad` al log y guardar imágenes de debug. Esta fase no modifica el volante ni la selección de carril; solo construye las señales y su observabilidad. En fases posteriores, cualquier sesgo o cambio de carril guiado por ruta deberá reutilizar la ocupación de espejos/laterales ya disponible en `EstadoEscena` como permiso de seguridad lateral.

**Tech Stack:** Python 3.11+, OpenCV, NumPy, pytest. Sin dependencias nuevas.

---

## Mapa de archivos

| Acción | Archivo | Responsabilidad |
|--------|---------|-----------------|
| Crear | `src/percepcion/minimapa.py` | ROI del minimapa, segmentación y clasificación de maniobra |
| Crear | `src/percepcion/limite_velocidad_hud.py` | ROI del límite, OCR/plantillas y confianza |
| Modificar | `src/percepcion/__init__.py` | Export del parser |
| Modificar | `src/tipos.py` | `ManiobraRuta` y `EstadoRuta` |
| Modificar | `config/default.yaml` | Configuración/ROI del minimapa y del límite |
| Modificar | `scripts/ejecutar_piloto.py` | Ejecutar parsers, log JSONL y debug del HUD |
| Crear | `tests/test_minimapa.py` | Tests unitarios del parser |
| Crear | `tests/test_limite_velocidad_hud.py` | Tests unitarios del lector del límite |

Nota: el gating por espejos/laterales se apoya en percepción ya existente (`EstadoEscena`, `contexto.py`, `fsm.py`) y se implementará cuando se active el sesgo/cambio de carril en fases posteriores.

---

## Tarea 1: Definir contratos y configuración del HUD de navegación

**Files:**
- Modify: `src/tipos.py`
- Modify: `config/default.yaml`

- [ ] **Paso 1: Agregar `ManiobraRuta` y `EstadoRuta` a `src/tipos.py`**

Crear un enum pequeño y estable:

- `SEGUIR_RECTO`
- `MANTENER_IZQ`
- `MANTENER_DER`
- `SALIDA_IZQ`
- `SALIDA_DER`
- `GIRO_IZQ`
- `GIRO_DER`
- `DESCONOCIDA`

Agregar un dataclass `EstadoRuta` con al menos:

- `visible: bool`
- `confianza: float`
- `maniobra: ManiobraRuta`
- `distancia_normalizada: float | None`
- `sesgo_lateral_objetivo: float`
- `ramal_objetivo: str`
- `requiere_cambio_carril: bool`

Agregar además un dataclass `EstadoLimiteVelocidadHUD` con al menos:

- `visible: bool`
- `confianza: float`
- `limite_kmh: int | None`

No mezclar todavía estos tipos dentro de `EstadoEscena`; en Fase 1 basta con que el piloto los produzca y los registre.

- [ ] **Paso 2: Agregar bloques `minimapa:` y `limite_velocidad_hud:` a `config/default.yaml`**

Definir:

- ROI relativa del widget
- intervalo de procesamiento (por ejemplo, cada 3 o 6 frames)
- flag para guardar debug del minimapa
- ROI del círculo/placa del límite
- thresholds o tolerancias mínimas para aceptar la lectura del límite

Mantener todo calibrado inicialmente para el layout actual 1920x1080 del HUD.

- [ ] **Verify**

```xml
<verify>
  <automated>python3 -m py_compile src/tipos.py scripts/ejecutar_piloto.py</automated>
</verify>
```

- [ ] **Done**

- Los tipos de navegación existen y son importables
- La configuración del minimapa y del límite vive en `config/default.yaml`
- No se alteró aún el comportamiento del control

---

## Tarea 2: Implementar parsers del minimapa y del límite de velocidad

**Files:**
- Create: `src/percepcion/minimapa.py`
- Create: `src/percepcion/limite_velocidad_hud.py`
- Create: `tests/test_minimapa.py`
- Create: `tests/test_limite_velocidad_hud.py`
- Modify: `src/percepcion/__init__.py`

- [ ] **Paso 1: Crear tests primero**

Crear fixtures sintéticas del minimapa que validen:

- recta
- salida derecha
- salida izquierda
- giro derecha fuerte
- confianza baja / ruta no visible

Los tests deben ejercitar una API simple, por ejemplo:

```python
estado = EstimadorMinimapa().estimar(frame_bgr)
```

y verificar `estado.maniobra`, `estado.visible` y `estado.confianza`.

Crear también fixtures sintéticas del límite de velocidad que validen al menos:

- `30`
- `50`
- `60`
- `80`
- señal no visible / confianza baja

- [ ] **Paso 2: Implementar `EstimadorMinimapa` y `EstimadorLimiteVelocidadHUD`**

Implementar un parser con estas etapas:

1. recorte ROI del minimapa
2. segmentación de la ruta resaltada por color
3. segmentación/estimación del icono del camión o, si no es robusto, uso de una referencia local fija dentro del widget
4. extracción de la trayectoria local de la ruta alrededor del camión
5. clasificación a la taxonomía de `ManiobraRuta`

Requisitos de diseño:

- devolver `DESCONOCIDA` cuando la señal no sea suficientemente confiable
- no asumir control lateral todavía
- incluir un método de debug tipo `roi_debug()` similar al OCR

Para el lector del límite:

- devolver `limite_kmh=None` cuando la señal no sea suficientemente confiable
- no reutilizar directamente `EstimadorVelocidadDashboard`; solo compartir ideas de OCR si conviene
- incluir `roi_debug()` y confianza explícita

- [ ] **Paso 3: Exportar el parser en `src/percepcion/__init__.py`**

Mantener consistencia con el resto de los módulos de percepción.

- [ ] **Verify**

```xml
<verify>
  <automated>pytest tests/test_minimapa.py tests/test_limite_velocidad_hud.py -q</automated>
</verify>
```

- [ ] **Done**

- El parser clasifica al menos recta vs salida/giro en fixtures controladas
- El lector clasifica al menos 30/50/60/80 en fixtures controladas
- La API devuelve `EstadoRuta`
- La API del límite devuelve `EstadoLimiteVelocidadHUD`
- El parser puede fallar en modo seguro (`DESCONOCIDA`) sin lanzar excepciones
- El lector del límite puede fallar en modo seguro (`limite_kmh=None`) sin lanzar excepciones

---

## Tarea 3: Integrar en el piloto en modo observación

**Files:**
- Modify: `scripts/ejecutar_piloto.py`

- [ ] **Paso 1: Instanciar `EstimadorMinimapa` y `EstimadorLimiteVelocidadHUD`**

Crear ambos parsers desde `config/default.yaml` y ejecutarlos en submuestreo razonable para no castigar FPS.

- [ ] **Paso 2: Registrar `estado_ruta` y `estado_limite_velocidad` en `sesion_*.jsonl`**

Agregar un evento o ampliar el log existente para capturar:

- `frame`
- `maniobra`
- `confianza`
- `visible`
- `distancia_normalizada`
- `ramal_objetivo`

Agregar para el límite:

- `frame`
- `limite_kmh`
- `confianza`
- `visible`

No modificar todavía `setpoint.desviacion_volante`, `PurePursuitVisual` ni la FSM.

- [ ] **Paso 3: Guardar debug del minimapa**

Cuando el modo debug esté activo, guardar imágenes `debug_minimapa_*.jpg`, `debug_limite_velocidad_*.jpg` o incorporar ambos paneles a `debug_modelo_*.jpg`.

El debug debe permitir verificar:

- ROI correcta
- máscara/segmentación de la ruta
- maniobra inferida
- confianza
- ROI correcta del límite
- lectura `30/50/60/80/...` y confianza asociada

- [ ] **Verify**

```xml
<verify>
  <automated>python3 -m py_compile scripts/ejecutar_piloto.py src/percepcion/minimapa.py src/percepcion/limite_velocidad_hud.py && pytest tests/test_minimapa.py tests/test_limite_velocidad_hud.py -q</automated>
</verify>
```

- [ ] **Done**

- El piloto produce `estado_ruta` y `estado_limite_velocidad` en logs sin romper el loop principal
- Existe evidencia visual del parser del minimapa y del lector del límite
- El control actual del camión no cambia todavía

---

## Criterio de éxito de la Fase 1

- En una sesión de prueba, el sistema registra maniobras del minimapa con suficiente estabilidad para distinguir recta vs salida/giro
- En una sesión de prueba, el sistema registra límites del HUD con suficiente estabilidad para detectar transiciones de zona
- El parser falla en modo seguro cuando la ruta no es legible
- El lector del límite falla en modo seguro cuando la señal no es legible
- El loop principal sigue comportándose como antes; esta fase no introduce regresiones de conducción
- Queda definido en contratos/docs qué maniobras futuras requerirán permiso por `espejo_izq_ocupado` / `espejo_der_ocupado` antes de invadir otro carril

---

## No hacer en esta fase

- no sesgar todavía el volante
- no limitar todavía la velocidad objetivo por el límite detectado
- no alterar la lógica de `PurePursuitVisual`
- no añadir estados nuevos a la FSM
- no intentar resolver intersecciones o cambios de carril todavía

Eso queda para Fase 2 y Fase 3, una vez que el parser del minimapa esté validado con logs reales.

## Nota para fases siguientes

- El control por límite del HUD entra en Fase 2.
- El sesgo/cambio de carril guiado por ruta entra en Fase 3 y debe estar bloqueado si el lado objetivo no está libre en espejos/laterales.
- La selección de ramal de Fase 4 no debe “forzar” una invasión lateral si la percepción de espejos reporta ocupación persistente.
