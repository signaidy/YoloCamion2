# Capa de Navegación por Minimapa — Diseño

**Fecha:** 2026-05-06  
**Estado:** Borrador para aprobación  
**Archivos previstos:** `src/percepcion/minimapa.py`, `src/percepcion/limite_velocidad_hud.py`, `src/tipos.py`, `src/decision/fsm.py`, `src/control/pure_pursuit.py`, `src/control/carril_speed_policy.py`, `scripts/ejecutar_piloto.py`, `config/default.yaml`, `tests/test_minimapa.py`, `tests/test_limite_velocidad_hud.py`

## Objetivo

Agregar una capa de navegación que use el minimapa/HUD de ETS2 para decidir:

1. hacia qué ramal debe comprometerse el camión en bifurcaciones
2. en qué carril debe ir colocándose antes de una salida o giro
3. cuándo una pérdida temporal de líneas debe resolverse siguiendo la ruta y no solo el carril local
4. cuál es el límite de velocidad vigente mostrado por el HUD para modular el target speed
5. cuándo un cambio de carril guiado por ruta es seguro según espejos/laterales visibles en la vista central

La meta no es reemplazar el seguimiento de carril actual, sino **inyectarle intención de ruta**.

---

## Contexto y problema

El piloto actual ya resuelve bastante bien la conducción local:

- `YOLO11n + tracker + contexto` detectan vehículos, peatones, semáforos y señales
- `YOLOP + PurePursuitVisual` centran el camión dentro del carril visible
- la FSM decide acelerar, frenar, mantener distancia y rebasar
- el OCR del HUD aporta velocidad propia sin telemetría interna

Lo que falta es información de **alto nivel**:

- en una bifurcación, `PurePursuitVisual` ve varias trayectorias plausibles pero no sabe cuál pertenece a la ruta
- en una salida, el camión no se pre-posiciona en el carril correcto con anticipación
- en un cruce urbano, si las líneas desaparecen, solo queda `da_mask` y memoria local; no existe intención global de giro
- al entrar a pueblos o zonas con restricción, el piloto no tiene aún una señal robusta del **límite de velocidad permitido** del HUD
- aunque el HUD y la ruta indiquen “muévete al carril izquierdo/derecho”, la navegación todavía no usa como requisito duro los espejos/laterales ya visibles en pantalla para evitar meterse sobre otro vehículo

En otras palabras: el sistema actual es un **lane follower reactivo**, no un **route follower**.

---

## Restricciones

- **Pure vision únicamente.** No usar telemetría del juego ni APIs internas.
- **Seguridad primero.** La navegación nunca debe sobrepasar las reglas de peatón, TTC, semáforo, paro manual o watchdog.
- **Compatibilidad incremental.** La fase inicial debe observar y registrar; no debe tocar el volante todavía.
- **Resolución base:** comenzar sobre el layout actual 1920x1080 escalado, igual que el resto del pipeline.
- **Sin ML nuevo al inicio.** El minimapa es un HUD con colores/forma estable; primero conviene usar segmentación por color y geometría.

---

## Requisitos v1

- **NAV-01**: extraer un ROI estable del minimapa desde el frame del juego
- **NAV-02**: detectar la ruta resaltada y la posición/rumbo local del camión dentro del minimapa
- **NAV-03**: clasificar la maniobra próxima a una taxonomía pequeña y estable
- **NAV-04**: producir una intención de navegación consumible por el loop principal sin romper el control actual
- **NAV-05**: sesgar el seguimiento de carril para pre-colocarse antes de salidas/ramales
- **NAV-06**: escoger el corredor correcto en bifurcaciones ambiguas cuando `ll_mask`/`da_mask` ofrecen más de una opción
- **NAV-07**: atravesar giros/intersecciones manteniendo intención de ruta aunque las líneas locales se degraden
- **NAV-08**: registrar y visualizar el estado de navegación para depuración comparable con `debug_yolop`, `debug_carriles` y `sesion_*.jsonl`
- **NAV-09**: mantener prioridad absoluta de reglas de seguridad y seguimiento de vehículo
- **NAV-10**: detectar el límite de velocidad vigente mostrado por el HUD con una confianza explícita
- **NAV-11**: usar el límite detectado para acotar la velocidad objetivo del piloto sin romper TTC, semáforos ni frenado preventivo
- **NAV-12**: cualquier cambio de carril o sesgo de ruta que implique invadir otro carril debe estar condicionado por espejos/laterales libres con histéresis temporal

---

## Enfoque recomendado

### 1. Parser determinista del minimapa

Crear `src/percepcion/minimapa.py` con un estimador dedicado al HUD:

- ROI configurable en `config/default.yaml`
- segmentación HSV/BGR para:
  - ruta resaltada
  - icono/flecha del camión
  - geometría vial local del minimapa
- cálculo de una ventana local alrededor del camión
- clasificación de maniobra por geometría local de la ruta:
  - `seguir_recto`
  - `mantener_izq`
  - `mantener_der`
  - `salida_izq`
  - `salida_der`
  - `giro_izq`
  - `giro_der`
  - `desconocida`

Este parser debe producir confianza y evidencia suficiente para saber cuándo **no** confiar en él.

### 2. Lector del límite de velocidad del HUD

Crear `src/percepcion/limite_velocidad_hud.py` como parser hermano del velocímetro:

- ROI configurable en `config/default.yaml`
- detección del círculo/señal del HUD cerca del minimapa
- OCR/plantillas para los dígitos del límite (`30`, `50`, `60`, `80`, etc.)
- salida con:
  - `limite_kmh`
  - `confianza`
  - `visible`

No debe reutilizar a ciegas la lectura del velocímetro actual: el problema visual es distinto. Puede compartir ideas de OCR y prototipos, pero debe tener thresholds y debug propios.

### 3. Nueva intención de ruta, no control directo

Agregar nuevos tipos en `src/tipos.py`:

- `ManiobraRuta` (enum)
- `EstadoRuta` (dataclass)

`EstadoRuta` debería incluir al menos:

- `maniobra`
- `confianza`
- `distancia_normalizada`
- `sesgo_lateral_objetivo`
- `requiere_cambio_carril`
- `ramal_objetivo` (`izq`, `centro`, `der`, `desconocido`)

La salida del parser no debe girar el volante por sí sola. Debe ser una **intención** que el control local consume.

### 4. Integración por capas

#### Capa A — observabilidad

Primero solo leer, clasificar y registrar:

- `scripts/ejecutar_piloto.py` instancia el parser de minimapa
- `scripts/ejecutar_piloto.py` instancia el lector del límite de velocidad del HUD
- se agrega `estado_ruta` al log `sesion_*.jsonl`
- se agrega `estado_limite_velocidad` al log `sesion_*.jsonl`
- se agrega `debug_minimapa_*.jpg` y/o `debug_limite_velocidad_*.jpg`, o ambos paneles dentro de `debug_modelo`

Sin control aún. Esta fase sirve para calibrar colores, ROI y taxonomía real del HUD.

#### Capa B — control de velocidad guiado por límite

Antes de usar la ruta para dirigir el volante, usar el límite de velocidad para gobernar el target speed:

- introducir un cap dinámico sobre `setpoint.velocidad_objetivo_norm`
- filtrar por confianza y persistencia para evitar lecturas espurias
- permitir margen configurable (por ejemplo, límite exacto o límite + tolerancia pequeña)
- mantener prioridad de TTC, semáforo, alto y peatón por encima del límite

Esto resuelve el problema de entrar a pueblos o zonas lentas antes de meter navegación urbana compleja.

#### Capa C — pre-posicionamiento en autopista

Usar `EstadoRuta` para sesgar el centrado del carril actual:

- no reemplazar `PurePursuitVisual`
- sí darle un **lane bias** suave y acotado
- solo cuando:
  - la confianza del minimapa es suficiente
  - la maniobra exige colocarse a izquierda/derecha
  - la FSM está en estados de conducción compatibles
  - los espejos/laterales del lado objetivo están libres y estables por una ventana mínima

La fuente de seguridad lateral no debe salir del minimapa. Debe reutilizar la percepción existente de `EstadoEscena`:

- `espejo_izq_ocupado`
- `espejo_der_ocupado`
- y, si conviene, las ROIs laterales ya definidas en `contexto.py`

Esto resuelve “colócate en el carril derecho antes de la salida” sin inventar maniobras bruscas.

#### Capa D — compromiso en bifurcaciones

Ampliar `PurePursuitVisual` para aceptar preferencia de corredor/ramal:

- si `ll_mask` o `da_mask` detectan múltiples ramas válidas
- escoger la rama consistente con `EstadoRuta`
- mantener memoria de ese compromiso durante algunos frames para evitar volver al ramal principal por oscilación
- no comprometerse a un ramal que implique cruce lateral si el lado objetivo sigue ocupado en espejos/laterales

Esta es la pieza crítica para seguir salidas reales.

#### Capa E — giros e intersecciones

Agregar estados nuevos o subestados en la FSM:

- `APROXIMANDO_GIRO`
- `EJECUTANDO_GIRO`
- opcionalmente `COMPROMETIDO_A_SALIDA`

Durante estos estados:

- bajar velocidad objetivo
- mantener intención lateral aunque las líneas se pierdan dentro del cruce
- volver al seguimiento normal cuando el nuevo carril ya esté reacquirido

---

## Fases derivadas de los requisitos

### Fase 1 — Percepción del minimapa, límite y observabilidad

**Cubre:** NAV-01, NAV-02, NAV-03, NAV-04, NAV-08, NAV-10  
**Meta:** El sistema puede leer el minimapa y el límite del HUD, registrarlos con confianza explícita y sin alterar todavía el control.

**Criterios de éxito:**

- El piloto genera un `EstadoRuta` por frame o submuestreo con `maniobra` y `confianza`
- El piloto genera una lectura de `limite_kmh` con `visible/confianza`
- El log JSONL registra cambios de maniobra y de límite detectado
- Existe visualización de debug suficiente para verificar ROI, ruta segmentada, clasificación y lectura del límite
- En runs de prueba, el parser distingue recta vs salida/giro y lee cambios de límite con reproducibilidad razonable

### Fase 2 — Gobernanza de velocidad por límite de HUD

**Cubre:** NAV-11, NAV-09  
**Meta:** El camión ajusta su velocidad objetivo al límite vigente del HUD sin romper la seguridad actual.

**Criterios de éxito:**

- Al cambiar el límite del HUD, el target speed converge al nuevo régimen sin frenazos absurdos por una sola lectura mala
- En pueblos o zonas limitadas, el camión deja de sostener velocidad de autopista si el límite detectado es inferior
- El control por límite solo se activa cuando la confianza/persistencia lo permiten
- Reglas de TTC, peatón, semáforo y paro manual siguen dominando

### Fase 3 — Sesgo de carril guiado por ruta en autopista

**Cubre:** NAV-05, NAV-09, NAV-12  
**Meta:** El camión se pre-coloca en el carril adecuado antes de una salida sin romper la seguridad actual.

**Criterios de éxito:**

- Antes de una salida derecha, el camión converge al carril derecho sin zig-zag agresivo
- El sesgo solo se activa cuando la confianza del minimapa y los espejos/laterales del lado objetivo lo permiten
- Si el espejo/lateral del lado objetivo está ocupado, el sesgo de cambio de carril no se activa o se mantiene bloqueado
- Reglas de TTC, peatón, semáforo y paro manual siguen dominando

### Fase 4 — Selección de ramal en bifurcaciones

**Cubre:** NAV-06, NAV-09, NAV-12  
**Meta:** En una bifurcación, el seguimiento local escoge el corredor de la ruta y no uno cualquiera.

**Criterios de éxito:**

- Cuando hay dos corredores plausibles, el sistema sigue el ramal objetivo del minimapa
- Una vez comprometido, mantiene la elección el tiempo suficiente para cruzar la bifurcación
- Las pérdidas breves de `ll` no devuelven el control al ramal equivocado
- Si tomar el ramal exige invadir un carril ocupado, la navegación espera o limita el compromiso según la seguridad lateral disponible

### Fase 5 — Giros e intersecciones

**Cubre:** NAV-07, NAV-09  
**Meta:** El sistema ejecuta giros de ruta en zonas donde las líneas locales son pobres o desaparecen.

**Criterios de éxito:**

- Reduce velocidad antes del giro
- Mantiene la intención de giro durante el cruce/intersección
- Reacquire el nuevo carril al salir del giro y vuelve a conducción normal

---

## Archivos previstos por área

### Percepción

- `src/percepcion/minimapa.py` — parser del HUD
- `src/percepcion/limite_velocidad_hud.py` — lector del límite de velocidad del HUD
- `src/percepcion/__init__.py` — export del nuevo estimador
- `config/default.yaml` — ROI/configuración del minimapa y del límite

### Tipos / integración

- `src/tipos.py` — `ManiobraRuta`, `EstadoRuta`
- `scripts/ejecutar_piloto.py` — creación, registro y debug del estado de ruta y del límite
- integración con `EstadoEscena` para leer `espejo_izq_ocupado` / `espejo_der_ocupado`

### Decisión

- `src/decision/fsm.py`
- `src/decision/estado.py`
- posible helper de “permiso de cambio de carril” basado en histéresis de espejos/laterales

### Control

- `src/control/pure_pursuit.py` — bias y selección de ramal
- posiblemente `src/control/carril_steering_policy.py` — límites cuando el sesgo de ruta esté activo
- `src/control/carril_speed_policy.py` — límites cuando el límite del HUD esté activo

### Tests

- `tests/test_minimapa.py`
- `tests/test_limite_velocidad_hud.py`
- `tests/test_pure_pursuit.py`
- `tests/test_decision.py`

---

## Riesgos principales

### 1. Colores del minimapa no tan estables como parecen

Mitigación:

- arrancar con pruebas sobre capturas reales del HUD
- guardar crops/minimap debug desde el primer día
- diseñar el parser con confianza explícita y fallback a `desconocida`

### 2. Querer que el minimapa “maneje” demasiado pronto

Mitigación:

- Fase 1 solo observa
- Fase 2 solo controla velocidad por límite
- Fase 3 solo sesga
- Fase 4 decide ramales únicamente en puntos ambiguos

### 3. Mezclar navegación con seguridad

Mitigación:

- la navegación nunca salta por encima de la FSM de seguridad
- la intención de ruta entra como contexto adicional, no como bypass

### 4. Falsos positivos/negativos en espejos bloqueando o habilitando maniobras

Mitigación:

- reutilizar la histéresis existente en espejos/laterales antes de permitir el cambio de carril
- exigir ventana mínima de “libre” antes de activar sesgo de ruta que invada otro carril
- registrar explícitamente cuándo el cambio de carril quedó bloqueado por seguridad lateral

### 5. Sobreajuste al layout actual del HUD

Mitigación:

- ROI configurable
- debug de minimapa obligatorio
- documentar explícitamente que v1 sigue calibrado a la vista/HUD actual

---

## Decisión de arquitectura

La mejor ruta para este repositorio es:

1. **parser determinista del minimapa**
2. **lector del límite de velocidad del HUD**
3. **logging y verificación offline**
4. **control de velocidad por límite**
5. **sesgo suave de carril condicionado por espejos/laterales**
6. **compromiso de ramal con gating lateral**
7. **giros/intersecciones**

No recomiendo empezar con un modelo entrenado end-to-end ni con un “planner” abstracto desacoplado del HUD actual. El minimapa de ETS2 es fijo, visible y altamente estructurado; un parser visual incremental encaja mucho mejor con la arquitectura existente y reduce riesgo.
