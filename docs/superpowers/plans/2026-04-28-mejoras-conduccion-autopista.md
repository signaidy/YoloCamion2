# Mejoras de Conducción en Autopista — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reemplazar el Pure Pursuit de fila fija por uno con bias de carril derecho, look-ahead dinámico y recuperación con memoria, para que el camión mantenga su carril en autopista sin acumular daño.

**Architecture:** Se reescribe `PurePursuitVisual` en `src/control/pure_pursuit.py` con tres mejoras encadenadas (bias derecho, look-ahead dinámico, suavizado multi-fila + recuperación). Se actualiza `ejecutar_piloto.py` para consumir la nueva firma `-> tuple[float, bool]` y reducir velocidad cuando el carril se pierde. Se ajustan las ganancias PID del volante en `gamepad_pid.py`.

**Tech Stack:** Python 3.11, NumPy, pytest, vgamepad

---

## Mapa de archivos

| Acción | Archivo | Responsabilidad |
|--------|---------|-----------------|
| Reescribir | `src/control/pure_pursuit.py` | Bias derecho + look-ahead dinámico + multi-fila + recuperación |
| Crear | `tests/test_pure_pursuit.py` | Tests TDD del nuevo Pure Pursuit |
| Modificar | `scripts/ejecutar_piloto.py` | Desempaquetar tuple, quitar bloqueo velocidad, reducción al perder carril |
| Modificar | `src/control/gamepad_pid.py` | Kp 0.55→0.65, Kd 0.08→0.12 |

---

## Tarea 1: Tests del nuevo PurePursuitVisual (TDD primero)

**Archivos:**
- Crear: `tests/test_pure_pursuit.py`

- [ ] **Paso 1: Crear el archivo de tests**

```python
# tests/test_pure_pursuit.py
import numpy as np
import pytest

from src.control.pure_pursuit import PurePursuitVisual


def test_carril_perdido_devuelve_True_y_error_cero_inicial():
    """Máscara vacía sin memoria previa → (0.0, True)."""
    pp = PurePursuitVisual()
    mascara = np.zeros((480, 640), dtype=np.uint8)
    error, perdido = pp.calcular_giro(mascara)
    assert perdido is True
    assert error == pytest.approx(0.0)


def test_decaimiento_memoria_tras_perder_carril():
    """Tras detectar error, máscara vacía devuelve error_anterior * 0.85."""
    pp = PurePursuitVisual()
    m1 = np.zeros((480, 640), dtype=np.uint8)
    m1[200:480, 50:250] = 1          # área a la izquierda → error positivo
    error_1, perdido_1 = pp.calcular_giro(m1)
    assert not perdido_1
    assert error_1 > 0

    m2 = np.zeros((480, 640), dtype=np.uint8)
    error_2, perdido_2 = pp.calcular_giro(m2)
    assert perdido_2 is True
    assert error_2 == pytest.approx(error_1 * 0.85, rel=0.05)


def test_via_ancha_bias_derecho_genera_error_negativo():
    """Vía de ancho completo (dos carriles): bias sitúa objetivo en carril derecho."""
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    m[100:480, 0:640] = 1            # toda la vía visible
    error, perdido = pp.calcular_giro(m)
    assert not perdido
    # Target cae a la derecha del centro → dx < 0 → error < 0 → PID gira derecha
    assert error < -0.10


def test_area_solo_izquierda_error_positivo():
    """Área manejable solo a la izquierda → camión debe girar izquierda (error > 0)."""
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    m[100:480, 0:200] = 1
    error, perdido = pp.calcular_giro(m)
    assert not perdido
    assert error > 0.10


def test_area_solo_derecha_error_negativo():
    """Área manejable solo a la derecha → camión debe girar derecha (error < 0)."""
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    m[100:480, 440:640] = 1
    error, perdido = pp.calcular_giro(m)
    assert not perdido
    assert error < -0.10


def test_error_acotado_entre_menos1_y_1():
    """El error normalizado nunca sale del rango [-1, 1]."""
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    m[100:480, 0:10] = 1             # franja extrema → dx muy grande
    error, _ = pp.calcular_giro(m)
    assert -1.0 <= error <= 1.0


def test_ultimo_punto_debug_none_cuando_carril_perdido():
    """Sin carril visible, ultimo_punto_debug debe ser None."""
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    pp.calcular_giro(m)
    assert pp.ultimo_punto_debug is None


def test_ultimo_punto_debug_dentro_de_la_imagen():
    """Con carril visible, ultimo_punto_debug cae dentro de los límites de la imagen."""
    pp = PurePursuitVisual()
    m = np.zeros((480, 640), dtype=np.uint8)
    m[100:480, 200:500] = 1
    pp.calcular_giro(m)
    assert pp.ultimo_punto_debug is not None
    x, y = pp.ultimo_punto_debug
    assert 0 <= x < 640
    assert 0 <= y < 480
```

- [ ] **Paso 2: Verificar que los tests fallan (clase no existe aún)**

```
pytest tests/test_pure_pursuit.py -v
```

Salida esperada: `ImportError` o `ModuleNotFoundError` — la clase no tiene la firma correcta.

---

## Tarea 2: Reescribir PurePursuitVisual

**Archivos:**
- Modificar: `src/control/pure_pursuit.py`

- [ ] **Paso 3: Reemplazar el contenido completo del archivo**

```python
# src/control/pure_pursuit.py
import numpy as np


class PurePursuitVisual:
    """
    Controlador Pure Pursuit visual con bias de carril derecho y look-ahead dinámico.

    Mejoras respecto a la versión anterior:
    - Bias derecho: solo considera el 70% derecho del área manejable para no
      apuntar a la línea central en autopistas de dos carriles.
    - Look-ahead dinámico: fila de anticipación se acorta en curvas y se alarga
      en rectas, medido por la curvatura del propio área manejable.
    - Suavizado multi-fila: promedia 5 filas con pesos gaussianos para reducir
      el ruido puntual de la máscara.
    - Recuperación con memoria: cuando se pierde el carril devuelve el último
      error multiplicado por 0.85 (decaimiento) en vez de devolver 0.0.
    """

    _DECAY = 0.85          # factor de decaimiento por frame cuando carril perdido
    _BIAS_FRAC = 0.30      # fracción izquierda del área verde que se descarta
    _FILA_LEJOS = 0.38     # fila relativa para look-ahead en recta (más lejos)
    _FILA_CERCA = 0.65     # fila relativa para look-ahead en curva (más cerca)
    _CURVATURA_SCALE = 6.0 # factor de amplificación de la curvatura cruda
    _ESCALA_ERROR = 0.35   # divisor de normalización (fracción del ancho)

    def __init__(self) -> None:
        self._ultimo_error: float = 0.0
        self._ultimo_punto: tuple[int, int] | None = None

    # ── API pública ────────────────────────────────────────────────────────────

    @property
    def ultimo_punto_debug(self) -> tuple[int, int] | None:
        """Último look-ahead point calculado; None si el carril estaba perdido."""
        return self._ultimo_punto

    def calcular_giro(self, mascara_camino: np.ndarray) -> tuple[float, bool]:
        """
        Calcula el error de dirección a partir de la máscara del área manejable.

        Returns:
            (error_norm, carril_perdido)
            error_norm  ∈ [-1, 1]: positivo → girar izquierda, negativo → girar derecha
            carril_perdido: True cuando no hay píxeles de carril visibles
        """
        alto, ancho = mascara_camino.shape
        x_camion = ancho // 2

        curvatura = self._estimar_curvatura(mascara_camino, alto, ancho)
        fila_base = int(alto * (self._FILA_LEJOS + curvatura * (self._FILA_CERCA - self._FILA_LEJOS)))

        # Suavizado: 5 filas con pesos gaussianos, separadas 10 px
        offsets = [-20, -10,  0, 10, 20]
        pesos   = [ 0.10, 0.20, 0.40, 0.20, 0.10]

        x_sum = 0.0
        w_sum = 0.0
        for off, peso in zip(offsets, pesos):
            y = max(0, min(fila_base + off, alto - 1))
            x = self._centroide_con_bias(mascara_camino, y, ancho)
            if x is not None:
                x_sum += x * peso
                w_sum += peso

        if w_sum == 0.0:
            self._ultimo_punto = None
            self._ultimo_error *= self._DECAY
            return self._ultimo_error, True

        x_obj = x_sum / w_sum
        self._ultimo_punto = (int(round(x_obj)), fila_base)

        dx = x_camion - x_obj
        error = float(np.clip(dx / (ancho * self._ESCALA_ERROR), -1.0, 1.0))
        self._ultimo_error = error
        return error, False

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _centroide_con_bias(self, mascara: np.ndarray, fila_y: int, ancho: int) -> int | None:
        """
        Centroide del 70% derecho del área manejable en la fila dada.
        Retorna None si no hay píxeles.
        """
        fila = mascara[fila_y, :]
        indices = np.where(fila > 0)[0]
        if len(indices) == 0:
            return None

        x_min = int(indices[0])
        x_max = int(indices[-1])
        ancho_area = x_max - x_min

        if ancho_area == 0:
            return x_min

        x_bias = x_min + int(ancho_area * self._BIAS_FRAC)
        indices_der = indices[indices >= x_bias]
        if len(indices_der) == 0:
            return int(np.mean(indices))
        return int(np.mean(indices_der))

    def _estimar_curvatura(self, mascara: np.ndarray, alto: int, ancho: int) -> float:
        """
        Curvatura ∈ [0, 1] basada en la diferencia horizontal entre el
        centroide cercano (fila 70%) y el lejano (fila 38%).
        """
        y_cerca = int(alto * 0.70)
        y_lejos = int(alto * 0.38)

        x_cerca = self._centroide_con_bias(mascara, y_cerca, ancho)
        x_lejos = self._centroide_con_bias(mascara, y_lejos, ancho)

        if x_cerca is None or x_lejos is None:
            return 0.0

        return float(np.clip(abs(x_cerca - x_lejos) / ancho * self._CURVATURA_SCALE, 0.0, 1.0))
```

- [ ] **Paso 4: Ejecutar los tests y verificar que pasan**

```
pytest tests/test_pure_pursuit.py -v
```

Salida esperada: **8 PASSED**. Si alguno falla, revisar la aritmética del bias o el límite de fila.

- [ ] **Paso 5: Commit de Tarea 1 y 2**

```bash
git add tests/test_pure_pursuit.py src/control/pure_pursuit.py
git commit -m "feat: Pure Pursuit con bias carril derecho, look-ahead dinamico y recuperacion"
```

---

## Tarea 3: Actualizar ejecutar_piloto.py

**Archivos:**
- Modificar: `scripts/ejecutar_piloto.py`

Hay **4 cambios** en este archivo. Aplicarlos en orden.

- [ ] **Paso 6: Cambiar alpha EMA del carril (línea ~246)**

Buscar:
```python
_ALPHA_EMA_CARRIL = 0.50
```
Reemplazar por:
```python
_ALPHA_EMA_CARRIL = 0.30
```

- [ ] **Paso 7: Instanciar PurePursuitVisual sin argumentos (línea ~184)**

Buscar:
```python
pure_pursuit = PurePursuitVisual(fila_lookahead_fija=0.6)
```
Reemplazar por:
```python
pure_pursuit = PurePursuitVisual()
```

- [ ] **Paso 8: Desempaquetar el tuple y quitar el bloqueo por velocidad**

Buscar el bloque actual (aprox. líneas 278-313):
```python
            # ── Detección de carril (cada frame) ───────────────
            # Extraer máscaras con YOLOP. Ignoramos detecciones aquí porque usamos el tracker para objetos.
            _, da_mask, ll_mask = yolop.procesar_frame(cuadro.imagen)
            
            # Calcular el giro del volante usando el área manejable (Drivable Area) con Pure Pursuit
            giro_pure_pursuit = pure_pursuit.calcular_giro(da_mask)

            # Actualizar EMA de desviación para suavizar movimientos del volante
            desv_ema = (_ALPHA_EMA_CARRIL * giro_pure_pursuit + (1.0 - _ALPHA_EMA_CARRIL) * desv_ema)
```

Reemplazar por:
```python
            # ── Detección de carril (cada frame) ───────────────
            _, da_mask, ll_mask = yolop.procesar_frame(cuadro.imagen)

            # Pure Pursuit: bias derecho + look-ahead dinámico
            giro_pure_pursuit, carril_perdido = pure_pursuit.calcular_giro(da_mask)

            # EMA de suavizado (alpha=0.30 — señal ya más estable que antes)
            desv_ema = (_ALPHA_EMA_CARRIL * giro_pure_pursuit + (1.0 - _ALPHA_EMA_CARRIL) * desv_ema)
```

- [ ] **Paso 9: Aplicar override de carril sin bloqueo por velocidad y añadir reducción de velocidad**

Buscar el bloque de override (aprox. líneas 309-313):
```python
            # Override de carril (Pure Pursuit con YOLOP):
            # Solo seguimos el carril si el estado es de mantenimiento de ruta o seguimiento
            if (resultado.accion not in _ACCIONES_CON_GIRO
                    and resultado.estado_nuevo in _ESTADOS_CARRIL
                    and velocidad_actual_norm > 0.05):
                desv = float(max(-1.0, min(1.0, desv_ema)))
                setpoint.desviacion_volante = desv
```

Reemplazar por:
```python
            # Override de carril: activo en estados de conducción normal
            # (sin bloqueo por velocidad — el estimador de flujo tiene ruido estático)
            if (resultado.accion not in _ACCIONES_CON_GIRO
                    and resultado.estado_nuevo in _ESTADOS_CARRIL):
                setpoint.desviacion_volante = float(np.clip(desv_ema, -1.0, 1.0))

            # Reducir velocidad cuando el carril se pierde por oclusión
            if carril_perdido and resultado.estado_nuevo in _ESTADOS_CARRIL:
                setpoint.velocidad_objetivo_norm *= 0.40
```

- [ ] **Paso 10: Actualizar el bloque de debug de imagen para usar ultimo_punto_debug**

Buscar el bloque de debug (aprox. líneas 334-341):
```python
                # Dibujar look-ahead point
                alto, ancho = dbg.shape[:2]
                y_objetivo = int(alto * pure_pursuit.fila_lookahead)
                punto = pure_pursuit._encontrar_punto_objetivo(da_mask, y_objetivo)
                if punto:
                    _cv2.circle(dbg, punto, 10, (255, 255, 0), -1)
```

Reemplazar por:
```python
                # Dibujar look-ahead point
                punto = pure_pursuit.ultimo_punto_debug
                if punto:
                    _cv2.circle(dbg, punto, 10, (255, 255, 0), -1)
```

- [ ] **Paso 11: Verificar que el script arranca sin error de sintaxis**

```
python -c "import scripts.ejecutar_piloto" 2>&1 || python scripts/ejecutar_piloto.py --help
```

Salida esperada: muestra el help de argparse sin traceback.

- [ ] **Paso 12: Ejecutar la suite completa de tests**

```
pytest tests/ -v
```

Salida esperada: todos los tests pasan (incluyendo los 8 nuevos de pure_pursuit).

- [ ] **Paso 13: Commit de Tarea 3**

```bash
git add scripts/ejecutar_piloto.py
git commit -m "feat: piloto usa nuevo Pure Pursuit (tuple, sin bloqueo vel, reduce vel al perder carril)"
```

---

## Tarea 4: Ajustar ganancias PID del volante

**Archivos:**
- Modificar: `src/control/gamepad_pid.py`

- [ ] **Paso 14: Actualizar las constantes de configuración PID**

Buscar la línea (aprox. línea 39):
```python
_CFG_VOLANTE_DEFAULT  = ConfigPID(kp=0.55, ki=0.015, kd=0.08)
```

Reemplazar por:
```python
_CFG_VOLANTE_DEFAULT  = ConfigPID(kp=0.65, ki=0.015, kd=0.12)
```

- [ ] **Paso 15: Ejecutar los tests de gamepad_pid para confirmar que no se rompe nada**

```
pytest tests/test_gamepad_pid.py -v
```

Salida esperada: PASSED (los tests mockean vgamepad, no dependen de los valores Kp/Kd).

- [ ] **Paso 16: Suite completa final**

```
pytest tests/ -v
```

Salida esperada: todos los tests pasan sin errores.

- [ ] **Paso 17: Commit final de Tarea 4**

```bash
git add src/control/gamepad_pid.py
git commit -m "tune: PID volante Kp 0.55->0.65 Kd 0.08->0.12 para nueva señal Pure Pursuit"
```

---

## Validación en juego (post-implementación)

Una vez que todos los tests pasen, probar en ETS2 con:

```
python scripts/ejecutar_piloto.py --fuente ventana --control gamepad --delay 5 --debug-carril --debug-carril-img
```

Criterios de éxito:
- El punto cyan (`ultimo_punto_debug`) aparece dentro del área verde, hacia la derecha del centroide total de la vía.
- En rectas, el punto se ve lejos (fila ~38% de la imagen).
- En curvas, el punto se ve más cerca (fila ~55-65%).
- En 5 minutos de autopista, el camión no acumula daño por salirse del carril.
- El log muestra `desv_pp` con signo consistente (negativo en recta = ligero pull a la derecha = correcto).
