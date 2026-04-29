# Mejoras de Conducción en Autopista — Diseño

**Fecha:** 2026-04-28
**Estado:** Aprobado por usuario
**Archivos afectados:** `src/control/pure_pursuit.py`, `scripts/ejecutar_piloto.py`, `src/control/gamepad_pid.py`
**Objetivo:** Que el camión mantenga su carril en autopistas con rectas, curvas y desvíos sin salirse de la vía.

---

## Contexto y problema

La integración YOLOP + Pure Pursuit existe y funciona en rectas. En curvas y bifurcaciones el camión acumula daño (20% → 56% en una sesión corta) por tres fallos:

1. **Centrado incorrecto:** `calcular_giro()` toma el centroide de todo el área manejable (`da_mask`). En una autopista de dos carriles, el centroide cae sobre la línea central, no en el carril derecho.
2. **Look-ahead fijo:** `fila_lookahead_fija=0.6` no se adapta a la curvatura — en curvas pronunciadas el punto objetivo llega demasiado tarde y el camión ya está saliendo del carril.
3. **Recuperación nula:** Cuando no hay píxeles de carril, devuelve `0.0` (volante centro), que puede ser incorrecto o correcto por azar.

Adicionalmente, la condición `velocidad_actual_norm > 0.05` en el piloto bloquea la corrección de carril cuando el estimador de flujo óptico devuelve ruido (~0.4-0.7 estático), impidiendo correcciones mientras el camión se mueve.

---

## Cambio 1 — Pure Pursuit mejorado (`src/control/pure_pursuit.py`)

### 1a. Bias de carril derecho

En cada fila analizada, se descartan los píxeles que están en el **30% izquierdo** del área manejable. Solo se considera el 70% derecho para calcular el centroide objetivo. Esto sitúa el punto de anticipación en el carril derecho en autopistas europeas independientemente de cuántos carriles tenga la vía.

```
x_min_fila, x_max_fila = rango de píxeles verdes en la fila
x_bias = x_min_fila + (x_max_fila - x_min_fila) * 0.30   # 30% desde izq = inicio carril derecho
centroide = mean(píxeles > x_bias)
```

### 1b. Look-ahead dinámico basado en curvatura

Se miden dos centroides en la máscara: uno cercano (fila 70% de la imagen) y uno lejano (fila 38%). La diferencia horizontal normalizada es la curvatura estimada:

```
curvatura = |x_cercano - x_lejano| / ancho_imagen
fila_dinámica = lerp(0.38, 0.65, clamp(curvatura * 6, 0, 1))
```

- Recta (curvatura ≈ 0) → fila 0.38 (mira lejos) → trayectoria suave
- Curva pronunciada (curvatura > 0.15) → fila 0.65 (mira cerca) → responde antes

### 1c. Suavizado multi-fila

En vez de una sola fila, se promedian 5 filas equiespaciadas ±20 px alrededor de `fila_dinámica`. Se pondera cada fila con un kernel gaussiano simple (pesos [0.1, 0.2, 0.4, 0.2, 0.1]) para que la fila central tenga más influencia.

### 1d. Recuperación con memoria

Cuando no hay píxeles de carril en las 5 filas:
- Se devuelve el **último error conocido multiplicado por 0.85** (decaimiento exponencial)
- Se activa una bandera `_carril_perdido` que el piloto puede consultar para reducir velocidad

**Nueva firma:**
```python
def calcular_giro(self, mascara_camino: np.ndarray) -> tuple[float, bool]:
    """Retorna (error_norm [-1,1], carril_perdido: bool)"""
```

---

## Cambio 2 — Piloto (`scripts/ejecutar_piloto.py`)

### 2a. Eliminar bloqueo por velocidad

Eliminar la condición:
```python
and velocidad_actual_norm > 0.05
```
La corrección de carril se aplica siempre que el estado FSM sea de conducción normal, sin importar el estimador de flujo óptico (que tiene ruido estático conocido).

### 2b. Bajar alpha EMA del carril

```python
_ALPHA_EMA_CARRIL = 0.50  →  0.30
```
La nueva señal Pure Pursuit es más estable; menos alpha = más inercia = menos oscilación en recta.

### 2c. Reducción de velocidad cuando carril perdido

Cuando `pure_pursuit.calcular_giro()` devuelve `carril_perdido=True`, el setpoint de velocidad se recorta al 40% del objetivo FSM mientras dure la pérdida.

```python
giro, carril_perdido = pure_pursuit.calcular_giro(da_mask)
if carril_perdido:
    setpoint.velocidad_objetivo_norm *= 0.40
```

---

## Cambio 3 — Ganancias PID del volante (`src/control/gamepad_pid.py`)

La nueva señal Pure Pursuit es más limpia y más centrada (menos ruido de borde de carril). Se ajustan las constantes:

| Parámetro | Antes | Después | Razón |
|-----------|-------|---------|-------|
| Kp | 0.55 | 0.65 | Señal más limpia → se puede pedir más respuesta |
| Kd | 0.08 | 0.12 | Más amortiguación en recta larga |
| Ki | 0.015 | 0.015 | Sin cambio |

---

## Restricciones

- **RNF-06/07:** No se toca YOLOP, no se consulta telemetría. Todo sigue siendo pure-vision.
- **Sin archivos nuevos:** Los tres cambios van en archivos existentes.
- **Compatibilidad:** La firma de `calcular_giro()` cambia de `-> float` a `-> tuple[float, bool]`. El único llamador es `ejecutar_piloto.py`, que se actualiza en el mismo commit.

---

## Criterio de éxito

- En una sesión de prueba en autopista de 5+ minutos, el camión no acumula daño por salirse del carril.
- El punto de anticipación (cyan) se mantiene dentro del carril derecho en rectas y curvas moderadas.
- Las imágenes de debug (`--debug-carril-img`) muestran el punto cyan consistentemente dentro del área verde y al lado derecho del centroide total.
