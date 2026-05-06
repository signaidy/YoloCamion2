# Uso de ll_mask para Centro de Carril Exacto — Diseño

**Fecha:** 2026-04-29
**Estado:** Aprobado por usuario
**Archivos afectados:** `src/control/pure_pursuit.py`, `scripts/ejecutar_piloto.py`, `tests/test_pure_pursuit.py`
**Objetivo:** Reemplazar el centroide de `da_mask` (que cubre múltiples carriles) con el centro geométrico del carril actual calculado desde las líneas pintadas (`ll_mask`), corrigiendo el sesgo lateral persistente y mejorando la respuesta en curvas.

---

## Problema raíz

`da_mask` (área manejable) cubre todos los carriles visibles. El centroide de esa región cae entre carriles, no en el centro del carril donde está el camión. Esto produce:

1. **Sesgo lateral constante**: en autopista de 2+ carriles el centroide está desplazado respecto al centro del carril derecho → error `desv_pp` con signo incorrecto → el camión no mantiene su carril.
2. **Curvatura dependiente de da_mask**: el estimador de curvatura compara centroides de da_mask a distintas alturas, que también heredan el sesgo de múltiples carriles.

`ll_mask` (líneas de carril) detecta los bordes pintados de cada carril y ya está disponible en `ejecutar_piloto.py` (guardado en `ll_mask_cache`) pero no se usa para calcular el error de dirección.

---

## Solución: `_centro_desde_ll()`

### Algoritmo central

En las mismas 5 filas de look-ahead (offsets `[-20, -10, 0, 10, 20]` alrededor de `fila_base`):

1. Acumular todos los píxeles activos de `ll_mask` en esas 5 filas.
2. Separar en dos grupos respecto al centro del frame (`x_camion = ancho // 2`):
   - `pixeles_izq`: índices con `x < x_camion`
   - `pixeles_der`: índices con `x >= x_camion`
3. Validar: cada grupo debe tener al menos `_MIN_LL_PIXELES = 15` píxeles.
4. Calcular bordes inmediatos del carril actual:
   - `borde_izq = max(pixeles_izq)` ← línea más cercana a la izquierda del camión
   - `borde_der = min(pixeles_der)` ← línea más cercana a la derecha del camión
5. `centro_carril = (borde_izq + borde_der) / 2`
6. `error = clip((x_camion - centro_carril) / (ancho × _ESCALA_ERROR), −1, 1)`

### Manejo multi-carril (automático)

En una autopista de N carriles, `ll_mask` captura todas las líneas pintadas visibles. El algoritmo toma `max(pixeles_izq)` y `min(pixeles_der)`, que son siempre **los bordes más cercanos al camión** — es decir, los bordes del carril actual, sin importar cuántos carriles adicionales haya a los lados.

Ejemplos:
- **1 carril**: berma izquierda + berma derecha → centro correcto.
- **2 carriles**: divisoria central + berma → centro del carril derecho.
- **3 carriles**: en carril derecho → `max(izq)` = línea entre carril 2 y 3; `min(der)` = berma → centro del carril derecho. En carril central → líneas inmediatas de cada lado → centro del carril central.

### Lógica de fallback (tres niveles)

```
Nivel 1: ll_mask con ambos lados ≥ 15 píxeles
         → usar centro_carril de ll_mask  (más preciso)

Nivel 2: ll_mask insuficiente (tramo sin líneas, oclusión, 1 solo lado)
         → usar centroide de da_mask con barrido adaptativo (comportamiento actual)

Nivel 3: da_mask también vacía (túnel, niebla densa)
         → último_error × 0.85 con flag carril_perdido=True (actual)
```

No hay blending entre niveles. Se usa el mejor disponible; si falla, el siguiente.

---

## Cambio 1 — `src/control/pure_pursuit.py`

### Nueva constante de clase

```python
_MIN_LL_PIXELES = 15   # píxeles mínimos por lado para considerar ll_mask válida
```

### Nueva firma de `calcular_giro`

```python
def calcular_giro(
    self,
    mascara_camino: np.ndarray,
    ll_mask: np.ndarray | None = None,
) -> tuple[float, bool]:
```

El parámetro `ll_mask` es opcional; con `None` el comportamiento es idéntico al actual (compatibilidad hacia atrás completa).

### Nuevo método privado `_centro_desde_ll`

```python
def _centro_desde_ll(
    self,
    ll_mask: np.ndarray,
    filas: list[int],
    x_camion: int,
) -> float | None:
    """
    Retorna el centro geométrico del carril actual desde ll_mask.
    Usa los bordes más cercanos a x_camion en cada lado.
    Retorna None si no hay suficiente señal (< _MIN_LL_PIXELES por lado).
    """
    pixeles_izq: list[int] = []
    pixeles_der: list[int] = []

    alto = ll_mask.shape[0]
    for fila_y in filas:
        if fila_y < 0 or fila_y >= alto:
            continue
        indices = np.where(ll_mask[fila_y, :] > 0)[0]
        for x in indices:
            if x < x_camion:
                pixeles_izq.append(int(x))
            else:
                pixeles_der.append(int(x))

    if len(pixeles_izq) < self._MIN_LL_PIXELES or len(pixeles_der) < self._MIN_LL_PIXELES:
        return None

    borde_izq = int(np.max(pixeles_izq))
    borde_der = int(np.min(pixeles_der))
    return (borde_izq + borde_der) / 2.0
```

### Modificación de `calcular_giro`

Antes del bloque de cálculo de `x_obj` desde `da_mask`, intentar ll_mask:

```python
# Intentar ll_mask primero (nivel 1)
filas_ll = [max(0, min(fila_base + off, alto - 1)) for off in offsets]
if ll_mask is not None:
    centro_ll = self._centro_desde_ll(ll_mask, filas_ll, x_camion)
else:
    centro_ll = None

if centro_ll is not None:
    x_obj = centro_ll
    self._ultimo_punto = (int(round(x_obj)), fila_base)
    dx = x_camion - x_obj
    error = float(np.clip(dx / (ancho * self._ESCALA_ERROR), -1.0, 1.0))
    self._ultimo_error = error
    return error, False

# Fallback nivel 2: centroide da_mask con barrido adaptativo (código actual sin cambios)
...
```

---

## Cambio 2 — `scripts/ejecutar_piloto.py`

Aplicar el mismo ROI (60%) a `ll_mask` y pasarlo a `calcular_giro`:

```python
# Antes (líneas ~291-295):
_fila_roi = int(da_mask.shape[0] * 0.60)
da_mask[:_fila_roi, :] = 0
giro_pure_pursuit, carril_perdido = pure_pursuit.calcular_giro(da_mask)

# Después:
_fila_roi = int(da_mask.shape[0] * 0.60)
da_mask[:_fila_roi, :] = 0
ll_mask_roi = ll_mask_cache.copy()
ll_mask_roi[:_fila_roi, :] = 0
giro_pure_pursuit, carril_perdido = pure_pursuit.calcular_giro(da_mask, ll_mask_roi)
```

---

## Cambio 3 — `tests/test_pure_pursuit.py`

### Test 1: ll_mask simétrico → error ≈ 0

```python
def test_ll_mask_simetrico_error_cero():
    """Líneas equidistantes del centro del frame → centro_carril = x_camion → error ≈ 0."""
    pp = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[100:480, 0:640] = 1
    ll = np.zeros((480, 640), dtype=np.uint8)
    ll[300:480, 215:225] = 1   # borde izquierdo  (centro≈220)
    ll[300:480, 415:425] = 1   # borde derecho    (centro≈420)
    # centro_carril = (220+420)/2 = 320 = x_camion → error ≈ 0
    error, perdido = pp.calcular_giro(da, ll)
    assert not perdido
    assert abs(error) < 0.05
```

### Test 2: ll_mask corrige sesgo de da_mask

```python
def test_ll_mask_corrige_bias_da_mask():
    """
    da_mask asimétrica (centroide ≈ 420, error negativo).
    ll_mask dice centro = 320 → error ≈ 0.
    """
    pp_con_ll = PurePursuitVisual()
    pp_sin_ll = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[200:480, 200:640] = 1   # da_mask desplazada, centroide > 320
    ll = np.zeros((480, 640), dtype=np.uint8)
    ll[300:480, 215:225] = 1   # borde izq en ≈220
    ll[300:480, 415:425] = 1   # borde der en ≈420 → centro = 320

    error_con, _ = pp_con_ll.calcular_giro(da, ll)
    error_sin, _ = pp_sin_ll.calcular_giro(da)

    assert abs(error_con) < 0.05    # ll_mask: centrado
    assert error_sin < -0.10        # da_mask sola: sesgo negativo (centroide > 320)
```

### Test 3: ll_mask vacío → fallback a da_mask

```python
def test_ll_mask_vacio_usa_da_mask():
    """ll_mask sin píxeles → misma señal que sin ll_mask."""
    pp1 = PurePursuitVisual()
    pp2 = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[100:480, 50:300] = 1
    ll = np.zeros((480, 640), dtype=np.uint8)

    e1, _ = pp1.calcular_giro(da)
    e2, _ = pp2.calcular_giro(da, ll)
    assert e1 == pytest.approx(e2, rel=0.01)
```

### Test 4: ll_mask con pocos píxeles → fallback a da_mask

```python
def test_ll_mask_pocos_pixeles_no_activa_ll():
    """Menos de _MIN_LL_PIXELES por lado → cae a da_mask."""
    pp_ll = PurePursuitVisual()
    pp_da = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[100:480, 0:640] = 1     # da simétrica → error ≈ 0
    ll = np.zeros((480, 640), dtype=np.uint8)
    ll[350, 100] = 1            # 1 pixel izq — insuficiente
    ll[350, 540] = 1            # 1 pixel der — insuficiente

    e_ll, _ = pp_ll.calcular_giro(da, ll)
    e_da, _ = pp_da.calcular_giro(da)
    assert e_ll == pytest.approx(e_da, rel=0.01)
```

---

## Restricciones

- **RNF-06/07:** Pure-vision. No se toca YOLOP ni telemetría.
- **Sin archivos nuevos.** Todo en los tres archivos existentes.
- **Compatibilidad hacia atrás:** `calcular_giro(da_mask)` sin `ll_mask` sigue funcionando igual; los 8 tests existentes no se tocan.
- **`ll_mask_cache` ya existe** en `ejecutar_piloto.py` (línea 246) — solo necesita aplicarle ROI y pasarlo.

---

## Criterio de éxito

- En una sesión de autopista de 5+ minutos, `desv_pp` oscila alrededor de 0 (no permanece negativo).
- En curvas de 1-3 carriles, el punto cyan (look-ahead) se mantiene dentro del carril actual.
- Los 12 tests (8 existentes + 4 nuevos) pasan.
- El camión no acumula daño por salirse del carril en rectas.
