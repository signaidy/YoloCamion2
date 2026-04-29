# ll_mask Lane Center Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Usar `ll_mask` (líneas pintadas detectadas por YOLOP) para calcular el centro exacto del carril actual, reemplazando el centroide de `da_mask` que cubre múltiples carriles y produce sesgo lateral.

**Architecture:** `PurePursuitVisual.calcular_giro()` acepta un nuevo parámetro opcional `ll_mask`. Si tiene suficientes píxeles en ambos lados del frame, calcula el centro del carril como `(borde_izq + borde_der) / 2` y retorna ese error. Si no, cae al centroide de `da_mask` (comportamiento actual). `ejecutar_piloto.py` aplica el mismo ROI al `ll_mask_cache` y lo pasa.

**Tech Stack:** numpy, pytest. No dependencias nuevas.

---

## Archivos a modificar

- **`tests/test_pure_pursuit.py`** — 4 tests nuevos al final del archivo (línea 90+)
- **`src/control/pure_pursuit.py`** — nueva constante, nuevo método `_centro_desde_ll`, firma modificada de `calcular_giro`
- **`scripts/ejecutar_piloto.py`** — aplicar ROI a `ll_mask_cache` y pasarlo a `calcular_giro`

---

## Task 1: Tests para ll_mask (TDD — escribir antes de implementar)

**Files:**
- Modify: `tests/test_pure_pursuit.py` (append después de línea 90)

- [ ] **Step 1: Agregar los 4 tests al final de `tests/test_pure_pursuit.py`**

Abrir `tests/test_pure_pursuit.py` y agregar exactamente este bloque al final del archivo (después de `test_ultimo_punto_debug_dentro_de_la_imagen`):

```python


def test_ll_mask_simetrico_error_cero():
    """Líneas equidistantes del centro del frame → centro_carril = x_camion → error ≈ 0."""
    pp = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[100:480, 0:640] = 1
    ll = np.zeros((480, 640), dtype=np.uint8)
    ll[300:480, 215:225] = 1   # borde izquierdo  (max = 224)
    ll[300:480, 415:425] = 1   # borde derecho    (min = 415)
    # centro_carril = (224 + 415) / 2 = 319.5 ≈ x_camion=320 → error ≈ 0
    error, perdido = pp.calcular_giro(da, ll)
    assert not perdido
    assert abs(error) < 0.05


def test_ll_mask_corrige_bias_da_mask():
    """
    da_mask asimétrica (centroide ≈ 420 → error negativo).
    ll_mask dice centro ≈ 320 → error ≈ 0.
    """
    pp_con_ll = PurePursuitVisual()
    pp_sin_ll = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[200:480, 200:640] = 1   # centroide ≈ 419 (a la derecha del frame)
    ll = np.zeros((480, 640), dtype=np.uint8)
    ll[300:480, 215:225] = 1   # borde izq ≈ 220
    ll[300:480, 415:425] = 1   # borde der ≈ 420 → centro ≈ 320

    error_con, _ = pp_con_ll.calcular_giro(da, ll)
    error_sin, _ = pp_sin_ll.calcular_giro(da)

    assert abs(error_con) < 0.05    # ll_mask: camión centrado
    assert error_sin < -0.10        # da_mask sola: sesgo negativo persistente


def test_ll_mask_vacio_usa_da_mask():
    """ll_mask sin píxeles → resultado idéntico a no pasar ll_mask."""
    pp1 = PurePursuitVisual()
    pp2 = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[100:480, 50:300] = 1    # área asimétrica
    ll = np.zeros((480, 640), dtype=np.uint8)   # vacía

    e1, _ = pp1.calcular_giro(da)
    e2, _ = pp2.calcular_giro(da, ll)
    assert e1 == pytest.approx(e2, rel=0.01)


def test_ll_mask_pocos_pixeles_no_activa_ll():
    """Menos de _MIN_LL_PIXELES (15) por lado → cae a da_mask, misma señal."""
    pp_ll = PurePursuitVisual()
    pp_da = PurePursuitVisual()
    da = np.zeros((480, 640), dtype=np.uint8)
    da[100:480, 0:640] = 1     # da simétrica
    ll = np.zeros((480, 640), dtype=np.uint8)
    ll[350, 100] = 1            # 1 pixel izq — insuficiente (< 15)
    ll[350, 540] = 1            # 1 pixel der — insuficiente (< 15)

    e_ll, _ = pp_ll.calcular_giro(da, ll)
    e_da, _ = pp_da.calcular_giro(da)
    assert e_ll == pytest.approx(e_da)
```

- [ ] **Step 2: Confirmar que los 4 tests FALLAN antes de implementar**

```
pytest tests/test_pure_pursuit.py::test_ll_mask_simetrico_error_cero tests/test_pure_pursuit.py::test_ll_mask_corrige_bias_da_mask tests/test_pure_pursuit.py::test_ll_mask_vacio_usa_da_mask tests/test_pure_pursuit.py::test_ll_mask_pocos_pixeles_no_activa_ll -v
```

Salida esperada: **4 FAILED** con `TypeError: calcular_giro() takes 2 positional arguments but 3 were given`.

Si el error es diferente, parar e investigar antes de continuar.

- [ ] **Step 3: Confirmar que los 8 tests existentes siguen pasando**

```
pytest tests/test_pure_pursuit.py -v -k "not ll_mask"
```

Salida esperada: **8 passed**.

---

## Task 2: Implementar `_centro_desde_ll` y modificar `calcular_giro`

**Files:**
- Modify: `src/control/pure_pursuit.py`

- [ ] **Step 1: Agregar `_MIN_LL_PIXELES` a las constantes de clase**

En `src/control/pure_pursuit.py`, localizar el bloque de constantes de clase que termina con `_ESCALA_ERROR`:

```python
    _ESCALA_ERROR = 0.40   # normalización intermedia (look-ahead a distancia media)
```

Reemplazar ese bloque completo con:

```python
    _ESCALA_ERROR = 0.40   # normalización intermedia (look-ahead a distancia media)
    _MIN_LL_PIXELES = 15   # píxeles mínimos por lado en ll_mask para activar nivel 1
```

- [ ] **Step 2: Modificar la firma de `calcular_giro`**

Localizar esta línea en `src/control/pure_pursuit.py`:

```python
    def calcular_giro(self, mascara_camino: np.ndarray) -> tuple[float, bool]:
```

Reemplazarla con:

```python
    def calcular_giro(self, mascara_camino: np.ndarray, ll_mask: np.ndarray | None = None) -> tuple[float, bool]:
```

- [ ] **Step 3: Agregar la lógica de nivel 1 (ll_mask) dentro de `calcular_giro`**

Localizar este bloque exacto en `calcular_giro` (justo después de la línea `fila_max = int(...)`):

```python
        # Barrido: intenta fila_base; si no hay verde, baja _SWEEP_PX y reintenta
        x_sum, w_sum, fila_usada = 0.0, 0.0, fila_base
```

Reemplazar por:

```python
        # Nivel 1: ll_mask — bordes pintados del carril actual
        if ll_mask is not None:
            filas_ll = [max(0, min(fila_base + off, alto - 1)) for off in offsets]
            centro_ll = self._centro_desde_ll(ll_mask, filas_ll, x_camion)
            if centro_ll is not None:
                self._ultimo_punto = (int(round(centro_ll)), fila_base)
                dx = x_camion - centro_ll
                error = float(np.clip(dx / (ancho * self._ESCALA_ERROR), -1.0, 1.0))
                self._ultimo_error = error
                return error, False

        # Nivel 2: centroide da_mask con barrido adaptativo
        x_sum, w_sum, fila_usada = 0.0, 0.0, fila_base
```

- [ ] **Step 4: Agregar el método `_centro_desde_ll` al final de la clase**

Al final de `src/control/pure_pursuit.py`, después del método `_estimar_curvatura`, agregar:

```python
    def _centro_desde_ll(
        self,
        ll_mask: np.ndarray,
        filas: list[int],
        x_camion: int,
    ) -> float | None:
        """
        Centro geométrico del carril actual desde ll_mask.

        Acumula píxeles de las filas dadas, los separa en izquierda/derecha
        respecto a x_camion (= frame center) y retorna (max_izq + min_der) / 2.
        En vías de N carriles, max_izq y min_der son siempre los bordes del
        carril actual (los más cercanos al camión), no los de los carriles vecinos.
        Retorna None si algún lado tiene menos de _MIN_LL_PIXELES píxeles.
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

- [ ] **Step 5: Verificar que los 4 tests nuevos pasan ahora**

```
pytest tests/test_pure_pursuit.py::test_ll_mask_simetrico_error_cero tests/test_pure_pursuit.py::test_ll_mask_corrige_bias_da_mask tests/test_pure_pursuit.py::test_ll_mask_vacio_usa_da_mask tests/test_pure_pursuit.py::test_ll_mask_pocos_pixeles_no_activa_ll -v
```

Salida esperada: **4 passed**.

- [ ] **Step 6: Verificar que los 8 tests existentes siguen pasando**

```
pytest tests/test_pure_pursuit.py -v
```

Salida esperada: **12 passed, 0 failed**.

- [ ] **Step 7: Commit**

```bash
git add src/control/pure_pursuit.py tests/test_pure_pursuit.py
git commit -m "feat: usar ll_mask para centro exacto del carril (nivel 1 con fallback a da_mask)"
```

---

## Task 3: Pasar ll_mask desde ejecutar_piloto.py

**Files:**
- Modify: `scripts/ejecutar_piloto.py` (sección del bucle principal, ~líneas 291-295)

**Contexto:** `ll_mask_cache` ya existe y se llena en línea 284. La variable `ll_mask = ll_mask_cache` se asigna en línea 286 y se usa en línea 348 para el debug image. Hay que crear `ll_mask_roi` (copia con ROI) para pasarlo a `calcular_giro` sin tocar `ll_mask` original (que sigue usándose para el debug sin ROI).

- [ ] **Step 1: Aplicar ROI a ll_mask y pasarlo a `calcular_giro`**

Localizar este bloque exacto en `scripts/ejecutar_piloto.py`:

```python
            # Enmascarar zona superior (60%): espejos virtuales ocupan hasta y≈55%.
            # Margen extra de 5% evita que variaciones de ángulo/resolución los expongan.
            # La carretera útil está en el 65-85% de la imagen.
            _fila_roi = int(da_mask.shape[0] * 0.60)
            da_mask[:_fila_roi, :] = 0

            # Pure Pursuit: bias derecho + look-ahead dinámico
            giro_pure_pursuit, carril_perdido = pure_pursuit.calcular_giro(da_mask)
```

Reemplazar por:

```python
            # Enmascarar zona superior (60%): espejos virtuales ocupan hasta y≈55%.
            # Margen extra de 5% evita que variaciones de ángulo/resolución los expongan.
            # La carretera útil está en el 65-85% de la imagen.
            _fila_roi = int(da_mask.shape[0] * 0.60)
            da_mask[:_fila_roi, :] = 0
            ll_mask_roi = ll_mask.copy()
            ll_mask_roi[:_fila_roi, :] = 0

            # Pure Pursuit: ll_mask (nivel 1) con fallback a da_mask (nivel 2)
            giro_pure_pursuit, carril_perdido = pure_pursuit.calcular_giro(da_mask, ll_mask_roi)
```

- [ ] **Step 2: Verificar que todos los tests siguen pasando**

```
pytest tests/test_pure_pursuit.py -v
```

Salida esperada: **12 passed**.

- [ ] **Step 3: Verificar que el script no tiene errores de sintaxis**

```
python -m py_compile scripts/ejecutar_piloto.py && echo "OK"
```

Salida esperada: `OK`. Si hay SyntaxError, revisar el bloque editado en Step 1.

- [ ] **Step 4: Commit**

```bash
git add scripts/ejecutar_piloto.py
git commit -m "feat: pasar ll_mask con ROI a pure_pursuit para centro de carril exacto"
```

---

## Verificación final

- [ ] **Correr suite completa**

```
pytest tests/test_pure_pursuit.py -v
```

Salida esperada:
```
tests/test_pure_pursuit.py::test_carril_perdido_devuelve_True_y_error_cero_inicial PASSED
tests/test_pure_pursuit.py::test_decaimiento_memoria_tras_perder_carril PASSED
tests/test_pure_pursuit.py::test_via_ancha_sin_sesgo_error_cercano_a_cero PASSED
tests/test_pure_pursuit.py::test_area_solo_izquierda_error_positivo PASSED
tests/test_pure_pursuit.py::test_area_solo_derecha_error_negativo PASSED
tests/test_pure_pursuit.py::test_error_acotado_entre_menos1_y_1 PASSED
tests/test_pure_pursuit.py::test_ultimo_punto_debug_none_cuando_carril_perdido PASSED
tests/test_pure_pursuit.py::test_ultimo_punto_debug_dentro_de_la_imagen PASSED
tests/test_pure_pursuit.py::test_ll_mask_simetrico_error_cero PASSED
tests/test_pure_pursuit.py::test_ll_mask_corrige_bias_da_mask PASSED
tests/test_pure_pursuit.py::test_ll_mask_vacio_usa_da_mask PASSED
tests/test_pure_pursuit.py::test_ll_mask_pocos_pixeles_no_activa_ll PASSED

12 passed in X.XXs
```
