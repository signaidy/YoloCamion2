"""Script de diagnóstico mínimo del gamepad virtual.

Crea un VX360Gamepad y aplica secuencialmente:
  1. RT=255 (acelerador al máximo) durante 5 segundos
  2. Pausa de 2 segundos (RT=0)
  3. LT=200 (freno) durante 2 segundos
  4. Pausa de 1 segundo
  5. Stick izquierdo X=-0.5 (giro izquierda) durante 2 segundos
  6. Stick izquierdo X=+0.5 (giro derecha) durante 2 segundos
  7. Libera todo

Uso:
  1. Abre ETS2, carga una partida, asegúrate de que el camión esté en marcha
     y el freno de mano SUELTO (presiona Space en ETS2 para soltar el freno).
  2. Cambia al juego.
  3. Corre: python scripts/diagnostico_gamepad.py
  4. Observa si el camión responde a cada paso.

Diagnóstico:
  - Si el camión acelera en paso 1: RT funciona, el problema es en el pipeline.
  - Si el camión NO acelera en paso 1 pero el timón sí gira en paso 5/6:
      ETS2 reconoce el gamepad pero RT no está mapeado como acelerador.
  - Si nada responde: el gamepad virtual no está siendo detectado por ETS2.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def esperar(segundos: float, mensaje: str) -> None:
    print(f"[{segundos:.0f}s] {mensaje}", flush=True)
    time.sleep(segundos)


def main() -> None:
    print("Iniciando diagnóstico de gamepad virtual...")

    # Gamepad se crea PRIMERO para que ETS2 lo detecte y registre completamente
    # antes de que lleguen los inputs. Si se crea después del countdown, ETS2
    # recibe los datos antes de terminar de inicializar el device y los descarta.
    import vgamepad as vg
    gp = vg.VX360Gamepad()
    print("Gamepad virtual creado (VX360).")
    print("ETS2 debería mostrar 'Controller Connected' ahora.\n")

    print("Tienes 8 segundos para cambiar al juego ETS2.")
    print("(El gamepad ya está conectado — ETS2 lo está registrando)")
    for i in range(8, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)
    print("¡Empezando!\n")

    # ── Paso 0: Botón A (digital) — verifica que ETS2 lee el gamepad ─────────
    print("=== PASO 0: Botón A presionado 3 veces (1 segundo cada vez) ===")
    print("    ¿ETS2 reacciona? (bocina, menú, cualquier cosa)")
    for _ in range(3):
        gp.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        gp.update()
        time.sleep(1.0)
        gp.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        gp.update()
        time.sleep(0.5)

    esperar(1.0, "Pausa tras botón A")

    # ── Paso 1: RT al máximo (acelerador) ────────────────────────────────────
    print("=== PASO 1: RT=255 (acelerador 100%) durante 5 segundos ===")
    print("    ¿El camión acelera?")
    gp.right_trigger(value=255)
    gp.left_trigger(value=0)
    gp.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
    gp.update()
    esperar(5.0, "RT activo...")

    # ── Pausa ─────────────────────────────────────────────────────────────────
    gp.right_trigger(value=0)
    gp.update()
    esperar(2.0, "Pausa (RT=0)")

    # ── Paso 2: LT (freno) ───────────────────────────────────────────────────
    print("\n=== PASO 2: LT=200 (freno ~78%) durante 2 segundos ===")
    print("    ¿El camión frena?")
    gp.left_trigger(value=200)
    gp.update()
    esperar(2.0, "LT activo...")
    gp.left_trigger(value=0)
    gp.update()
    esperar(1.0, "Pausa")

    # ── Paso 3: Stick izquierdo ──────────────────────────────────────────────
    print("\n=== PASO 3: Stick X=-0.5 (giro izquierda) durante 2 segundos ===")
    print("    ¿El timón gira a la izquierda?")
    gp.left_joystick_float(x_value_float=-0.5, y_value_float=0.0)
    gp.update()
    esperar(2.0, "Stick izquierda...")

    print("\n=== PASO 4: Stick X=+0.5 (giro derecha) durante 2 segundos ===")
    print("    ¿El timón gira a la derecha?")
    gp.left_joystick_float(x_value_float=0.5, y_value_float=0.0)
    gp.update()
    esperar(2.0, "Stick derecha...")

    # ── Liberar todo ─────────────────────────────────────────────────────────
    gp.right_trigger(value=0)
    gp.left_trigger(value=0)
    gp.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
    gp.update()

    # ── Paso 5: Leer XInput directamente desde Windows ───────────────────────
    print("\n=== PASO 5: Verificación XInput (lectura desde Windows) ===")
    print("    Mandando RT=200, stick X=-0.7 y leyendo lo que Windows ve...\n")

    import ctypes
    try:
        xinput = ctypes.WinDLL("xinput1_4.dll")
    except OSError:
        xinput = ctypes.WinDLL("xinput9_1_0.dll")

    class _Gamepad(ctypes.Structure):
        _fields_ = [
            ("wButtons",      ctypes.c_ushort),
            ("bLeftTrigger",  ctypes.c_ubyte),
            ("bRightTrigger", ctypes.c_ubyte),
            ("sThumbLX",      ctypes.c_short),
            ("sThumbLY",      ctypes.c_short),
            ("sThumbRX",      ctypes.c_short),
            ("sThumbRY",      ctypes.c_short),
        ]

    class _State(ctypes.Structure):
        _fields_ = [("dwPacketNumber", ctypes.c_uint), ("Gamepad", _Gamepad)]

    # Aplicar valores conocidos al gamepad virtual
    gp.right_trigger(value=200)
    gp.left_trigger(value=0)
    gp.left_joystick_float(x_value_float=-0.7, y_value_float=0.0)
    gp.update()
    time.sleep(0.1)  # dar tiempo al driver a propagar

    print(f"{'Slot':<6} {'Estado':<14} {'RT':<6} {'LT':<6} {'Stick X':<10} {'Botones'}")
    print("-" * 55)
    alguno_conectado = False
    for slot in range(4):
        state = _State()
        ret = xinput.XInputGetState(slot, ctypes.byref(state))
        if ret == 0:  # ERROR_SUCCESS
            alguno_conectado = True
            gp_data = state.Gamepad
            print(f"{slot:<6} {'CONECTADO':<14} {gp_data.bRightTrigger:<6} "
                  f"{gp_data.bLeftTrigger:<6} {gp_data.sThumbLX:<10} {gp_data.wButtons:#06x}")
        else:
            print(f"{slot:<6} {'desconectado':<14}")

    if not alguno_conectado:
        print("\n[!] Ningun slot XInput conectado — ViGEmBus no expone el gamepad via XInput.")
        print("    Solucion: reinstalar ViGEmBus.")
    else:
        print("\nInterpretacion:")
        print("  RT esperado: ~200  | Si aparece 0 -> XInput no refleja los valores (reinstalar ViGEmBus)")
        print("  Stick X esp: ~-23000 | Si aparece 0 -> mismo problema")
        print("  Si los valores coinciden -> XInput OK; el problema esta en los bindings de ETS2")

    # Liberar
    gp.right_trigger(value=0)
    gp.left_trigger(value=0)
    gp.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
    gp.update()

    print("\n=== DIAGNÓSTICO COMPLETADO ===")
    print("Reporta qué respondió y qué no para identificar la causa raíz.")


if __name__ == "__main__":
    main()
