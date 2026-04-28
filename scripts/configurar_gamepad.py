"""Configurador interactivo del gamepad virtual.

Mantiene un X360 virtual vivo y permite empujar manualmente cada eje
con el teclado, para que ETS2 los capture y los asigne al perfil
correcto (con el GUID del virtual, no de un mando fisico).

Teclas (mantener pulsadas mientras ETS2 espera el input):
    a   -> RT (acelerador) a fondo
    s   -> LT (freno) a fondo
    j   -> stick izquierdo X = -1 (volante a la izquierda)
    l   -> stick izquierdo X = +1 (volante a la derecha)
    b   -> boton A
    n   -> boton B
    q   -> salir

Procedimiento:
  1. Ejecuta este script. ETS2 ya debe estar abierto.
  2. ETS2 -> Opciones -> Controles -> Entradas:
       Primario  = Controller (Xbox 360 For Windows)
       Secundario = Teclado (o ninguno)
  3. Click en la celda "Acelerador" -> ETS2 dice "mueva el control".
  4. Vuelve a esta terminal (Alt+Tab) y MANTEN PULSADA la tecla 'a'.
  5. Vuelve a ETS2 (Alt+Tab) sin soltar la tecla. Capturara RT del virtual.
  6. Repite con 's' para Freno (LT), 'j'/'l' para Volante.
  7. Verifica que las barras de prueba responden cuando mantienes la tecla.
  8. Guarda el perfil. Cierra menu. Pulsa 'q' aqui para salir.
"""
import sys
import time

import vgamepad as vg
from pynput import keyboard


def main() -> None:
    gp = vg.VX360Gamepad()
    gp.update()
    print("=" * 60)
    print("  GAMEPAD VIRTUAL ACTIVO — modo configurador interactivo")
    print("=" * 60)
    print("  [a] RT a fondo       (Acelerador)")
    print("  [s] LT a fondo       (Freno)")
    print("  [j] Stick X = -1.0   (Volante izquierda)")
    print("  [l] Stick X = +1.0   (Volante derecha)")
    print("  [b] Boton A")
    print("  [n] Boton B")
    print("  [q] Salir")
    print()
    print("Manten pulsada la tecla mientras ETS2 captura el control.")
    print()

    estado = {"a": False, "s": False, "j": False, "l": False,
              "b": False, "n": False}
    salir = {"flag": False}

    def on_press(key):
        try:
            c = key.char.lower() if key.char else None
        except AttributeError:
            return
        if c == "q":
            salir["flag"] = True
            return False
        if c in estado:
            estado[c] = True

    def on_release(key):
        try:
            c = key.char.lower() if key.char else None
        except AttributeError:
            return
        if c in estado:
            estado[c] = False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        while not salir["flag"]:
            gp.right_trigger(value=255 if estado["a"] else 0)
            gp.left_trigger(value=255 if estado["s"] else 0)
            x = (-1.0 if estado["j"] else 0.0) + (1.0 if estado["l"] else 0.0)
            gp.left_joystick_float(x_value_float=x, y_value_float=0.0)
            if estado["b"]:
                gp.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            else:
                gp.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            if estado["n"]:
                gp.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            else:
                gp.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            gp.update()
            time.sleep(0.02)
    finally:
        gp.right_trigger(value=0)
        gp.left_trigger(value=0)
        gp.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        gp.update()
        listener.stop()
        print("\nGamepad liberado. Cierra ETS2 y vuelve a abrir el piloto.")


if __name__ == "__main__":
    sys.exit(main())
