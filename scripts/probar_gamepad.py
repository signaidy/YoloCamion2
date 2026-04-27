"""Prueba directa del gamepad virtual — independiente del piloto.

Corre este script CON ETS2 ABIERTO Y EN FOCO.
Si el camión se mueve, el gamepad funciona y ETS2 lo detecta.
Si no se mueve, hay que configurar el gamepad en ETS2 primero.

Uso:
  python scripts/probar_gamepad.py            # prueba RT (acelerador) por 4s
  python scripts/probar_gamepad.py --teclado  # prueba tecla W en su lugar
"""
import argparse
import sys
import time


def probar_gamepad():
    import vgamepad as vg

    print("\nGamepad virtual — iniciando prueba de acelerador (RT)")
    print("Asegurate de que ETS2 esta en pantalla completa y enfocado")
    print("Motor encendido, freno de mano suelto\n")

    gp = vg.VX360Gamepad()

    for i in range(5, 0, -1):
        print(f"  Aplicando acelerador en {i}s...", flush=True)
        time.sleep(1)

    print(">>> ACELERADOR AL 60% por 4 segundos <<<")
    gp.right_trigger(value=int(0.6 * 255))  # RT = acelerador
    gp.update()
    time.sleep(4)

    print(">>> FRENO por 2 segundos <<<")
    gp.right_trigger(value=0)
    gp.left_trigger(value=int(0.8 * 255))   # LT = freno
    gp.update()
    time.sleep(2)

    print(">>> SOLTANDO todo <<<")
    gp.right_trigger(value=0)
    gp.left_trigger(value=0)
    gp.update()

    if hasattr(gp, 'reset'):
        gp.reset()

    print("\nPrueba completada.")
    print("- Si el camion se movio: gamepad OK, puedes usar --control gamepad")
    print("- Si NO se movio: configura el gamepad en ETS2 (Opciones -> Controles)")
    print("  o usa --control teclado que no necesita configuracion\n")


def probar_teclado():
    import pydirectinput

    print("\nTeclado — prueba de tecla W (avanzar)")
    print("Asegurate de que ETS2 esta en pantalla completa y enfocado\n")

    for i in range(5, 0, -1):
        print(f"  Presionando W en {i}s...", flush=True)
        time.sleep(1)

    print(">>> W presionada por 4 segundos <<<")
    pydirectinput.keyDown('w')
    time.sleep(4)
    pydirectinput.keyUp('w')

    print(">>> S presionada por 1 segundo (frenar) <<<")
    pydirectinput.keyDown('s')
    time.sleep(1)
    pydirectinput.keyUp('s')

    print("\nPrueba completada.")
    print("- Si el camion se movio: teclado OK, usa --control teclado")
    print("- Si NO se movio: ETS2 no recibio los inputs (ventana no enfocada?)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teclado", action="store_true", help="Probar teclado en lugar de gamepad")
    args = parser.parse_args()

    if args.teclado:
        probar_teclado()
    else:
        probar_gamepad()
