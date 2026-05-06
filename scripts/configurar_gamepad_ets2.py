"""Mantiene el gamepad virtual activo Y lo mueve continuamente para que
ETS2 lo detecte y puedas mapearlo en Opciones -> Controles.

Ejecutar como Administrador:
  python scripts/configurar_gamepad_ets2.py

Pasos:
  1. Corre este script (deja la terminal abierta)
  2. En ETS2: Opciones -> Controles
  3. Cambia dispositivo a "Xbox 360 Controller" o "Gamepad"
  4. Haz clic en cada campo y mueve el gatillo/stick correspondiente:
     - Acelerador  -> mueve RT (gatillo derecho)
     - Freno       -> mueve LT (gatillo izquierdo)
     - Volante      -> mueve Stick Izquierdo a los lados
  5. Guarda la configuracion
  6. Ctrl+C aqui para salir
"""
import time
import vgamepad as vg

print("=" * 55)
print("  CONFIGURADOR DE GAMEPAD PARA ETS2")
print("=" * 55)
print()

gp = vg.VX360Gamepad()
gp.update()
print("Xbox 360 Controller CONECTADO.")
print("Ahora abre ETS2 -> Opciones -> Controles -> Joystick")
print("y mapea los ejes moviendo los gatillos/stick.")
print("\nCtrl+C para terminar.\n")

ciclo = 0
try:
    while True:
        ciclo += 1
        fase = ciclo % 6

        # Pulsar ejes cada 3s para facilitar el mapeo en ETS2
        if fase == 0:
            print(f"[{ciclo*3}s] Moviendo RT (acelerador)...", flush=True)
            for v in [0, 80, 150, 200, 150, 80, 0]:
                gp.right_trigger(value=v)
                gp.update()
                time.sleep(0.08)

        elif fase == 2:
            print(f"[{ciclo*3}s] Moviendo LT (freno)...", flush=True)
            for v in [0, 80, 150, 200, 150, 80, 0]:
                gp.left_trigger(value=v)
                gp.update()
                time.sleep(0.08)

        elif fase == 4:
            print(f"[{ciclo*3}s] Moviendo Stick Izq (volante)...", flush=True)
            for x in [0.0, -0.5, -1.0, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0]:
                gp.left_joystick_float(x_value_float=x, y_value_float=0.0)
                gp.update()
                time.sleep(0.08)
            gp.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            gp.update()

        time.sleep(3)

except KeyboardInterrupt:
    gp.right_trigger(value=0)
    gp.left_trigger(value=0)
    gp.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
    gp.update()
    print("\nGamepad desconectado. Configuracion guardada en ETS2.")
