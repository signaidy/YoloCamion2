"""Mantiene el gamepad virtual activo para configurarlo en ETS2.

1. Corre este script (deja la terminal abierta)
2. En ETS2: Opciones -> Controles -> selecciona Xbox 360 Controller
3. Mapea: RT=Acelerador, LT=Freno, Stick Izq X=Volante
4. Guarda y cierra opciones
5. Ctrl+C aqui para terminar

El gamepad desaparece de Windows cuando este script termina.
Ejecuta SIEMPRE como Administrador si ETS2 corre como Admin.
"""
import sys
import time
import vgamepad as vg

print("=" * 50)
print("  GAMEPAD VIRTUAL ACTIVO")
print("=" * 50)
gp = vg.VX360Gamepad()
gp.update()
print("\nXbox 360 Controller conectado.")
print("Abre ETS2 -> Opciones -> Controles y configura los ejes.")
print("Presiona Ctrl+C cuando termines.\n")

# Pulsar los ejes cada 2s para que ETS2 los detecte
t = 0
try:
    while True:
        # Pulso suave en RT para que ETS2 detecte el eje
        if t % 10 == 0:
            gp.right_trigger(value=30)
            gp.update()
            time.sleep(0.1)
            gp.right_trigger(value=0)
            gp.update()
            print(f"  [{int(t)}s] Gamepad activo — pulso RT enviado")
        time.sleep(1)
        t += 1
except KeyboardInterrupt:
    gp.right_trigger(value=0)
    gp.left_trigger(value=0)
    gp.update()
    print("\nGamepad desconectado.")
