from machine import Pin, PWM
import time

servo = None  # lo declaramos global para usarlo después

pulgar = 6
indice = 7
medio = 0
anular = 1
meñique = 8
pulgar2 = 10

# Rango típico de un SG90 (ajustable según tu servo)
min_us = 500    # Pulso mínimo (~0.5 ms)
max_us = 2500   # Pulso máximo (~2.5 ms)



def set_pulse(us):
    """
    Mueve el servo a partir de un pulso en microsegundos (500–2500).
    """
    duty = int(us * 1023 // 20000)  # convertir microsegundos a duty (resolución 10 bits)
    servo.duty(duty)
    print(f"Pulso: {us} us -> duty: {duty}")

# --- Selección del pin ---
while True:
    try:
        pin_number = int(input("👉 Ingresa el número de GPIO donde está conectado el servo: "))
        servo = PWM(Pin(pin_number), freq=50)  # 50 Hz típico de servos
        print(f"✅ Servo inicializado en GPIO{pin_number}")
        break
    except ValueError:
        print("⚠️ Ingresa un número válido.")
    except Exception as e:
        print(f"⚠️ Error: {e}, intenta con otro pin.")

# --- Control manual ---
print("Control manual del servo por pulsos (500 a 2500). Escribe 'salir' para terminar.")

while True:
    cmd = input("Pulso en us: ")
    if cmd.lower() == "salir":
        servo.deinit()
        print("Servo deshabilitado.")
        break
    try:
        us = int(cmd)
        if us < min_us or us > max_us:
            print(f"⚠️ El rango seguro es {min_us} - {max_us} us")
        else:
            set_pulse(us)
    except ValueError:
        print("⚠️ Ingresa un número válido o 'salir'")
