from machine import Pin, PWM
import time

# Configuración de pines y rangos máximos
dedos_config = {
    "pulgar":   {"pin": 6, "rang_max": 2500},
    "indice":   {"pin": 7, "rang_max": 2500},
    "medio":    {"pin": 0, "rang_max": 2500},
    "anular":   {"pin": 1, "rang_max": 2500},
    "menique":  {"pin": 8, "rang_max": 2500},
    "pulgar2":  {"pin": 10, "rang_max": 1500},
}

# Inicializar servos
for dedo, cfg in dedos_config.items():
    cfg["servo"] = PWM(Pin(cfg["pin"]), freq=50)

# Funciones generales
def move(servo, us):
    duty = int(us * 1023 // 20000)  # convertir microsegundos a duty (10 bits)
    servo.duty(duty)
    print(f"Pulso: {us} us -> duty: {duty}")

def move_max(servo, rang_max):
    move(servo, rang_max)

def move_min(servo):
    move(servo, 500)

# Función unificada para mover cualquier dedo
def mover_dedo(nombre, estado):
    """ Mueve el dedo al máximo si estado=1, al mínimo si estado=0 """
    dedo = dedos_config.get(nombre)
    if not dedo:
        print(f"Dedo '{nombre}' no encontrado")
        return

    if estado:
        move_max(dedo["servo"], dedo["rang_max"])
    else:
        move_min(dedo["servo"])


while True:
    cmd = input("Ingresa comando (e.g., 'pulgar 1' o 'indice 0') o 'salir': ")
    if cmd.lower() == "salir":
        for cfg in dedos_config.values():
            cfg["servo"].deinit()
        print("Servos deshabilitados.")
        break

    try:
        dedo_nombre, estado_str = cmd.split()
        estado = int(estado_str)
        if estado not in (0, 1):
            raise ValueError
        mover_dedo(dedo_nombre, estado)
    except ValueError:
        print("Comando inválido. Usa formato: 'dedo estado' (e.g., 'pulgar 1')")
    except Exception as e:
        print(f"Error: {e}")