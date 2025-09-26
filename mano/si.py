from machine import Pin, PWM
import time

# Configuración de pines y rangos máximos
dedos_config = {
    "pulgar":   {"pin": 6, "rang_max": 1800},
    "indice":   {"pin": 10, "rang_max": 2100},
    "medio":    {"pin": 8, "rang_max": 2200},
    "anular":   {"pin": 1, "rang_max": 2500},
    "menique":  {"pin": 0, "rang_max": 2500},
    "pulgar2":  {"pin": 7, "rang_max": 1500},
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

def mover_dedo_custom(nombre, estado, rm):
    """ Mueve el dedo al máximo si estado=1, al mínimo si estado=0 """
    dedo = dedos_config.get(nombre)
    if not dedo:
        print(f"Dedo '{nombre}' no encontrado")
        return

    if estado:
        move_max(dedo["servo"], rm)
    else:
        move_min(dedo["servo"])

def reset_all():
    mover_dedo("indice", 0)
    mover_dedo("medio", 0)
    mover_dedo("anular", 0)
    mover_dedo("menique", 0)
    time.sleep(0.5)
    mover_dedo("pulgar", 0)
    mover_dedo("pulgar2", 0)

def a():
    reset_all()
    mover_dedo("indice", 1)
    mover_dedo("medio", 1)
    mover_dedo("anular", 1)
    mover_dedo("menique", 1)
    #ime.sleep(2)

def b():
    reset_all()
    mover_dedo("pulgar", 1)
    #time.sleep(2)

def c():
    reset_all()
    mover_dedo_custom("pulgar", 1, 1000)
    mover_dedo_custom("indice", 1, 1250)
    mover_dedo_custom("medio", 1, 1500)
    mover_dedo_custom("anular", 1, 1500)
    mover_dedo_custom("menique", 1, 1500)
    mover_dedo("pulgar2", 1)

def d():        
    reset_all()
    mover_dedo_custom("pulgar", 1, 1500)
    mover_dedo_custom("medio", 1, 2000)
    mover_dedo_custom("anular", 1, 1500)
    mover_dedo_custom("menique", 1, 1800)
    mover_dedo_custom("pulgar2", 1, 1000)

def e():
    reset_all()
    mover_dedo_custom("pulgar", 1, 700)
    mover_dedo_custom("indice", 1, 700)
    mover_dedo_custom("medio", 1, 700)
    mover_dedo_custom("anular", 1, 700)
    mover_dedo_custom("menique", 1, 700)
    mover_dedo("pulgar2", 1)

def f():
    reset_all()
    mover_dedo("medio", 1)
    mover_dedo("anular", 1)
    mover_dedo("menique", 1)
    mover_dedo("pulgar2", 1)
    mover_dedo("pulgar", 1)

def i():
    reset_all()
    mover_dedo("medio", 1)
    mover_dedo("anular", 1)
    mover_dedo("indice", 1)
    mover_dedo("pulgar2", 1)
    time.sleep(0.5)
    mover_dedo_custom("pulgar", 1, 1000)

def k():
    reset_all()
    mover_dedo("menique", 1)
    mover_dedo("anular", 1)
    mover_dedo_custom("pulgar2", 1, 1300)
    time.sleep(0.5)
    mover_dedo("pulgar", 1)

def l():
    reset_all()
    mover_dedo("pulgar2", 1)
    mover_dedo("menique", 1)
    mover_dedo("anular", 1)
    mover_dedo("medio", 1)
def m():
    reset_all()
    mover_dedo("menique", 1)
    time.sleep(0.5)
    mover_dedo("pulgar", 1)
    mover_dedo("indice", 1)
    mover_dedo("medio", 1)
    mover_dedo("anular", 1)
def n():
    reset_all()
    mover_dedo("menique", 1)
    mover_dedo("anular", 1)
    time.sleep(0.5)
    mover_dedo_custom("pulgar", 1, 1200)
    mover_dedo("indice", 1)
    mover_dedo("medio", 1)
    
def o():
    reset_all()
    mover_dedo_custom("pulgar", 1, 1200)
    mover_dedo_custom("indice", 1, 2000)
    mover_dedo_custom("medio", 1, 2000)
    mover_dedo_custom("anular", 1, 2000)
    mover_dedo_custom("menique", 1, 2000)
    mover_dedo("pulgar2", 1)
def p():
    reset_all()
    mover_dedo("pulgar2", 1)
    mover_dedo("menique", 1)
    mover_dedo("anular", 1)
    mover_dedo_custom("medio", 1, 1200)
    mover_dedo_custom("indice", 1, 1700)
def q():
    reset_all()
    mover_dedo_custom("pulgar2", 1, 1200)
    mover_dedo("pulgar", 1)
    mover_dedo_custom("indice", 1, 1100)
    mover_dedo_custom("medio", 1, 1300)
    mover_dedo_custom("anular", 1, 1300)
    mover_dedo_custom("menique", 1, 1300)
def r():
    reset_all()
    mover_dedo("menique", 1)
    mover_dedo("anular", 1)
    mover_dedo_custom("pulgar2", 1, 1200)
    time.sleep(0.5)
    mover_dedo("pulgar", 1)
    mover_dedo_custom("medio", 1, 700)
def t():
    reset_all()
    mover_dedo("pulgar2", 1)
    mover_dedo_custom("indice", 1, 1500)
    time.sleep(0.5)
    mover_dedo_custom("pulgar", 1, 1300)
    mover_dedo_custom("medio", 1, 800)
    mover_dedo_custom("anular", 1, 900)
def u():
    reset_all()
    mover_dedo("medio", 1)
    mover_dedo("anular", 1)
    mover_dedo_custom("pulgar2", 1, 1300)
    time.sleep(0.5)
    mover_dedo_custom("pulgar", 1, 1700)
    #mover_dedo("pulgar", 1)
def v():
    reset_all()
    mover_dedo("menique", 1)
    mover_dedo("anular", 1)
    mover_dedo_custom("pulgar2", 1, 1200)
    time.sleep(0.5)
    mover_dedo("pulgar", 1)
def w():
    reset_all()
    mover_dedo("menique", 1)
    mover_dedo_custom("pulgar2", 1, 800)
    time.sleep(0.5)
    mover_dedo("pulgar", 1)
def x():
    reset_all()
    mover_dedo("pulgar", 1 )
    time.sleep(0.5)
    mover_dedo_custom("menique", 1, 2300)
    mover_dedo_custom("anular", 1, 2400)
    mover_dedo("medio", 1)
    mover_dedo_custom("indice", 1, 700)
def y():
    reset_all()
    mover_dedo("medio", 1)
    mover_dedo("anular", 1)
    mover_dedo("indice", 1)


# while True:
#     cmd = input("Ingresa comando (e.g., 'pulgar 1' o 'indice 0') o 'salir': ")
#     if cmd.lower() == "salir":
#         for cfg in dedos_config.values():
#             cfg["servo"].deinit()
#         print("Servos deshabilitados.")
#         break

#     try:
#         dedo_nombre, estado_str = cmd.split()
#         estado = int(estado_str)
#         if estado not in (0, 1):
#             raise ValueError
#         mover_dedo(dedo_nombre, estado)
#     except ValueError:
#         print("Comando inválido. Usa formato: 'dedo estado' (e.g., 'pulgar 1')")
#     except Exception as e:
#         print(f"Error: {e}")

#Diccionario que mapea letras a funciones
acciones = {
    "a": a,
    "b": b,
    "c": c,
    "d": d,
    "e": e,
    "f": f,
    "i": i,
    "k": k,
    "l": l,
    "m": m,
    "n": n,
    "o": o,
    "p": p,
    "q": q,
    "r": r,
    "t": t,
    "u": u,
    "v": v,
    "w": w,
    "x": x,
    "y": y,
}

while True:
    x = input("Letra: ").lower()  # normalizamos a minúscula
    if x == "salir":
        break
    elif x in acciones:
        acciones[x]()   # ejecuta la función asociada
    elif x == "reset":
        reset_all()
    elif x == "pr":
        while True:
            cmd = input("Ingresa comando (e.g., 'pulgar 1' o 'indice 0') o 'salir': ")
            if cmd.lower() == "salir":
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
    else:
        print("Letra no válida")



reset_all()