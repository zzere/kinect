import serial
import subprocess

def ejecutar_comando():
    try:
        comando = "mpremote fs cp si.py :m.py"
        # Ejecuta el comando en la terminal y captura la salida
        resultado = subprocess.run(comando, shell=True, check=True, text=True, capture_output=True)
        print("Salida:")
        print(resultado.stdout)
    except subprocess.CalledProcessError as e:
        print("Error al ejecutar el comando:")
        print(e.stderr)

def esp32_conectado():
    # Ajusta el puerto según tu sistema (Linux suele ser /dev/ttyUSB0 o /dev/ttyACM0)
    esp = serial.Serial('/dev/ttyACM0', baudrate=115200, timeout=1)

    print("⚡ Conectado al ESP32C6. Escribe comandos MicroPython:")
    try:
        while True:
            cmd = input(">>> ")
            if cmd in ["exit", "quit", "salirP"]:
                break
            esp.write((cmd + "\r\n").encode())
            respuesta = esp.read_all().decode(errors="ignore")
            print(respuesta)
    except KeyboardInterrupt:
        esp.close()

if __name__ == "__main__":
    ejecutar_comando()
    esp32_conectado()
