
import subprocess
import serial
import socket

class Cliente:
    def __init__(self, host="127.0.0.1", port=5000):
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def conectar(self):
        """Conecta al servidor."""
        self.client_socket.connect((self.host, self.port))

    def enviar(self, mensaje: str,esp):
        """Envía un mensaje al servidor."""
        self.client_socket.sendall(mensaje.encode("utf-8"))
        try:
            #Desde archivo
            # test_audio = "prueba/B/B_1.wav"  # cambia por un audio de prueba
            # pred = predict_letter(test_audio)
            # print("🔤 Predicción archivo:", pred)

            # Desde micrófono (descomenta si quieres probar)
            esp32_con = esp
            l = mensaje
            esp32_con.enviar_letra(l)

        except FileNotFoundError as e:
            print(e)
            print("💡 Para predecir, entrena el modelo primero: train_model()")
        except KeyboardInterrupt:
            self.cerrar()


    def cerrar(self):
        """Cierra la conexión con el servidor."""
        self.client_socket.close()


class Esp32():
    def __init__(self):
        try:
            comando = "mpremote fs cp si.py :m.py"
            # Ejecuta el comando en la terminal y captura la salida
            resultado = subprocess.run(comando, shell=True, check=True, text=True, capture_output=True)
            print("Salida:")
            print(resultado.stdout)

            esp = serial.Serial('/dev/ttyACM0', baudrate=115200, timeout=1)
            print("⚡ Conectado al ESP32C6. Escribe comandos MicroPython:")
            esp.write(("import m\r\n").encode())
            respuesta = esp.read_all().decode(errors="ignore")
            print(respuesta)
            self.esp = esp
        except subprocess.CalledProcessError as e:
            print("Error al ejecutar el comando:")
            print(e.stderr)

    def close(self):
        self.enviar_letra("salir")
        self.esp.close()
        print("Conexión cerrada")
    
    def enviar_letra(self, letra):
        self.esp.write((f"{letra}\r\n").encode())
        respuesta = self.esp.read_all().decode(errors="ignore")
        print(respuesta)
    

if __name__ == "__main__":
    cliente = Cliente()
    cliente.conectar()
    esp = Esp32()
    letras = ['g','h','j','s','z']
    while True:
        msg = input("👉 Escribe un mensaje (o 'salir' para terminar): ")
        if msg.lower() == "salir":
            esp.close()
            break
        elif msg  in letras:
            print("letra no valida")
        else:
            cliente.enviar(msg,esp)

    cliente.cerrar()
    