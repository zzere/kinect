import speech_recognition as sr
import serial
import subprocess
import time
from gtts import gTTS
import os

class DecirNombre:
    def decir(self, texto):
        tts = gTTS(text=f"Tu nombre es {texto}", lang='es', slow=False)
        tts.save("nombre.mp3")
        os.system("mpg123 nombre.mp3")
        os.remove("nombre.mp3")

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

class VoiceRecognizer:
    """Clase para capturar y reconocer voz."""
    def __init__(self, language='es-ES'):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.language = language

    def listen_and_recognize(self):
        """Escucha el micrófono y devuelve el texto reconocido."""
        with self.microphone as source:
            print("🎙️ Di tu nombre...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)

        try:
            texto = self.recognizer.recognize_google(audio, language=self.language)
            print(f"🗣️ Tu nombre es: {texto}")
            return texto
        except sr.UnknownValueError:
            print("❌ No se reconoció el nombre. Intenta de nuevo.")
            return None
        except sr.RequestError as e:
            print(f"⚠️ Error con el servicio de reconocimiento: {e}")
            return None

class NameHandler:
    """Clase para manejar la lectura y escritura del nombre."""
    def __init__(self, filename="nombre.txt"):
        self.filename = filename

    def save_name(self, name):
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(name)
        print(f"💾 Nombre guardado en '{self.filename}'")

    def read_name(self):
        with open(self.filename, "r", encoding="utf-8") as f:
            return f.read().strip()
        
def main():

    esp = Esp32()
    voice = VoiceRecognizer()
    nombre = NameHandler()
    letras = ['g','h','j','s','z']
    voz = DecirNombre()
    
    try:
        while True:
            canson = input("1 para grabar")
            
            if canson == "1":
                texto = voice.listen_and_recognize()

                if texto == None:
                    continue
                elif any(letra in letras for letra in texto):
                    print("hay una letra no valida en el nombre")
                else:
                    nombre.save_name(texto)
                    textoL = texto.lower()
                    textoL = list(textoL)
                    for letra in textoL:
                        esp.enviar_letra(letra)
                        print(letra)
                        time.sleep(4)
                    
                    voz.decir(texto)
            else:
                print("input no valido")
                break

    except KeyboardInterrupt:
        esp.close()
        print("programa finalizado")
    finally:
        esp.close()
        voz.decir("caaaaaaaansoooooooooooooooooooooooooooon")


if __name__ == "__main__":
    main()