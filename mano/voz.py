import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import sounddevice as sd
import numpy as np
import subprocess
import serial

# Modelo Wav2Vec2 para español
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Función para grabar audio de 1 segundo
def grabar_audio(fs=16000):
    duracion = 1  # siempre 1 segundo
    print("Grabando 1 segundo...")
    audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1)
    sd.wait()
    audio = audio.flatten()
    print("Grabación finalizada.")
    return audio

# Función para transcribir audio y mostrar solo la primera letra
def transcribir_audio(audio, fs=16000):
    input_values = processor(audio, sampling_rate=fs, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    
    # Obtener la primera letra
    primera_letra = transcription.strip()[0] if transcription else ""
    return primera_letra

# Menú principal en español
def menu():
    esp = Esp32()
    while True:
        print("\n---MENÚ---")
        print("1. Hablar y mostrar primera letra (1 segundo)")
        print("2. Salir")
        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            audio = grabar_audio()
            letra = transcribir_audio(audio)
            print(f"\nPrimera letra detectada: {letra}")
            esp.enviar_letra(letra)
        elif opcion == "2":
            print("Saliendo...")
            break
        elif opcion == "pr":
            esp.enviar_letra("a")
        else:
            print("Opción inválida, intente nuevamente.")

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
    menu()