import os
import sounddevice as sd
from scipy.io.wavfile import write
import time

# Configuración
SAMPLE_RATE = 16000  # Frecuencia de muestreo (16kHz es estándar para voz)
DURATION = 1  # segundos de grabación por audio
OUTPUT_DIR = "backup/dataset2"  # carpeta base donde se guardarán las letras

def grabar_audio(letra, index):
    # Crear carpeta para la letra si no existe
    folder = os.path.join(OUTPUT_DIR, letra.upper())
    os.makedirs(folder, exist_ok=True)

    print(f"🎤 Grabando letra '{letra}' ({DURATION} seg)...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()  # esperar a que termine la grabación

    filename = os.path.join(folder, f"{letra}_{index}(1).wav")
    write(filename, SAMPLE_RATE, audio)
    print(f"✅ Guardado en {filename}")

if __name__ == "__main__":
    print("=== Grabador de dataset de letras ===")
    contador = {}

    while True:
        letra = input("Escribe una letra (o 'salir' para terminar): ").strip().upper()
        if letra == "SALIR":
            break
        if len(letra) != 1 or not letra.isalpha():
            print("⚠️ Solo se permiten letras individuales.")
            continue

        # Contador por letra
        contador[letra] = contador.get(letra, 0) + 1
        grabar_audio(letra, contador[letra])

        time.sleep(1)  # pequeña pausa antes de la siguiente grabación
