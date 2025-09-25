#!/usr/bin/env python3
"""
Kinect + MediaPipe + PyTorch (CUDA) con POO + GUI
- Guardado en carpeta 'prueba', imágenes 350x350
- Gestos:
  - Pulgar hacia interior: borrar canvas
  - Pulgar 2s: borrar última letra
  - Pulgar + meñique: predecir letra
  - Índice + medio: guardar predicciones, generar MP3 y salir
  - Solo meñique: cambiar color línea
- Botones: dibujar, borrar, color, guardar
- Feedback visual y hover de 2s para activar botones
"""

import sys, os, time, math, datetime, string
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from elevenlabs import ElevenLabs, save

try:
    import freenect
except Exception:
    print("Kinect no conectada o librería freenect no disponible.")
    sys.exit(1)

# -----------------------------
# Configuración global
# -----------------------------
WIDTH, HEIGHT = 640, 480
SAVE_COOLDOWN = 2
DATASET_DIR = "prueba"
MODEL_FILE = "modelo_letras.pth"
PREDICTION_COOLDOWN = 1.0
IMAGE_SIZE = 350
GESTURE_HOLD_TIME = 1.0

device = "cpu"

# Transformaciones para ResNet
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# -----------------------------
# Clase principal
# -----------------------------
class KinectHandSystem:
    def __init__(self, elevenlabs_api_key=None):
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

        # Estado
        self.canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.prev_point = None
        self.prev_palm = None
        self.selected_letter = None
        self.estado = "Dibujo OFF"
        self.mensaje_superior = ""
        self.last_save_time = 0
        self.last_prediction_time = 0
        self.predicciones = []
        self.modo = 1
        self.model = None
        self.letters = []

        # Gestos
        self.gesto_borrar_letra_start = None
        self.feedback_gesto = ""
        self.button_hover_start = None
        self.gesto_guardar_predicciones_start = None
        self.line_color = (0,255,0)  # Verde inicial

        # Imagenes guardadas
        self.saved_images = []

        # ElevenLabs
        self.client = ElevenLabs(api_key=elevenlabs_api_key) if elevenlabs_api_key else None

        # Botones GUI
        self.buttons = {
            "dibujar": (10,400,110,440,"Dibujar"),
            "borrar": (120,400,220,440,"Borrar"),
            "color": (230,400,330,440,"Color"),
            "guardar": (340,400,460,440,"Guardar")
        }

        if self.modo == 1:
            self.cargar_modelo()

    # -------------------------
    # Kinect frame
    # -------------------------
    def kinect_rgb_frame(self):
        try:
            res = freenect.sync_get_video()
            if not res: return None
            frame = cv2.cvtColor(res[0], cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            return cv2.flip(frame, 1)
        except Exception:
            return None

    # -------------------------
    # Procesar dibujo
    # -------------------------
    @staticmethod
    def procesar_dibujo(canvas):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
        blurred = cv2.GaussianBlur(resized, (3,3), 0)
        _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
        return binary

    # -------------------------
    # Dedos levantados
    # -------------------------
    @staticmethod
    def fingers_up(hand_landmarks):
        lm = hand_landmarks.landmark
        return [
            1 if lm[4].x < lm[3].x else 0,    # pulgar
            1 if lm[8].y < lm[6].y else 0,    # índice
            1 if lm[12].y < lm[10].y else 0,  # medio
            1 if lm[16].y < lm[14].y else 0,  # anular
            1 if lm[20].y < lm[18].y else 0   # meñique
        ]

    # -------------------------
    # Gestos
    # -------------------------
    def gesto_borrar_ultima_letra(self, fingers):
        return fingers == [1,0,0,0,0]

    def gesto_predecir(self, fingers):
        return fingers == [1,0,0,0,1]

    def gesto_guardar_predicciones(self, fingers):
        return fingers == [0,1,1,0,0]

    # -------------------------
    # Borrar canvas
    # -------------------------
    def borrar_canvas(self):
        self.canvas[:] = 0
        self.estado = "Borrado"
        print("[INFO] Canvas borrado")

    # -------------------------
    # Guardar predicciones en txt y generar MP3
    # -------------------------
    def guardar_predicciones_txt(self):
        if not self.predicciones:
            print("[INFO] No hay predicciones para guardar.")
            return
        os.makedirs("resultados", exist_ok=True)
        filename = os.path.join("predicciones.txt")
        with open(filename, "w") as f:
            f.write("".join(self.predicciones))
        print(f"[OK] Predicciones guardadas en {filename}")
        self.mensaje_superior = f"Guardado en {filename}"

        # Generar MP3 si client está definido
        if self.client:
            self.txt_a_mp3_elevenlabs(filename, "predicciones.mp3")

        # Guardar imagen final combinando todas las imágenes
        if self.saved_images:
            imgs = [Image.fromarray(img) for img in self.saved_images]
            widths, heights = zip(*(i.size for i in imgs))
            total_width = sum(widths)
            max_height = max(heights)
            new_im = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for im in imgs:
                new_im.paste(im, (x_offset,0))
                x_offset += im.width
            new_im.save("imagen_final.png")
            print("[INFO] Imagen final creada: imagen_final.png")

        self.predicciones = []

    # -------------------------
    # Cargar modelo
    # -------------------------
    def cargar_modelo(self):
        try:
            if os.path.exists("Dataset"):
                self.letters = sorted([f for f in os.listdir("Dataset") 
                                       if os.path.isdir(os.path.join("Dataset", f))])
            else:
                self.letters = list(string.ascii_lowercase)
            print(f"[INFO] Letras detectadas: {self.letters}")
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.letters))
            self.model = self.model.to(device)
            if os.path.exists(MODEL_FILE):
                self.model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
                self.model.eval()
                print("[INFO] Modelo ResNet cargado exitosamente")
            else:
                print(f"[ERROR] No se encontró el archivo del modelo: {MODEL_FILE}")
                self.model = None
        except Exception as e:
            print(f"[ERROR] Error al cargar el modelo: {e}")
            self.model = None

    # -------------------------
    # Predecir letra
    # -------------------------
    def predecir_letra(self):
        if not self.model or not self.letters:
            return "?", 0.0
        current_time = time.time()
        if current_time - self.last_prediction_time < PREDICTION_COOLDOWN:
            return None, None
        proc = self.procesar_dibujo(self.canvas)
        pil_image = Image.fromarray(proc)
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs.data, 1)
        letra = self.letters[pred.item()] if pred.item() < len(self.letters) else "?"
        self.last_prediction_time = current_time

        # Guardar imagen usada para predicción
        self.saved_images.append(proc.copy())
        return letra, conf.item()

    # -------------------------
    # Generar MP3 con ElevenLabs
    # -------------------------
    def txt_a_mp3_elevenlabs(self, ruta_txt, ruta_mp3):
        if not self.client:
            print("❌ No hay cliente ElevenLabs configurado.")
            return
        try:
            with open(ruta_txt, 'r', encoding='utf-8') as f:
                texto = f.read()
            response = self.client.text_to_speech.convert(
                voice_id="nPczCjzI2devNBz1zQrb",
                model_id="eleven_multilingual_v2",
                text=texto
            )
            save(response, ruta_mp3)
            print(f"✅ Voz generada y guardada en {ruta_mp3}")
        except Exception as e:
            print(f"⚠️ Error generando MP3: {e}")

    # -------------------------
    # Loop principal
    # -------------------------
    def run(self):
        if self.kinect_rgb_frame() is None:
            print("Kinect no conectada. Terminando.")
            return

        while True:
            frame = self.kinect_rgb_frame()
            if frame is None: break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            self.feedback_gesto = ""
            index_point = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                fingers = self.fingers_up(hand_landmarks)
                index_point = (int(hand_landmarks.landmark[8].x*WIDTH),
                               int(hand_landmarks.landmark[8].y*HEIGHT))

                # Solo meñique = cambiar color
                if fingers == [0,0,0,0,1]:
                    if self.line_color == (0,255,0):
                        self.line_color = (0,0,255)
                        self.feedback_gesto = "Color Rojo ✅"
                    else:
                        self.line_color = (0,255,0)
                        self.feedback_gesto = "Color Verde ✅"

                # Pulgar solo = borrar canvas
                if self.gesto_borrar_ultima_letra(fingers) and sum(fingers)==1:
                    self.borrar_canvas()
                    self.estado = "Borrar canvas"
                    self.prev_point = None

                # Pulgar 2s = borrar última letra
                if self.gesto_borrar_ultima_letra(fingers):
                    if self.gesto_borrar_letra_start is None:
                        self.gesto_borrar_letra_start = time.time()
                    elif time.time()-self.gesto_borrar_letra_start>=GESTURE_HOLD_TIME:
                        if self.predicciones:
                            removed=self.predicciones.pop()
                            print(f"[INFO] Última letra borrada: {removed}")
                        self.feedback_gesto="Última letra borrada ✅"
                        self.gesto_borrar_letra_start=None
                else:
                    self.gesto_borrar_letra_start=None

                # Pulgar + meñique = predecir
                if self.gesto_predecir(fingers):
                    letra, conf = self.predecir_letra()
                    if letra is not None and conf is not None:
                        self.predicciones.append(letra)
                        print(f"[PRED] {letra} ({conf:.2%})")
                        self.feedback_gesto = f"Predicción: {letra} ✅"

                # Índice + medio = guardar y salir
                if self.gesto_guardar_predicciones(fingers):
                    if self.gesto_guardar_predicciones_start is None:
                        self.gesto_guardar_predicciones_start = time.time()
                    elif time.time()-self.gesto_guardar_predicciones_start>=GESTURE_HOLD_TIME:
                        self.guardar_predicciones_txt()
                        print("[INFO] Programa finalizado tras guardar predicciones.")
                        cv2.destroyAllWindows()
                        self.hands.close()
                        sys.exit(0)

                # Dibujar con índice
                if fingers==[0,1,0,0,0]:
                    x,y=index_point
                    if self.prev_point is not None:
                        cv2.line(self.canvas,self.prev_point,(x,y),self.line_color,4)
                    self.prev_point=(x,y)
                    self.estado="Dibujo ON"
                else:
                    self.prev_point=None
                    self.estado="Dibujo OFF"

            # Botones GUI
            for b, coords in self.buttons.items():
                x1,y1,x2,y2,text = coords
                cv2.rectangle(frame,(x1,y1),(x2,y2),(100,100,100),-1)
                cv2.putText(frame,text,(x1+5,y1+30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
                if index_point and x1<=index_point[0]<=x2 and y1<=index_point[1]<=y2:
                    if self.button_hover_start is None:
                        self.button_hover_start=time.time()
                    elif time.time()-self.button_hover_start>=GESTURE_HOLD_TIME:
                        if b=="borrar":
                            self.borrar_canvas()
                        elif b=="guardar":
                            self.guardar_predicciones_txt()
                        elif b=="color":
                            self.line_color = (0,0,255) if self.line_color==(0,255,0) else (0,255,0)
                        elif b=="dibujar":
                            self.estado="Dibujo ON"
                        self.button_hover_start=None
                else:
                    self.button_hover_start=None

            # Combinar canvas y frame
            combined=cv2.addWeighted(frame,1,self.canvas,1,0)
            cv2.putText(combined,f"Modo: {self.modo} - {self.estado}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            letra_texto=f"Predicciones: {''.join(self.predicciones[-10:])}"
            cv2.putText(combined,letra_texto,(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
            if self.feedback_gesto:
                cv2.putText(combined,self.feedback_gesto,(10,110),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

            cv2.imshow("Kinect Sistema",combined)
            key=cv2.waitKey(1)&0xFF
            if key==27: break

        self.hands.close()
        cv2.destroyAllWindows()


# -----------------------------
# Main
# -----------------------------
if __name__=="__main__":
    ELEVENLABS_API_KEY="sk_a9e5ec86b63fe701e969fb5024daa2f9294360f76af0a506"
    system=KinectHandSystem(elevenlabs_api_key=ELEVENLABS_API_KEY)
    system.run()
