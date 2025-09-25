#!/usr/bin/env python3
"""
Kinect + MediaPipe + PyTorch (CUDA) con POO
Actualizado: Guardado en carpeta 'prueba', imágenes 350x350
"""

import sys, os, time, math, datetime, string
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim

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
DATASET_DIR = "prueba"  # Carpeta principal para guardar imágenes
MODEL_FILE = "modelo_letras.pth"
PREDICTION_COOLDOWN = 1.0
IMAGE_SIZE = 350  # Tamaño de las imágenes procesadas

device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] PyTorch usando: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}, Memoria: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

# -----------------------------
# Modelo CNN
# -----------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*86*86, 128)  # 350x350 -> pooling x2 -> 86x86
        self.fc2 = nn.Linear(128, 26)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -----------------------------
# Clase principal del sistema
# -----------------------------
class KinectHandSystem:
    def __init__(self):
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
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
        self.ultima_prediccion_time = 0
        self.predicciones = []
        self.modo = 2  # default dataset
        self.model = None

    # -------------------------
    # Kinect
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
    # Procesamiento de dibujo
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
            1 if lm[4].x < lm[3].x else 0,
            1 if lm[8].y < lm[6].y else 0,
            1 if lm[12].y < lm[10].y else 0,
            1 if lm[16].y < lm[14].y else 0,
            1 if lm[20].y < lm[18].y else 0
        ]

    # -------------------------
    # Guardar dibujo en carpeta prueba
    # -------------------------
    def guardar_dibujo_dataset(self):
        if not self.selected_letter: return
        if time.time() - self.last_save_time < SAVE_COOLDOWN: return

        proc = self.procesar_dibujo(self.canvas)
        final_img = np.full_like(proc, 255)
        final_img[proc==0] = 0

        # Guardar en carpeta principal 'prueba' con subcarpetas por letra
        letter_dir = os.path.join(DATASET_DIR, self.selected_letter)
        os.makedirs(letter_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(letter_dir, f"dibujo_{timestamp}.png")
        cv2.imwrite(filename, final_img)
        print(f"[OK] Imagen guardada: {filename}")
        self.last_save_time = time.time()

    # -------------------------
    # Cargar modelo
    # -------------------------
    def cargar_modelo(self):
        self.model = CNNModel().to(device)
        if os.path.exists(MODEL_FILE):
            self.model.load_state_dict(torch.load(MODEL_FILE,map_location=device))
            self.model.eval()
            print("[INFO] Modelo cargado exitosamente")

    # -------------------------
    # Predecir letra
    # -------------------------
    def predecir_letra(self):
        proc = self.procesar_dibujo(self.canvas)
        proc_tensor = torch.FloatTensor(proc.reshape(1,1,IMAGE_SIZE,IMAGE_SIZE)/255.0).to(device)
        with torch.no_grad():
            output = self.model(proc_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs.data,1)
        letra = string.ascii_uppercase[pred.item()]
        return letra, conf.item()

    # -------------------------
    # Loop principal
    # -------------------------
    def run(self):
        if self.kinect_rgb_frame() is None:
            print("Kinect no conectada. Terminando.")
            return

        print("Modos: 1=Predicción, 2=Dataset, 3=Entrenamiento")
        print("ESC para salir")

        while True:
            frame = self.kinect_rgb_frame()
            if frame is None: break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    fingers = self.fingers_up(hand_landmarks)
                    palm_x = int((hand_landmarks.landmark[0].x + hand_landmarks.landmark[9].x)/2 * WIDTH)
                    palm_y = int((hand_landmarks.landmark[0].y + hand_landmarks.landmark[9].y)/2 * HEIGHT)
                    palm = (palm_x, palm_y)

                    if fingers == [0,1,0,0,0]:
                        self.estado = "Dibujo ON"
                        x = int(hand_landmarks.landmark[8].x * WIDTH)
                        y = int(hand_landmarks.landmark[8].y * HEIGHT)
                        if self.prev_point is not None:
                            cv2.line(self.canvas, self.prev_point, (x,y), (0,255,0), 4)
                        self.prev_point = (x,y)
                    elif sum(fingers)==0:
                        self.estado="Dibujo OFF"
                        self.prev_point=None
                    elif sum(fingers)==5:
                        if self.prev_palm is not None and math.hypot(palm[0]-self.prev_palm[0], palm[1]-self.prev_palm[1])>50:
                            if self.modo==2: self.guardar_dibujo_dataset()
                            elif self.modo==1 and self.model:
                                letra, conf = self.predecir_letra()
                                self.predicciones.append(letra)
                                print(f"[PRED] {letra} ({conf:.2%})")
                            self.canvas[:]=0
                            self.estado="Borrado"
                        self.prev_point=None
                    self.prev_palm=palm
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            combined = cv2.addWeighted(frame,1,self.canvas,1,0)
            cv2.putText(combined,f"Modo: {self.modo} - {self.estado}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            if self.modo==2:
                letra_texto = f"Letra: {self.selected_letter}" if self.selected_letter else "Letra no seleccionada"
            elif self.modo==1:
                letra_texto = f"Predicciones: {''.join(self.predicciones)}"
            else:
                letra_texto = self.mensaje_superior
            cv2.putText(combined, letra_texto, (10,50), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
            cv2.imshow("Kinect Sistema", combined)

            key = cv2.waitKey(1)&0xFF
            if key==27: break
            elif key in [ord("1"), ord("2"), ord("3")]:
                self.modo=int(chr(key))
                if self.modo==1 and self.model is None:
                    self.cargar_modelo()
            elif self.modo==2 and chr(key).lower() in string.ascii_lowercase:
                self.selected_letter=chr(key).lower()
                print(f"[INFO] Letra seleccionada: {self.selected_letter}")

        self.hands.close()
        cv2.destroyAllWindows()

# -----------------------------
# Ejecutar
# -----------------------------
if __name__=="__main__":
    system = KinectHandSystem()
    system.run()
