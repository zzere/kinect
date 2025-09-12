#!/usr/bin/env python3
"""
Kinect + MediaPipe + PyTorch (CUDA) con POO
Actualizado: Guardado en carpeta 'prueba', imágenes 350x350
Modo 1 (Predicción) ahora es el predeterminado.
"""

import sys, os, time, math, datetime, string
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image

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
PREDICTION_COOLDOWN = 1.0  # 1 segundo entre predicciones
IMAGE_SIZE = 350  # Tamaño de las imágenes procesadas
SHAKE_THRESHOLD = 100  # Umbral de movimiento para detectar agitado

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] PyTorch usando: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}, Memoria: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

# -----------------------------
# Transformaciones para el modelo ResNet
# ------------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convierte B/N a 3 canales para ResNet
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

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
            max_num_hands=1,  # Solo una mano ahora
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
        self.last_prediction_time = 0  # Tiempo de la última predicción
        self.predicciones = []
        self.modo = 1  # ✅ default predicción
        self.model = None
        self.letters = []  # Lista de letras para mapeo
        self.shake_start_time = 0  # Tiempo de inicio del agitado
        self.shake_detected = False  # Si se detectó agitado

        # ✅ Si arranca en modo predicción, carga el modelo de inmediato
        if self.modo == 1:
            self.cargar_modelo()

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
    # Detectar mano cerrada (todos los dedos abajo)
    # -------------------------
    @staticmethod
    def mano_cerrada(hand_landmarks):
        lm = hand_landmarks.landmark
        return all([
            lm[4].x > lm[3].x,
            lm[8].y > lm[6].y,
            lm[12].y > lm[10].y,
            lm[16].y > lm[14].y,
            lm[20].y > lm[18].y
        ])

    # -------------------------
    # Detectar agitado de mano
    # -------------------------
    def detectar_agitado(self, current_palm):
        if self.prev_palm is None:
            self.prev_palm = current_palm
            return False
        dx = current_palm[0] - self.prev_palm[0]
        dy = current_palm[1] - self.prev_palm[1]
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > SHAKE_THRESHOLD:
            if not self.shake_detected:
                self.shake_start_time = time.time()
                self.shake_detected = True
                return True
        else:
            self.shake_detected = False
        return False

    # -------------------------
    # Borrar canvas
    # -------------------------
    def borrar_canvas(self):
        self.canvas[:] = 0
        self.estado = "Borrado"
        print("[INFO] Canvas borrado")

    # -------------------------
    # Guardar dibujo en carpeta prueba
    # -------------------------
    def guardar_dibujo_dataset(self):
        if not self.selected_letter: return
        if time.time() - self.last_save_time < SAVE_COOLDOWN: return

        proc = self.procesar_dibujo(self.canvas)
        final_img = np.full_like(proc, 255)
        final_img[proc==0] = 0

        letter_dir = os.path.join(DATASET_DIR, self.selected_letter)
        os.makedirs(letter_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(letter_dir, f"dibujo_{timestamp}.png")
        cv2.imwrite(filename, final_img)
        print(f"[OK] Imagen guardada: {filename}")
        self.last_save_time = time.time()

    # -------------------------
    # Cargar modelo ResNet
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
                print("[INFO] Entrena primero el modelo con el script de entrenamiento")
                self.model = None
                
        except Exception as e:
            print(f"[ERROR] Error al cargar el modelo: {e}")
            self.model = None

    # -------------------------
    # Predecir letra con ResNet
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
        return letra, conf.item()

    # -------------------------
    # Loop principal
    # -------------------------
    def run(self):
        if self.kinect_rgb_frame() is None:
            print("Kinect no conectada. Terminando.")
            return

        print("Gestos:")
        print("- Índice levantado: Dibujar")
        print("- Mano cerrada + agitado: Borrar")
        print("- Mano abierta: Guardar/Predecir")
        print("ESC para salir")

        while True:
            frame = self.kinect_rgb_frame()
            if frame is None: break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                fingers = self.fingers_up(hand_landmarks)
                palm_x = int((hand_landmarks.landmark[0].x + hand_landmarks.landmark[9].x)/2 * WIDTH)
                palm_y = int((hand_landmarks.landmark[0].y + hand_landmarks.landmark[9].y)/2 * HEIGHT)
                palm = (palm_x, palm_y)

                if self.mano_cerrada(hand_landmarks):
                    if self.detectar_agitado(palm):
                        self.borrar_canvas()
                    self.estado = "Mano cerrada"
                    self.prev_point = None
                
                elif fingers == [0,1,0,0,0]:
                    self.estado = "Dibujo ON"
                    x = int(hand_landmarks.landmark[8].x * WIDTH)
                    y = int(hand_landmarks.landmark[8].y * HEIGHT)
                    if self.prev_point is not None:
                        cv2.line(self.canvas, self.prev_point, (x,y), (0,255,0), 4)
                    self.prev_point = (x,y)
                
                elif sum(fingers) == 5:
                    if self.prev_palm is not None and math.hypot(palm[0]-self.prev_palm[0], palm[1]-self.prev_palm[1])>50:
                        if self.modo==2: 
                            self.guardar_dibujo_dataset()
                        elif self.modo==1 and self.model:
                            letra, conf = self.predecir_letra()
                            if letra is not None and conf is not None:
                                self.predicciones.append(letra)
                                print(f"[PRED] {letra} ({conf:.2%})")
                        self.borrar_canvas()
                    self.prev_point = None
                    self.estado = "Mano abierta"
                
                else:
                    self.estado = "Dibujo OFF"
                    self.prev_point = None
                
                self.prev_palm = palm
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            combined = cv2.addWeighted(frame,1,self.canvas,1,0)
            cv2.putText(combined,f"Modo: {self.modo} - {self.estado}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            
            if self.modo == 1:
                time_remaining = max(0, PREDICTION_COOLDOWN - (time.time() - self.last_prediction_time))
                cooldown_text = f"Cooldown: {time_remaining:.1f}s"
                cv2.putText(combined, cooldown_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            if self.modo==2:
                letra_texto = f"Letra: {self.selected_letter}" if self.selected_letter else "Letra no seleccionada"
            elif self.modo==1:
                letra_texto = f"Predicciones: {''.join(self.predicciones[-10:])}"
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
