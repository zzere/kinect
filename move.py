import freenect
import cv2
import numpy as np  # ✅ faltaba este import

# Inicializar contexto y abrir Kinect (dispositivo 0)
ctx = freenect.init()
dev = freenect.open_device(ctx, 0)

angulo = int(input("Ingrese el ángulo de inclinación deseado (-30 a 30): "))

freenect.set_tilt_degs(dev,angulo)
cv2.destroyAllWindows()
print("Kinect centrado en 0° y programa finalizado.")
