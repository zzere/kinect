import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image

# ------------------------------
# Configuración
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "Dataset"  # Cambia esta ruta a la carpeta raíz de tu dataset
model_path = "modelo_letras.pth"

# ------------------------------
# Transformaciones para las imágenes
# ------------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convierte B/N a 3 canales para ResNet
    transforms.Resize((350, 350)),  # Ajuste a 350x350
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ------------------------------
# Dataset personalizado
# ------------------------------
class LetrasDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.letters = []

        # Listar las carpetas de letras (a, b, c, ...)
        self.letters = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
        self.label_map = {letter: idx for idx, letter in enumerate(self.letters)}

        # Recorrer subcarpetas may/min
        for letter in self.letters:
            letter_folder = os.path.join(data_dir, letter)
            for subfolder in os.listdir(letter_folder):  # may / min
                subfolder_path = os.path.join(letter_folder, subfolder)
                if os.path.isdir(subfolder_path):
                    for img_name in os.listdir(subfolder_path):
                        if img_name.endswith(".png"):
                            self.image_paths.append(os.path.join(subfolder_path, img_name))
                            self.labels.append(self.label_map[letter])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label

# ------------------------------
# Cargar datos (necesario para obtener las clases)
# ------------------------------
dataset = LetrasDataset(data_dir, transform)

# ------------------------------
# Modelo
# ------------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(dataset.letters))
model = model.to(device)

# ------------------------------
# Verificar si el modelo ya existe
# ------------------------------
if os.path.exists(model_path):
    print("Cargando modelo existente...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Modelo cargado exitosamente.")
    
    # Evaluación del modelo cargado
    model.eval()
    correct = 0
    total = 0
    
    # Crear dataloader de validación
    indices = list(range(len(dataset)))
    _, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy en validación: {accuracy:.2f}%")
    
else:
    print("Modelo no encontrado. Iniciando entrenamiento...")
    
    # ------------------------------
    # Dividir datos para entrenamiento
    # ------------------------------
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # ------------------------------
    # Optimizer y función de pérdida
    # ------------------------------
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------
    # Entrenamiento
    # ------------------------------
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # ------------------------------
    # Guardar modelo entrenado
    # ------------------------------
    torch.save(model.state_dict(), model_path)
    print("Modelo guardado como modelo_letras.pth")

    # ------------------------------
    # Evaluación
    # ------------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy en validación: {accuracy:.2f}%")

# ------------------------------
# Función para predecir letra de nueva imagen
# ------------------------------
def predecir_letra(imagen_path):
    image = Image.open(imagen_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    letras = dataset.letters
    return letras[predicted.item()]

# ------------------------------
# Ejemplo de predicción
# ------------------------------
letra = predecir_letra("prueba/w/dibujo_20250908_231718.png") #w
print(f"La letra predicha es: {letra}")