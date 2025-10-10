import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import sounddevice as sd
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import Wav2Vec2Model
import warnings



# ===============================
# Configuración (ajustable)
# ===============================
SAMPLE_RATE = 16000
DATASET_PATH = "dataset"  # cambia esta ruta a tu dataset
MODEL_PATH = "modelo_letras_wav2vec.pth"

# Variables de entrenamiento (ajusta aquí para cambiar precisión, loss, épocas, etc.)
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3
HIDDEN_SIZE_CLASSIFIER = 128
TRAIN_SPLIT = 0.8  # Porcentaje para entrenamiento (resto para validación)
CRITERION_TYPE = 'CrossEntropyLoss'  # Opciones: 'CrossEntropyLoss', 'MSELoss', etc.
OPTIMIZER_TYPE = 'Adam'  # Opciones: 'Adam', 'SGD', etc.
WEIGHT_DECAY = 1e-5  # Decay para regularización L2 en el optimizador
TARGET_VAL_ACC = 0.95  # Precisión de validación deseada (0.95 = 95%). None = entrena todas las épocas.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


warnings.filterwarnings("ignore", message=".*torchaudio.*")

# ===============================
# Dataset personalizado
# ===============================
class LetterAudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.files = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        for label in self.classes:
            folder = os.path.join(root_dir, label)
            for f in os.listdir(folder):
                if f.endswith(".wav"):
                    self.files.append(os.path.join(folder, f))
                    self.labels.append(self.classes.index(label))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]
        waveform, sr = torchaudio.load(audio_path)

        # resample
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # pad/trim a 1 segundo
        if waveform.size(1) > self.sample_rate:
            waveform = waveform[:, :self.sample_rate]
        elif waveform.size(1) < self.sample_rate:
            pad_size = self.sample_rate - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))

        return waveform.squeeze(0), label

# ===============================
# Modelo con Wav2Vec2 (usa variables configurables)
# ===============================
class Wav2VecClassifier(nn.Module):
    def __init__(self, base_model, num_classes, hidden_size=128, dropout=0.3):
        super(Wav2VecClassifier, self).__init__()
        self.base = base_model
        self.classifier = nn.Sequential(
            nn.Linear(self.base.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.base(x).last_hidden_state
            pooled = features.mean(dim=1)  # average pooling
        return self.classifier(pooled)

# ===============================
# Funciones de entrenamiento (internas)
# ===============================
def get_criterion(criterion_type='CrossEntropyLoss'):
    if criterion_type == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif criterion_type == 'MSELoss':
        return nn.MSELoss()
    else:
        raise ValueError(f"Tipo de criterion no soportado: {criterion_type}")

def get_optimizer(model, optimizer_type='Adam', lr=1e-3, weight_decay=1e-5):
    if optimizer_type == 'Adam':
        return optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        return optim.SGD(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Tipo de optimizador no soportado: {optimizer_type}")

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader), correct / total

def validate(model, dataloader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(dataloader), correct / total

# ===============================
# Función principal de entrenamiento (nueva)
# ===============================
def train_model(dataset_path=DATASET_PATH, epochs=EPOCHS, batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE, dropout_rate=DROPOUT_RATE,
                hidden_size=HIDDEN_SIZE_CLASSIFIER, train_split=TRAIN_SPLIT,
                criterion_type=CRITERION_TYPE, optimizer_type=OPTIMIZER_TYPE,
                weight_decay=WEIGHT_DECAY, target_val_acc=TARGET_VAL_ACC, model_path=MODEL_PATH):
    """
    Entrena el modelo con las configuraciones proporcionadas.
    Solo se ejecuta cuando se llama esta función.
    
    Args:
        target_val_acc (float, optional): Precisión de validación deseada (e.g., 0.95).
            Si se alcanza o supera, detiene el entrenamiento temprano. None = entrena todas las épocas.
    """
    print("🚀 Iniciando entrenamiento...")
    if target_val_acc is not None:
        print(f"🎯 Objetivo de precisión de validación: {target_val_acc:.2%}")
    
    # Cargar dataset
    dataset = LetterAudioDataset(dataset_path, SAMPLE_RATE)
    num_classes = len(dataset.classes)

    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=16, pin_memory=True)

    # Crear modelo con configs
    wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model = Wav2VecClassifier(wav2vec2, num_classes, hidden_size=hidden_size, dropout=dropout_rate).to(DEVICE)

    # Criterion y optimizer con configs
    criterion = get_criterion(criterion_type)
    optimizer = get_optimizer(model, optimizer_type, lr=learning_rate, weight_decay=weight_decay)

    # Entrenamiento con early stopping opcional
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Verificar early stopping basado en val_acc deseado
        if target_val_acc is not None and val_acc >= target_val_acc:
            print(f"🎯 Precisión de validación deseada alcanzada: {val_acc:.4f} (>= {target_val_acc:.4f})")
            print("⏹️  Deteniendo entrenamiento temprano.")
            break

    # Guardar modelo (incluso si se detuvo temprano)
    save_model(model, dataset.classes, model_path)
    final_val_acc = validate(model, val_loader, criterion)[1]  # Recalcular para confirmar
    print(f"✅ Entrenamiento completado. Precisión final de validación: {final_val_acc:.4f}")
    if target_val_acc is not None:
        print(f"📊 ¿Objetivo alcanzado? {'Sí' if final_val_acc >= target_val_acc else 'No'}")


# ===============================
# Guardar / Cargar modelo (actualizado para usar configs)
# ===============================
def save_model(model, classes, path=MODEL_PATH):
    # Guardar configs junto con el modelo para recreación
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'classes': classes,
        'config': {
            'hidden_size': model.classifier[0].out_features,  # Primer linear out
            'dropout': model.classifier[2].p  # Dropout rate
        }
    }
    torch.save(checkpoint, path)
    print(f"📦 Modelo guardado en {path}")

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el modelo en {path}. Entrena primero con train_model().")
    
    checkpoint = torch.load(path, map_location=DEVICE)
    classes = checkpoint['classes']
    config = checkpoint.get('config', {'hidden_size': HIDDEN_SIZE_CLASSIFIER, 'dropout': DROPOUT_RATE})
    num_classes = len(classes)
    wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model = Wav2VecClassifier(wav2vec2, num_classes, hidden_size=config['hidden_size'], dropout=config['dropout']).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, classes

# ===============================
# Predicción (sin cambios)
# ===============================
def predict_letter(audio_path, model_path=MODEL_PATH):
    model, classes = load_model(model_path)

    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    if waveform.size(1) > SAMPLE_RATE:
        waveform = waveform[:, :SAMPLE_RATE]
    elif waveform.size(1) < SAMPLE_RATE:
        pad_size = SAMPLE_RATE - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))

    waveform = waveform.squeeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(waveform)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]

def predict_microphone(duration=1.0, model_path=MODEL_PATH):
    print("🎤 Di una letra... grabando...")
    audio = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("✅ Grabación terminada")

    waveform = torch.tensor(audio.T, dtype=torch.float32)
    if waveform.size(1) > SAMPLE_RATE:
        waveform = waveform[:, :SAMPLE_RATE]
    elif waveform.size(1) < SAMPLE_RATE:
        pad_size = SAMPLE_RATE - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))

    waveform = waveform.squeeze(0).unsqueeze(0).to(DEVICE)

    model, classes = load_model(model_path)

    with torch.no_grad():
        outputs = model(waveform)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]

# ===============================
# Main (ahora no entrena por defecto)
# ===============================
if __name__ == "__main__":
    # Descomenta la siguiente línea si quieres entrenar al iniciar
    train_model(epochs=5000, target_val_acc=0.95)  # Usa configs por defecto, o pasa parámetros: train_model(epochs=30, learning_rate=5e-4)

    # Predicciones (solo si el modelo existe)
    try:
        #Desde archivo
        test_audio = "prueba/B/B_1.wav"  # cambia por un audio de prueba
        pred = predict_letter(test_audio)
        print("🔤 Predicción archivo:", pred)

        # Desde micrófono (descomenta si quieres probar)
        # pred = predict_microphone(duration=1.0)
        # print("🔤 Predicción micrófono:", pred)
    except FileNotFoundError as e:
        print(e)
        print("💡 Para predecir, entrena el modelo primero: train_model()")
