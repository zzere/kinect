import os
import sounddevice as sd
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DROPOUT_RATE = 0.3
SAMPLE_RATE = 16000
MODEL_PATH = "modelo_letras_wav2vec.pth"
HIDDEN_SIZE_CLASSIFIER = 128

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

if __name__ == "__main__":
    # Descomenta la siguiente línea si quieres entrenar al iniciar
    #train_model(epochs=5000, target_val_acc=0.95)  # Usa configs por defecto, o pasa parámetros: train_model(epochs=30, learning_rate=5e-4)

    # Predicciones (solo si el modelo existe)
    try:
        #Desde archivo
        # test_audio = "prueba/B/B_1.wav"  # cambia por un audio de prueba
        # pred = predict_letter(test_audio)
        # print("🔤 Predicción archivo:", pred)

        # Desde micrófono (descomenta si quieres probar)
        while True:
            if input("¿Quieres predecir una letra desde el micrófono? (s/n): ").lower() != 's':
                break
            else :
                pred = predict_microphone(duration=1.0)
                print("🔤 Predicción micrófono:", pred)
                print(type(pred))

            
       
    except FileNotFoundError as e:
        print(e)
        print("💡 Para predecir, entrena el modelo primero: train_model()")