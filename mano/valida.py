import torch
import torchaudio
from torch.utils.data import DataLoader
import os

# Configuración
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_model(model_path, dataset_path="dataset"):
    """
    Valida un modelo entrenado con el dataset completo
    """
    
    # Cargar modelo
    checkpoint = torch.load(model_path, map_location=DEVICE)
    classes = checkpoint['classes']
    config = checkpoint.get('config', {'hidden_size': 128, 'dropout': 0.3})
    
    # Cargar dataset de validación
    from transformers import Wav2Vec2Model
    
    class LetterAudioDataset(torch.utils.data.Dataset):
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

            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            if waveform.size(1) > self.sample_rate:
                waveform = waveform[:, :self.sample_rate]
            elif waveform.size(1) < self.sample_rate:
                pad_size = self.sample_rate - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, pad_size))

            return waveform.squeeze(0), label
    
    # Crear modelo
    class Wav2VecClassifier(torch.nn.Module):
        def __init__(self, base_model, num_classes, hidden_size=128, dropout=0.3):
            super(Wav2VecClassifier, self).__init__()
            self.base = base_model
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.base.config.hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size, num_classes)
            )

        def forward(self, x):
            with torch.no_grad():
                features = self.base(x).last_hidden_state
                pooled = features.mean(dim=1)
            return self.classifier(pooled)
    
    # Cargar modelo y dataset
    wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model = Wav2VecClassifier(wav2vec2, len(classes), 
                             hidden_size=config['hidden_size'], 
                             dropout=config['dropout']).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dataset = LetterAudioDataset(dataset_path, SAMPLE_RATE)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Validación
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

# Uso
if __name__ == "__main__":
    p = "dataset"
    models = ["modelo_letras_wav2vec.pth", "1.pth", "primero.pth", "big.pth"]
    
    for model_path in models:
        if os.path.exists(model_path):
            acc = validate_model(model_path, p)
            print(f"{model_path}: {acc:.4f}")
        else:
            print(f"{model_path}: No encontrado")