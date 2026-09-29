"""Train a small CNN to classify the 4 spoken color keywords in voice_dataset/.

Uses a log-spectrogram (via scipy) as input features — no extra audio
dependencies needed beyond numpy/scipy/torch, which are already installed.
Run `record_voice_dataset.py` first to build voice_dataset/<word>/*.wav.
"""

import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import spectrogram
from torch.utils.data import DataLoader, Dataset, random_split

WORDS = ["red", "blue", "yellow", "green"]
SAMPLE_RATE = 16000
CLIP_SECONDS = 2
N_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
N_FREQ_BINS = 65  # keep low/mid frequencies (up to ~4kHz), where speech energy lives

REPO_ROOT = Path(__file__).resolve().parent
DATASET_DIR = REPO_ROOT / "voice_dataset"
MODEL_PATH = REPO_ROOT / "voice_classifier_2.pt"

EPOCHS = 10000
BATCH_SIZE = 8
LEARNING_RATE = 1e-3


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        raw = wav_file.readframes(wav_file.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]
    return audio


def extract_features(audio: np.ndarray) -> np.ndarray:
    # audio is always exactly N_SAMPLES long, so the spectrogram shape below
    # is deterministic across every clip.
    _, _, spec = spectrogram(audio, fs=SAMPLE_RATE, nperseg=256, noverlap=128)
    spec = np.log(spec + 1e-6)
    spec = spec[:N_FREQ_BINS, :]
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec.astype(np.float32)


class VoiceDataset(Dataset):
    def __init__(self, dataset_dir: Path):
        self.samples: list[tuple[Path, int]] = []
        for label_idx, word in enumerate(WORDS):
            word_dir = dataset_dir / word
            for wav_path in sorted(word_dir.glob("*.wav")):
                self.samples.append((wav_path, label_idx))

        if not self.samples:
            raise RuntimeError(f"No .wav files found under {dataset_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        features = extract_features(load_wav(path))
        return torch.from_numpy(features).unsqueeze(0), label


class VoiceCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def evaluate(model: VoiceCNN, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            preds = model(features).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = VoiceDataset(DATASET_DIR)
    print(f"Loaded {len(dataset)} samples across {len(WORDS)} words")

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = VoiceCNN(n_classes=len(WORDS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{EPOCHS}  loss={total_loss / len(train_loader):.4f}  val_acc={val_acc:.2%}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "words": WORDS,
            "sample_rate": SAMPLE_RATE,
            "clip_seconds": CLIP_SECONDS,
            "n_freq_bins": N_FREQ_BINS,
        },
        MODEL_PATH,
    )
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
