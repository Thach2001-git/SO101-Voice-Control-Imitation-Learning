"""Record a small speech-command dataset for 4 color keywords.

For each word, records SAMPLES_PER_WORD short clips via the microphone
(`arecord`) and saves them as WAV files under voice_dataset/<word>/. Re-running
appends more samples instead of overwriting existing ones.
"""

import subprocess
import time
from pathlib import Path

WORDS = ["red", "blue", "yellow", "green"]
SAMPLES_PER_WORD = 30
SAMPLE_RATE = 16000
RECORD_SECONDS = 2  # arecord's -d only accepts whole seconds
PAUSE_SECONDS = 0.5

REPO_ROOT = Path(__file__).resolve().parent
DATASET_DIR = REPO_ROOT / "voice_dataset"


def record_clip(path: Path, duration: float, sample_rate: int) -> None:
    subprocess.run(
        [
            "arecord",
            "-q",
            "-f", "S16_LE",
            "-r", str(sample_rate),
            "-c", "1",
            "-d", str(duration),
            str(path),
        ],
        check=True,
    )


def record_word(word: str, n_samples: int) -> None:
    word_dir = DATASET_DIR / word
    word_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(word_dir.glob("*.wav")))

    input(f"\n=== Word: '{word.upper()}' — press Enter to record {n_samples} samples ===")
    for i in range(n_samples):
        idx = existing + i
        path = word_dir / f"{word}_{idx:03d}.wav"
        print(f"[{word}] sample {i + 1}/{n_samples} — say '{word}' now...")
        record_clip(path, RECORD_SECONDS, SAMPLE_RATE)
        time.sleep(PAUSE_SECONDS)
    print(f"Done: {n_samples} new samples saved to {word_dir}")


def main() -> None:
    print(f"Dataset will be saved under: {DATASET_DIR}")
    print(f"Recording {SAMPLES_PER_WORD} samples per word for: {', '.join(WORDS)}")
    for word in WORDS:
        record_word(word, SAMPLES_PER_WORD)
    print("\nAll done.")


if __name__ == "__main__":
    main()
