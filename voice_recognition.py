"""Voice-controlled policy rollout using a custom keyword classifier.

Records a short clip from the microphone (via `arecord`), classifies it with
the small CNN trained by `train_voice_classifier.py` (see `voice_classifier.pt`)
into one of the 4 cube colors (red / blue / yellow / green), then runs the
trained ACT "_cube" policy for that color on the real robot via
`lerobot-rollout`.
"""

import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch

from train_voice_classifier import N_SAMPLES, VoiceCNN, WORDS, extract_features

SAMPLE_RATE = 16000
RECORD_SECONDS = 2  # must match train_voice_classifier.CLIP_SECONDS
CONFIDENCE_THRESHOLD = 0.6
COLORS = WORDS

REPO_ROOT = Path(__file__).resolve().parent
MODEL_PATH = REPO_ROOT / "voice_classifier_2.pt"

# Checkpoint dir per color. All 4 point at the "_cube" runs (not the older,
# plain red/blue/yellow policies).
POLICY_PATHS = {
    "red": REPO_ROOT / "outputs/train/red_cube_act/checkpoints/last/pretrained_model",
    "blue": REPO_ROOT / "outputs/train/blue_cube_act/checkpoints/last/pretrained_model",
    "yellow": REPO_ROOT / "outputs/train/yellow_cube_act/checkpoints/last/pretrained_model",
    "green": REPO_ROOT / "outputs/train/green_cube_act/checkpoints/last/pretrained_model",
}

TASK_DESCRIPTIONS = {
    "red": "Pick and Place the red cube",
    "blue": "Pick and Place the blue cube",
    "yellow": "Pick and Place the yellow cube",
    "green": "Pick and Place the green cube",
}

# Hardware setup — must match your actual robot/camera wiring.
ROBOT_TYPE = "so101_follower"
ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "my_awesome_follower_arm"
ROBOT_CAMERAS = (
    "{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, "
    "up: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}"
)
# All 4 "_cube" datasets have similar episode lengths (mean ~16s, p90 ~18-20s).
ROLLOUT_DURATION_S = 15


def record_audio(duration: float, sample_rate: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "clip.wav"
        subprocess.run(
            [
                "arecord",
                "-q",
                "-f", "S16_LE",
                "-r", str(sample_rate),
                "-c", "1",
                "-d", str(duration),
                str(wav_path),
            ],
            check=True,
        )
        with wave.open(str(wav_path), "rb") as wav_file:
            raw = wav_file.readframes(wav_file.getnframes())

    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0


def detect_color(audio: np.ndarray, model: VoiceCNN, device: str) -> tuple[str | None, float]:
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]

    features = extract_features(audio)
    x = torch.from_numpy(features).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    confidence, idx = probs.max(dim=0)
    confidence, idx = confidence.item(), idx.item()

    if confidence < CONFIDENCE_THRESHOLD:
        return None, confidence
    return WORDS[idx], confidence


def run_policy(color: str) -> None:
    policy_path = POLICY_PATHS.get(color)
    if policy_path is None or not policy_path.exists():
        print(f"=> No trained checkpoint for '{color}_cube' ({policy_path}). Skipping.")
        return

    task = TASK_DESCRIPTIONS[color]
    print(f"=> Running policy for {color.upper()} CUBE ({policy_path})...")
    subprocess.run(
        [
            "lerobot-rollout",
            "--strategy.type=base",
            f"--policy.path={policy_path}",
            f"--robot.type={ROBOT_TYPE}",
            f"--robot.port={ROBOT_PORT}",
            f"--robot.id={ROBOT_ID}",
            f"--robot.cameras={ROBOT_CAMERAS}",
            f"--task={task}",
            f"--duration={ROLLOUT_DURATION_S}",
        ],
        check=False,
    )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading voice classifier from '{MODEL_PATH}' (device={device})...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = VoiceCNN(n_classes=len(checkpoint["words"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print("Ready. Ctrl+C to quit.")

    try:
        while True:
            input("\nPress Enter to record...")
            print(f"Listening ({RECORD_SECONDS}s)...")
            audio = record_audio(RECORD_SECONDS, SAMPLE_RATE)

            color, confidence = detect_color(audio, model, device)
            if color:
                print(f"=> Detected color: {color.upper()} (confidence={confidence:.0%})")
                run_policy(color)
            else:
                print(f"=> No color detected (best guess confidence={confidence:.0%}, below threshold).")
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
