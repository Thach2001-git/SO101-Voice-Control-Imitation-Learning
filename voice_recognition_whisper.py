"""Voice-controlled policy rollout using OpenAI Whisper.

Records a short clip from the microphone (via `arecord`), transcribes it with
Whisper, matches the words against the 4 cube colors (red / blue / yellow /
green), then runs the trained ACT "_cube" policy for that color on the real
robot via `lerobot-rollout`.

See `voice_recognition.py` for the alternative version that uses a small
custom-trained classifier (`train_voice_classifier.py`) instead of Whisper.
"""

import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import whisper

SAMPLE_RATE = 16000
RECORD_SECONDS = 3
MODEL_NAME = "base"
COLORS = ["red", "blue", "yellow", "green"]

REPO_ROOT = Path(__file__).resolve().parent

# Checkpoint dir per color. All 4 point at the "_cube" runs.
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


def detect_color(text: str) -> str | None:
    text = text.lower()
    for color in COLORS:
        if re.search(rf"\b{color}\b", text):
            return color
    return None


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
    print(f"Loading Whisper model '{MODEL_NAME}'...")
    model = whisper.load_model(MODEL_NAME)
    print("Ready. Press Enter and say a command (red / blue / yellow / green). Ctrl+C to quit.")

    try:
        while True:
            input("\nPress Enter to record...")
            print(f"Listening ({RECORD_SECONDS}s)...")
            audio = record_audio(RECORD_SECONDS, SAMPLE_RATE)

            result = model.transcribe(audio, language="en", fp16=False)
            text = result["text"].strip()
            print(f"Heard: '{text}'")

            color = detect_color(text)
            if color:
                print(f"=> Detected color: {color.upper()}")
                run_policy(color)
            else:
                print("=> No color detected among the 4 colors (red/blue/yellow/green).")
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
