# mp3 to json

import whisper
import json
from pathlib import Path

# --------------------------------------------------------
# Load Whisper Model
# --------------------------------------------------------

print("Loading Whisper Model...")

model = whisper.load_model("base")

print("Model Loaded Successfully")

# --------------------------------------------------------
# Paths
# --------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

audio_path = BASE_DIR / "audios" / "sample.mp3"

output_path = BASE_DIR / "data" / "output.json"

# --------------------------------------------------------
# Transcribe Audio
# --------------------------------------------------------

print("Transcribing Audio...")

result = model.transcribe(
    str(audio_path),
    language="hi",
    task="translate",
    word_timestamps=False
)

# --------------------------------------------------------
# Create Chunks
# --------------------------------------------------------

chunks = []

for i, segment in enumerate(result["segments"]):

    chunks.append({

        "chunk_id": i,

        "start": round(segment["start"], 2),

        "end": round(segment["end"], 2),

        "text": segment["text"].strip()

    })

# --------------------------------------------------------
# Save JSON
# --------------------------------------------------------

output_path.parent.mkdir(exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:

    json.dump(chunks, f, indent=4, ensure_ascii=False)

print()

print("Transcript Saved")

print(f"Total Chunks : {len(chunks)}")

print(f"Location : {output_path}")