# Preproceesing json  - to create embeddings and store them in a joblib file for later use. Then, it allows the user to ask a question and retrieves the most relevant chunks based on cosine similarity.
import json
import joblib
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_JSON = BASE_DIR / "data" / "output.json"

OUTPUT_JOBLIB = BASE_DIR / "data" / "embeddings.joblib"

# ==========================================================
# Load Embedding Model
# ==========================================================

print("=" * 60)
print("Loading Embedding Model...")
print("=" * 60)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Model Loaded Successfully!")

# ==========================================================
# Load Transcript
# ==========================================================

print("\nReading Transcript...")

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Total Chunks : {len(chunks)}")

# ==========================================================
# Extract Text
# ==========================================================

texts = [chunk["text"] for chunk in chunks]

# ==========================================================
# Generate Embeddings
# ==========================================================

print("\nCreating Embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Embeddings Created Successfully!")

# ==========================================================
# Create DataFrame
# ==========================================================

records = []

for chunk, embedding in zip(chunks, embeddings):

    records.append({

        "chunk_id": chunk["chunk_id"],

        "start": chunk["start"],

        "end": chunk["end"],

        "text": chunk["text"],

        "embedding": embedding

    })

df = pd.DataFrame(records)

# ==========================================================
# Save DataFrame
# ==========================================================

joblib.dump(df, OUTPUT_JOBLIB)

print("\nEmbeddings Saved Successfully!")

print(f"Location : {OUTPUT_JOBLIB}")

print("\nDataFrame Shape :", df.shape)

print(df.head())