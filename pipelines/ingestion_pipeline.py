import json
import os

from src.ingestion.load_data import load_cnn_dailymail
from src.preprocessing.clean_text import clean_text
from src.preprocessing.chunk_text import chunk_text


RAW_PATH = "data/raw/data.json"
PROCESSED_PATH = "data/processed/data_chunks.json"


def run_pipeline():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("Loading dataset...")
    data = load_cnn_dailymail(split="train[:1000]")  # limit for now

    print("Saving raw data...")
    with open(RAW_PATH, "w") as f:
        json.dump(data, f, indent=2)

    processed_data = []

    print("Processing data...")
    for item in data:
        clean_article = clean_text(item["article"])
        chunks = chunk_text(clean_article)

        for chunk in chunks:
            processed_data.append({
                "chunk": chunk,
                "summary": item["summary"]
            })

    print("Saving processed data...")
    with open(PROCESSED_PATH, "w") as f:
        json.dump(processed_data, f, indent=2)

    print("✅ Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()
