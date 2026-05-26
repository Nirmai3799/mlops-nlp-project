import json
import os

from src.preprocessing.clean_text import clean_text
from src.preprocessing.sentence_chunking import sentence_chunking


RAW_PATH = "data/raw/data.json"
PROCESSED_PATH = "data/processed/data_final.json"


def run_pipeline():
    os.makedirs("data/processed", exist_ok=True)

    print("Loading raw data...")
    with open(RAW_PATH, "r") as f:
        data = json.load(f)

    final_data = []

    print("Processing data...")
    for item in data:
        article = clean_text(item["article"])
        summary = clean_text(item["summary"])

        # filter short texts
        if len(article.split()) < 50:
            continue

        chunks = sentence_chunking(article)

        for chunk in chunks:
            if len(chunk.split()) < 20:
                continue

            final_data.append({
                "input_text": chunk,
                "target_summary": summary
            })

    print(f"Total samples: {len(final_data)}")

    print("Saving final dataset...")
    with open(PROCESSED_PATH, "w") as f:
        json.dump(final_data, f, indent=2)

    print("✅ Preprocessing completed!")


if __name__ == "__main__":
    run_pipeline()