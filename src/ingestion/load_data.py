from datasets import load_dataset
import os


def load_cnn_dailymail(split="train", save_path="data/raw"):
    os.makedirs(save_path, exist_ok=True)

    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split=split)

    data = []
    for item in dataset:
        data.append({
            "article": item["article"],
            "summary": item["highlights"]
        })

    return data
