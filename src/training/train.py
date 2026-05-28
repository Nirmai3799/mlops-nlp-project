import os
from datasets import load_dataset
from transformers import (
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments
)

from src.training.preprocess import get_tokenizer, preprocess_function


def get_latest_checkpoint(output_dir):
    """Return latest checkpoint path if exists, else None"""
    if not os.path.exists(output_dir):
        return None

    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint")
    ]

    if len(checkpoints) == 0:
        return None

    # Sort by step number
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]


def train_model(data_path="data/processed/data_final.json"):

    # Load dataset
    dataset = load_dataset("json", data_files=data_path)

    # Tokenizer & model
    tokenizer = get_tokenizer()

    output_dir = "models/bart"

    # 🔥 Check for checkpoint
    checkpoint = get_latest_checkpoint(output_dir)

    if checkpoint:
        print(f"✅ Loading from checkpoint: {checkpoint}")
        model = BartForConditionalGeneration.from_pretrained(checkpoint)
    else:
        print("🚀 No checkpoint found. Training from scratch.")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    # Preprocess
    preprocess = preprocess_function(tokenizer)
    tokenized = dataset["train"].map(preprocess, batched=True)

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        report_to="mlflow"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized
    )

    return trainer, tokenizer, dataset, checkpoint