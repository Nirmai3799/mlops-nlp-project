import mlflow
import torch

from datasets import load_dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Trainer,
    TrainingArguments
)


def train_model(data_path="data/processed/data_final.json"):

    # Load dataset
    dataset = load_dataset("json", data_files=data_path)

    # Load tokenizer + model
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    def preprocess(example):
        inputs = tokenizer(
            example["input_text"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )

        targets = tokenizer(
            example["target_summary"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )

        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized = dataset["train"].map(preprocess, batched=True)

    # Training config
    training_args = TrainingArguments(
        output_dir="models/bart",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized
    )

    return trainer

trainer = train_model()

mlflow.set_tracking_uri("http://127.0.0.1:5000")

with mlflow.start_run(run_name="bart-training"):

    trainer.train()

    mlflow.log_param("model", "bart-base")
    mlflow.log_param("epochs", 1)

    mlflow.log_metric("dummy_metric", 1)

    print("✅ Training completed!")