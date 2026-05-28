import mlflow

from src.training.train import train_model
from src.training.evaluate import evaluate_model


def run():

    # MLflow setup
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("bart-summarization")

    # Get trainer
    trainer, tokenizer, dataset, checkpoint = train_model()

    # Resume training if checkpoint exists
    trainer.train(resume_from_checkpoint=checkpoint)

    with mlflow.start_run(run_name="bart-training"):

        # Params
        mlflow.log_param("model", "facebook/bart-base")
        mlflow.log_param("epochs", 1)
        mlflow.log_param("batch_size", 2)

        # Train
        trainer.train()

        # Evaluate
        results = evaluate_model(trainer, dataset["train"], tokenizer)

        for k, v in results.items():
            mlflow.log_metric(k, v)

        # Save final model
        trainer.save_model("models/bart-final")
        tokenizer.save_pretrained("models/bart-final")

        print("✅ Training + Evaluation Completed!")


if __name__ == "__main__":
    run()