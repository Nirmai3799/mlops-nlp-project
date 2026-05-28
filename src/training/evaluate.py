import evaluate

rouge = evaluate.load("rouge")

def evaluate_model(model, dataset, tokenizer):

    predictions = []
    references = []

    for sample in dataset.select(range(50)):

        inputs = tokenizer(
            sample["input_text"],
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to('cpu') for k, v in inputs.items()}

        output_ids = model.generate(**inputs, max_length=128)

        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        predictions.append(pred)
        references.append(sample["target_summary"])

    return rouge.compute(predictions=predictions, references=references)