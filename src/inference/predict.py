from transformers import BartTokenizer, BartForConditionalGeneration


def load_model(model_path="models/bart-final"):

    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)

    return tokenizer, model


def summarize(text):

    tokenizer, model = load_model()

    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    output_ids = model.generate(**inputs, max_length=128)

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return summary


if __name__ == "__main__":

    text = "Your long article here..."
    print(summarize(text))