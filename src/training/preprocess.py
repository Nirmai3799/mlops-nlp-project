from transformers import BartTokenizer

def get_tokenizer():
    return BartTokenizer.from_pretrained("facebook/bart-base")


def preprocess_function(tokenizer):

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

    return preprocess