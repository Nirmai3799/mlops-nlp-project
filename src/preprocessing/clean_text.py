import re


def clean_text(text: str) -> str:
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+", "", text)

    # remove special characters (keep punctuation)
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()