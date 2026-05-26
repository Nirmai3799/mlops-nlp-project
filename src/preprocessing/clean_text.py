import re


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)  # keep basic chars
    return text.strip()
