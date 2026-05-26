import spacy

def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

def sentence_chunking(text, max_tokens=200):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    chunks = []
    current_chunk = []

    current_length = 0

    for sent in sentences:
        length = len(sent.split())

        if current_length + length <= max_tokens:
            current_chunk.append(sent)
            current_length += length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_length = length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks