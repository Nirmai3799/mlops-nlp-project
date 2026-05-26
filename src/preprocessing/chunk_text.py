def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        if len(chunk) > 0:
            chunks.append(" ".join(chunk))

    return chunks
