import re

def semantic_chunk(text: str, max_sentences: int = 3, overlap: int = 1):

    # split on paragraph breaks first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    all_sentences = []
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?]) +', para)
        all_sentences.extend([s.strip() for s in sentences if s.strip()])

    chunks = []
    i = 0

    while i < len(all_sentences):
        chunk = all_sentences[i: i + max_sentences]
        chunks.append(" ".join(chunk))
        i += max_sentences - overlap  # overlap keeps context between chunks

    return chunks