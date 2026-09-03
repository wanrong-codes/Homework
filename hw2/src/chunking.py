from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_index: int
    start_char: int
    end_char: int
    strategy: str  # "fixed" or "sliding"


def fixed_size_chunk(doc_id: str, text: str, chunk_size: int = 300) -> list[Chunk]:
    chunks = []
    idx = 0
    pos = 0
    n = len(text)
    while pos < n:
        end = min(pos + chunk_size, n)
        chunk_text = text[pos:end]
        chunks.append(Chunk(
            text=chunk_text, doc_id=doc_id, chunk_index=idx,
            start_char=pos, end_char=end, strategy="fixed",
        ))
        idx += 1
        pos = end  
    return chunks


def sliding_window_chunk(doc_id: str, text: str, chunk_size: int = 300, overlap: int = 80) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")
    chunks = []
    idx = 0
    pos = 0
    n = len(text)
    step = chunk_size - overlap
    while pos < n:
        end = min(pos + chunk_size, n)
        chunk_text = text[pos:end]
        chunks.append(Chunk(
            text=chunk_text, doc_id=doc_id, chunk_index=idx,
            start_char=pos, end_char=end, strategy="sliding",
        ))
        idx += 1
        if end == n:
            break
        pos += step
    return chunks


def chunk_documents(docs, chunk_size: int = 300, overlap: int = 80):
    fixed_chunks, sliding_chunks = [], []
    for d in docs:
        fixed_chunks += fixed_size_chunk(d.doc_id, d.text, chunk_size)
        sliding_chunks += sliding_window_chunk(d.doc_id, d.text, chunk_size, overlap)
    return fixed_chunks, sliding_chunks


if __name__ == "__main__":
    from loader import load_documents
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs = load_documents(os.path.join(here, "data"))
    fixed, sliding = chunk_documents(docs, chunk_size=300, overlap=80)
    print(f"fixed-size: {len(fixed)} chunk")
    print(f"sliding-window: {len(sliding)} chunk")
