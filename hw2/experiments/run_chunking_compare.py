import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from loader import load_documents
from chunking import chunk_documents
from embedding import OllamaEmbedder
from vector_store import ChromaStore
from test_questions import QUESTIONS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOP_K = 3
CHUNK_SIZE = 300
OVERLAP = 80


def build_store(chunks, persist_dir, embedder):
    store = ChromaStore(persist_dir=persist_dir, collection_name="docs")
    store.reset()
    ids = [f"{c.doc_id}_{c.chunk_index}" for c in chunks]
    texts = [c.text for c in chunks]
    metadatas = [{"doc_id": c.doc_id, "chunk_index": c.chunk_index} for c in chunks]

    BATCH = 16
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        vecs = embedder.embed_documents(batch)
        store.add(ids=ids[i:i + BATCH], embeddings=vecs,
                   documents=batch, metadatas=metadatas[i:i + BATCH])
    return store


def evaluate(store, embedder, top_k=TOP_K):
    results = []
    for q in QUESTIONS:
        q_vec = embedder.embed_query(q["question"])
        res = store.query(q_vec, top_k=top_k)
        retrieved_texts = res["documents"][0]
        hit = any(q["answer_span"] in t for t in retrieved_texts)
        results.append(hit)
    return results


def main():
    docs = load_documents(os.path.join(ROOT, "data"))
    fixed_chunks, sliding_chunks = chunk_documents(docs, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    print(f"fixed-size 共 {len(fixed_chunks)} 个chunk, sliding-window 共 {len(sliding_chunks)} 个chunk\n")

    embedder = OllamaEmbedder() 

    fixed_store = build_store(fixed_chunks, os.path.join(ROOT, "chroma_db_fixed"), embedder)
    sliding_store = build_store(sliding_chunks, os.path.join(ROOT, "chroma_db_sliding"), embedder)

    fixed_hits = evaluate(fixed_store, embedder)
    sliding_hits = evaluate(sliding_store, embedder)

    print(f"{'问题':<32}{'边界风险':<8}{'fixed命中':<10}{'sliding命中':<10}")
    for q, fh, sh in zip(QUESTIONS, fixed_hits, sliding_hits):
        print(f"{q['question']:<32}{str(q['boundary_risk']):<8}{str(fh):<10}{str(sh):<10}")

    fixed_rate = sum(fixed_hits) / len(fixed_hits)
    sliding_rate = sum(sliding_hits) / len(sliding_hits)
    print(f"\n整体命中率: fixed={fixed_rate:.0%}  sliding={sliding_rate:.0%}")

    boundary_idx = [i for i, q in enumerate(QUESTIONS) if q["boundary_risk"]]
    b_fixed = sum(fixed_hits[i] for i in boundary_idx) / len(boundary_idx)
    b_sliding = sum(sliding_hits[i] for i in boundary_idx) / len(boundary_idx)
    print(f"仅看边界风险题目命中率: fixed={b_fixed:.0%}  sliding={b_sliding:.0%}")


if __name__ == "__main__":
    main()
