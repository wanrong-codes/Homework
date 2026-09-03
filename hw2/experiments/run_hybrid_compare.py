import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from loader import load_documents
from chunking import sliding_window_chunk
from embedding import OllamaEmbedder
from vector_store import ChromaStore
from keyword_search import BM25Search
from hybrid_search import hybrid_query
from test_questions import QUESTIONS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOP_K = 3
CHUNK_SIZE = 300
OVERLAP = 80


def main():
    docs = load_documents(os.path.join(ROOT, "data"))
    chunks = []
    for d in docs:
        chunks += sliding_window_chunk(d.doc_id, d.text, CHUNK_SIZE, OVERLAP)

    ids = [f"{c.doc_id}_{c.chunk_index}" for c in chunks]
    texts = [c.text for c in chunks]
    metadatas = [{"doc_id": c.doc_id, "chunk_index": c.chunk_index} for c in chunks]

    embedder = OllamaEmbedder()  # 没装FlagEmbedding就改成 "ollama"

    vec_store = ChromaStore(persist_dir=os.path.join(ROOT, "chroma_db_hybrid"), collection_name="docs")
    vec_store.reset()
    BATCH = 16
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        vecs = embedder.embed_documents(batch)
        vec_store.add(ids=ids[i:i + BATCH], embeddings=vecs,
                       documents=batch, metadatas=metadatas[i:i + BATCH])

    bm25 = BM25Search(chunk_texts=texts, chunk_ids=ids)

    rows = []
    for q in QUESTIONS:
        bm25_res = bm25.query(q["question"], top_k=TOP_K)
        bm25_texts = [r[2] for r in bm25_res]
        bm25_hit = any(q["answer_span"] in t for t in bm25_texts)

        q_vec = embedder.embed_query(q["question"])
        vec_res = vec_store.query(q_vec, top_k=TOP_K)
        vec_texts = vec_res["documents"][0]
        vec_hit = any(q["answer_span"] in t for t in vec_texts)

        hyb_res = hybrid_query(q["question"], bm25, vec_store, embedder, top_k=TOP_K)
        hyb_texts = [r[2] for r in hyb_res]
        hyb_hit = any(q["answer_span"] in t for t in hyb_texts)

        rows.append((q["question"], q["qtype"], bm25_hit, vec_hit, hyb_hit))

    print(f"{'问题':<32}{'类型':<10}{'BM25':<8}{'向量':<8}{'混合':<8}")
    for r in rows:
        print(f"{r[0]:<32}{r[1]:<10}{str(r[2]):<8}{str(r[3]):<8}{str(r[4]):<8}")

    def rate(rows_subset, idx):
        return sum(r[idx] for r in rows_subset) / len(rows_subset)

    for qtype in ("keyword", "semantic"):
        subset = [r for r in rows if r[1] == qtype]
        print(f"\n[{qtype}型问题] BM25命中率={rate(subset, 2):.0%}  "
              f"向量命中率={rate(subset, 3):.0%}  混合命中率={rate(subset, 4):.0%}")

    print(f"\n[整体] BM25命中率={rate(rows, 2):.0%}  向量命中率={rate(rows, 3):.0%}  混合命中率={rate(rows, 4):.0%}")

if __name__ == "__main__":
    main()
