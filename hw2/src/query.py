import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ollama

from loader import load_documents
from chunking import sliding_window_chunk
from embedding import OllamaEmbedder
from vector_store import ChromaStore
from keyword_search import BM25Search
from hybrid_search import hybrid_query

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CHUNK_SIZE = 300
OVERLAP = 80
TOP_K = 3

PROMPT_TEMPLATE = """请仅根据以下参考资料回答问题，如果资料中没有相关信息，请直接说明"根据现有资料无法回答"。
回答完之后，另起一行标注"来源: "，列出你参考的资料是第几段。

【参考资料】
{context}

【问题】
{question}

【回答】"""


def build_index(embedder):
    docs = load_documents(os.path.join(ROOT, "data"))
    chunks = []
    for d in docs:
        chunks += sliding_window_chunk(d.doc_id, d.text, CHUNK_SIZE, OVERLAP)

    ids = [f"{c.doc_id}_{c.chunk_index}" for c in chunks]
    texts = [c.text for c in chunks]
    metadatas = [{"doc_id": c.doc_id, "chunk_index": c.chunk_index} for c in chunks]

    vec_store = ChromaStore(persist_dir=os.path.join(ROOT, "chroma_db_query"), collection_name="docs")
    vec_store.reset()
    BATCH = 16
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        vecs = embedder.embed_documents(batch)
        vec_store.add(ids=ids[i:i + BATCH], embeddings=vecs,
                       documents=batch, metadatas=metadatas[i:i + BATCH])

    bm25 = BM25Search(chunk_texts=texts, chunk_ids=ids)
    return vec_store, bm25


def ask(question: str, vec_store, bm25, embedder, model: str = "qwen2.5:7b"):
    results = hybrid_query(question, bm25, vec_store, embedder, top_k=TOP_K)
    context = "\n\n---\n\n".join(f"[第{i+1}段]\n{text}" for i, (_, _, text) in enumerate(results))

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    answer = resp["message"]["content"]
    return answer, results


def main():
    print("正在建立索引(分块+向量化)，稍等...")
    embedder = OllamaEmbedder () 
    vec_store, bm25 = build_index(embedder)
    print("索引建好了，可以开始提问。输入 exit 退出。\n")

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer, results = ask(question, vec_store, bm25, embedder)
        print(f"问: {question}\n答: {answer}\n")
        print("检索到的原文片段:")
        for i, (cid, score, text) in enumerate(results):
            print(f"  [{i+1}] (id={cid}) {text[:60]}...")
        return

    while True:
        question = input("你的问题: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue
        answer, results = ask(question, vec_store, bm25, embedder)
        print(f"\n回答: {answer}\n")
        print("引用来源:")
        for i, (cid, score, text) in enumerate(results):
            print(f"  [{i+1}] (id={cid}) {text[:60]}...")
        print()


if __name__ == "__main__":
    main()