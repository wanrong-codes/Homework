import json
import re
import zlib
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from .config import EMBED_MODE, EMBED_MODEL, EMBED_PROVIDER
from .providers import Provider


# --------------------------------------------------------------------------- #
# 1. Chunking：按段落聚合到 chunk_size，超长段落硬切，相邻块保留 overlap
# --------------------------------------------------------------------------- #
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    overlap = max(0, min(overlap, chunk_size // 2))
    limit = max(chunk_size - overlap - 1, 1)  

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        if len(p) <= limit:
            pieces.append(p)
        else:  
            pieces.extend(p[i:i + limit] for i in range(0, len(p), limit))

    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 2 > chunk_size:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail}\n{piece}" if tail else piece
        else:
            buf = f"{buf}\n\n{piece}" if buf else piece
    if buf:
        chunks.append(buf)
    return chunks


# --------------------------------------------------------------------------- #
# 2. Embedding
# --------------------------------------------------------------------------- #
class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashEmbedder:

    def __init__(self, dim: int = 768):
        self.dim = dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        text = text.lower()
        words = re.findall(r"[a-z0-9]+", text)
        cjk = re.findall(r"[\u4e00-\u9fff]", text)
        bigrams = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
        return words + cjk + bigrams

    async def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in self._tokens(t):
                out[i, zlib.crc32(tok.encode("utf-8")) % self.dim] += 1.0
        return _l2_normalize(out)


class ApiEmbedder:

    def __init__(self, provider: Provider, model: str, batch_size: int = 64):
        self.provider, self.model, self.batch_size = provider, model, batch_size

    async def embed(self, texts: list[str]) -> np.ndarray:
        vecs: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vecs.extend(await self.provider.embed(self.model, texts[i:i + self.batch_size]))
        return _l2_normalize(np.asarray(vecs, dtype=np.float32))


def build_embedder(providers: dict[str, Provider]) -> Embedder:
    if EMBED_MODE == "api":
        return ApiEmbedder(providers[EMBED_PROVIDER], EMBED_MODEL)
    return HashEmbedder()


# --------------------------------------------------------------------------- #
# 3. 向量库：内存 numpy 矩阵 + JSON 落盘
# --------------------------------------------------------------------------- #
class VectorStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.items: list[dict] = []
        self.vecs: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.items = data["items"]
            if self.items:
                self.vecs = np.asarray(data["vecs"], dtype=np.float32).reshape(len(self.items), -1)

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"items": self.items, "vecs": self.vecs.tolist()},
                                        ensure_ascii=False), encoding="utf-8")

    def add(self, items: list[dict], vecs: np.ndarray) -> None:
        if len(items) != len(vecs):
            raise ValueError("items/vecs length mismatch")
        if len(self.items) and vecs.shape[1] != self.vecs.shape[1]:
            raise ValueError("embedding dim mismatch with existing store; delete data/store.json and re-ingest")
        self.vecs = vecs.astype(np.float32) if not len(self.items) else np.vstack([self.vecs, vecs])
        self.items.extend(items)
        self._save()

    def delete_doc(self, doc_id: str) -> int:
        keep = [i for i, it in enumerate(self.items) if it["doc_id"] != doc_id]
        removed = len(self.items) - len(keep)
        if removed:
            self.items = [self.items[i] for i in keep]
            self.vecs = self.vecs[keep] if keep else np.zeros((0, 0), dtype=np.float32)
            self._save()
        return removed

    def search(self, qvec: np.ndarray, k: int = 4) -> list[tuple[float, dict]]:
        if not self.items:
            return []
        if qvec.shape[0] != self.vecs.shape[1]:
            raise ValueError("query embedding dim mismatch with store")
        sims = self.vecs @ qvec  # 向量已归一化，点积 = 余弦相似度
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.items[i]) for i in idx]

    def stats(self) -> dict:
        docs: dict[str, int] = {}
        for it in self.items:
            docs[it["doc_id"]] = docs.get(it["doc_id"], 0) + 1
        return {"chunks": len(self.items), "docs": docs}


# --------------------------------------------------------------------------- #
# 4. RAG 服务：ingest / retrieve / build_context
# --------------------------------------------------------------------------- #
class RAGService:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder, self.store = embedder, store

    async def ingest(self, doc_id: str, text: str, chunk_size: int = 500, overlap: int = 80) -> dict:
        chunks = chunk_text(text, chunk_size, overlap)
        if not chunks:
            raise ValueError("document is empty")
        vecs = await self.embedder.embed(chunks)
        self.store.delete_doc(doc_id)  # 同一个 doc_id 重复上传 = 覆盖
        self.store.add([{"doc_id": doc_id, "chunk_id": i, "text": c} for i, c in enumerate(chunks)], vecs)
        return {"doc_id": doc_id, "chunks": len(chunks)}

    async def retrieve(self, query: str, top_k: int = 4, min_score: float = 0.05) -> list[dict]:
        qvec = (await self.embedder.embed([query]))[0]
        hits = self.store.search(qvec, top_k)
        return [{"doc_id": it["doc_id"], "chunk_id": it["chunk_id"], "score": round(s, 4), "text": it["text"]}
                for s, it in hits if s >= min_score]

    @staticmethod
    def build_context(hits: list[dict]) -> str:
        if not hits:
            return "（未检索到相关资料）"
        return "\n\n".join(f"[{i}] ({h['doc_id']}#{h['chunk_id']}) {h['text']}" for i, h in enumerate(hits, 1))
