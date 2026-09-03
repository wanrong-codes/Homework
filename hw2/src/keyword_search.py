import jieba
from rank_bm25 import BM25Okapi


class BM25Search:
    def __init__(self, chunk_texts: list[str], chunk_ids: list[str]):
        assert len(chunk_texts) == len(chunk_ids)
        self.chunk_ids = chunk_ids
        self.texts = chunk_texts
        self.tokenized = [self._tokenize(t) for t in chunk_texts]
        self.bm25 = BM25Okapi(self.tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [w for w in jieba.cut(text) if w.strip()]

    def query(self, question: str, top_k: int = 5):
        tokens = self._tokenize(question)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(zip(self.chunk_ids, scores, self.texts),
                         key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
