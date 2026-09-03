from typing import Protocol


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...



class OllamaEmbedder:
    def __init__(self, model_name: str = "bge-m3"):
        import ollama
        self.client = ollama
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        resp = self.client.embeddings(model=self.model_name, prompt=text)
        return resp["embedding"]

