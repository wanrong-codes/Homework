import chromadb


class ChromaStore:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "rag_docs"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: list[str], embeddings: list[list[float]],
            documents: list[str], metadatas: list[dict]):
        self.collection.add(ids=ids, embeddings=embeddings,
                             documents=documents, metadatas=metadatas)

    def query(self, query_embedding: list[float], top_k: int = 5):
        return self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name, metadata={"hnsw:space": "cosine"},
        )
