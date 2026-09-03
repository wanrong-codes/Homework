
def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60, top_k: int = 5):

    scores: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]


def hybrid_query(question: str, bm25_search, chroma_store, embedder, top_k: int = 5, fetch_k: int = 20):

    bm25_results = bm25_search.query(question, top_k=fetch_k)
    bm25_ids = [r[0] for r in bm25_results]

    q_vec = embedder.embed_query(question)
    vec_result = chroma_store.query(q_vec, top_k=fetch_k)
    vec_ids = vec_result["ids"][0]

    fused = reciprocal_rank_fusion([bm25_ids, vec_ids], top_k=top_k)

    id_to_text = {r[0]: r[2] for r in bm25_results}
    for cid, doc in zip(vec_result["ids"][0], vec_result["documents"][0]):
        id_to_text.setdefault(cid, doc)

    return [(cid, score, id_to_text.get(cid, "")) for cid, score in fused]
