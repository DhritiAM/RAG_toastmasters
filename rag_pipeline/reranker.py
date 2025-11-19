from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents):
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        reranked = [doc for _, doc in sorted(zip(scores, documents), reverse=True)]
        return reranked
