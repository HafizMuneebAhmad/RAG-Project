from sentence_transformers import CrossEncoder

class DocumentReranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
    def rerank(self, query, documents, top_k = 3):
        pairs = [[query,document.page_content] for document in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(documents, scores),
                        key= lambda x: x[1],
                        reverse=True)
        return [documents for documents, score in ranked[:top_k]]