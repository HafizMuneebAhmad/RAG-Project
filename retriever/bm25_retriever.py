from rank_bm25 import BM25Okapi
from retriever.base_retriever import BaseRetriever

class BM25Retriever(BaseRetriever):
    def __init__(self, documents):
         self.documents = documents
        
         tokenized_documents  = [
            document.page_content.lower().split()
            for document in documents
         ]
         self.bm25 = BM25Okapi(tokenized_documents)
        
    def search(self, query, k=3):
        tokenized_query = query.lower().split()
        result = self.bm25.get_top_n(
            tokenized_query,
            self.documents,
            n=k
        )    
        return result