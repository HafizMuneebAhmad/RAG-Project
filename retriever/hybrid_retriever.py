from retriever.bm25_retriever import BM25Retriever
from retriever.rrf import RRFFussion
from vectorstore.chroma import Vectordatabase
from retriever.base_retriever import BaseRetriever

class HybridRetriever(BaseRetriever):
    def __init__(self, documents):
        self.rrf = RRFFussion()
        self.bm25 = BM25Retriever(documents)
        self.vector_db = Vectordatabase()
        
    def search(self, query, k=3):
        vector_search = self.vector_db.search(query, k=k)   
        bm25_search = self.bm25.search(query, k=k) 
        
        
                
        return self.rrf.combined(vector_search,bm25_search,k=k)