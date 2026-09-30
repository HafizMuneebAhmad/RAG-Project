from retriever.bm25_retriever import BM25Retriever
from retriever.rrf import RRFFussion
from reranker.reranker import DocumentReranker
from vectorstore.chroma import Vectordatabase
from retriever.base_retriever import BaseRetriever
from utils.exception import RetrievalError
from utils.logger import get_logger

logger = get_logger(__name__)

class AdvancedRetriever(BaseRetriever):
    def __init__(self, documents):
        if not documents:
            raise RetrievalError("No documents available for retrieval.")
        self.bm25 = BM25Retriever(documents)
        self.documents= documents
        self.rrf = RRFFussion()
        self.vector_db = Vectordatabase()
        self.reranker = DocumentReranker()
        
        
    def search(self, query, candidate_k=10, final_k=3):
        try:
            if not query.strip():
                raise RetrievalError("Query cannot be empty")
            
            logger.info(f"Searching for query: {query}")   
            
            vector_results = self.vector_db.search(query, k=candidate_k)  
            
            bm25_results = self.bm25.search(query, k = candidate_k)  
            
            candidates = self.rrf.combined(vector_results, bm25_results, k= candidate_k)
            
            final_documents = self.reranker.rerank(query, candidates, top_k= final_k)
            
            logger.info(f"Final documents: {len(final_documents)} retrieved for query: {query}")
            
            return final_documents 
        
        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(f"Retrieval Failed:{e}")