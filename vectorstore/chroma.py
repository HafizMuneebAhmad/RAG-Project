from langchain_chroma import Chroma
from embeddings.embeddings import EmbeddingModel
from config.settings import settings


class Vectordatabase:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.db = Chroma(
            collection_name="rag_documents",
            embedding_function=self.embedding_model.embeddings,
            persist_directory=settings.chroma_dir
        )
    def add_documents(self, documents):
        ids = [
            f"{document.metadata.get('source')}_{i}"
            for i, document in enumerate(documents)
        ]
        
        self.db.add_documents(
                            documents= documents,
                            ids = ids)  
    
    def search(self, query, k=3):
        result = self.db.similarity_search(
            query,
            k=k
        )
        return result      
    def search_by_topic(self, query,topic, k=3):
        result = self.db.similarity_search(
            query,
            k=k,
            filter={"topic": topic}
        )