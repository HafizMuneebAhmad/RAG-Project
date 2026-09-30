import os
import sys
"""ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src"))"""

from chains.advanced_rag_chain import AdvancedRAGChain
from ingestion.ingestion import DocumentIngestion


class RAGService:
    def __init__(self):
        self.rag = None
        
    def initialize(self):
        ingestion = DocumentIngestion(r"C:\Users\PMLS\Desktop\RAG\data\OSIModel.pdf") 
        chunks = ingestion.run()
        
        self.rag = AdvancedRAGChain(documents= chunks)
        
    def ask(self, question):
        if self.rag is None:
            raise RuntimeError("RAG service is not initialize")
    
        return self.rag(question)
    
    def stream(self, question):
        if self.rag is None:
            raise RuntimeError("RAG service is not initialized")
        
        return self.rag.stream(question)
    
    
rag_service = RAGService()    
