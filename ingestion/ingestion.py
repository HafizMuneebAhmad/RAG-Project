from loaders.pdf_loader import pdfloader
from splitters.text_splitters import text_splitters
from vectorstore.chroma import Vectordatabase

class DocumentIngestion:
    def __init__(self, file_path):
        self.loader =pdfloader(file_path)
        self.splitter = text_splitters()
        self.vector_db = Vectordatabase()
        
    def run(self):
        documents  = self.loader.load()
        
        chunks = self.splitter.split(documents)  
        
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"chunk_{i}"
        
        self.vector_db.add_documents(chunks) 
        
        return chunks
