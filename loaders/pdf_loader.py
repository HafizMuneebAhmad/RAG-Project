from langchain_community.document_loaders import PyPDFLoader
from utils.exception import DocumentLoadError
from utils.logger import get_logger

logger = get_logger(__name__)

class pdfloader:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def load (self):
        try:
            logger.info(f"Loading PDF from: {self.file_path}")
            
            loader = PyPDFLoader(self.file_path)   
             
            documents = loader.load()
            
            if not documents:
                raise DocumentLoadError("PDF contain no readable pages.")
            
            logger.info(f"Successfully loaded {len(documents)} pages from PDF.")
            return documents
        
        except DocumentLoadError:
            raise
        
        except Exception as e:
            logger.exception("An error occurred while loading the PDF.")
        
        except Exception as e:
            raise DocumentLoadError(f"Failed to load PDF: {e}")