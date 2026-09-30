class RAGError(Exception):
    "base exception for RAG application"
    pass

class DocumentLoadError(RAGError):
    pass

class RetrievalError(RAGError):
    pass

class LLMError(RAGError):
    pass
