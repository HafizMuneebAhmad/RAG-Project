from langchain_huggingface import HuggingFaceEmbeddings 


class EmbeddingModel:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
    def create_embedding(self, text):
        vector = self.embeddings.embed_query(text)
        return vector     
 