from vectorstore.chroma import Vectordatabase

class DocumentRetriever:
    def __init__(self):
        self.vector_db = Vectordatabase()
        
    def get_retriever(self):
           retriever = self.vector_db.db.as_retriever(
               search_type = "mmr",
               search_kwarges = {"k":3,
                                 "fetch_k": 10,
                                 "lambda_mult": 0.5 # Range(0-1) 0 for diversity and 1 for relevance
                                 }
           )
           return retriever 
    def search(self, query, k=3):
        retriever = self.get_retriever()
        return retriever.invoke(query)   
       