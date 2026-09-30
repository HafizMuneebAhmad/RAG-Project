from chains.query_transformer import QueryTransformer
from retriever.retriever import DocumentRetriever

class QueryRetriever:
    def __init__(self):
        self.transformer = QueryTransformer()
        self.retriever = DocumentRetriever()
        
    def search(self, question):
        better_query = self.transformer.transform(question)   
        documents = self.retriever.search(better_query)
        
        return better_query, documents
         