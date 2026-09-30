from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sympy import im
from retriever.retriever import DocumentRetriever
from chains.query_transformer import QueryTransformer
from chains.query_retriever import QueryRetriever
from config.settings import settings

def format_docs(documents):
    return "\n\n".join(
        document.page_content
        for document in documents
    )

class RAGChain:
    def __init__(self):
        self.llm = ChatGroq(
            model = settings.llm_model,
            temperature = 0.2
        )
        self.query_transform = QueryTransformer()
        self.retriever = DocumentRetriever()
        
        
        self.prompt = ChatPromptTemplate.from_template(
            '''
            You are a helpful assistant that uses the following context to answer the question. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            \n\nContext: {context}\n\nQuestion: {question}\nAnswer:
                '''
        )
       
    
    def ask(self, question):
        better_query =self.query_transform.transform(question) 
        
        documents = self.retriever.search(better_query)   
        
        context =format_docs(documents)
        
        response = (self.prompt| self.llm| StrOutputParser()).invoke({
            "context": context,
            "question": question
        })
        return response