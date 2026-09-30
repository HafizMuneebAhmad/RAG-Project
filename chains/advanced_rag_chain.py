from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from chains.query_transformer import QueryTransformer
from retriever.advanced_retriever import AdvancedRetriever
from chains.chat_history import ChatHistory
from config.settings import settings

def format_docs(documents):
    formated = []
    for doc in documents:
        if isinstance(doc, list):
            doc = doc[0] if doc else None
        if hasattr(doc, "page_content"):
                formated.append(doc.page_content)    
        else:
            formated.append(str(doc))
    return "\n\n".join(formated)

class AdvancedRAGChain:
    def __init__(self, documents):
        self.llm = ChatGroq(
            model = settings.llm_model,
            temperature = 0
        )    
        self.query_transformer = QueryTransformer()
        self.retriever = AdvancedRetriever(documents)
        self.history = ChatHistory()
        
        self.prompt = ChatPromptTemplate.from_template(
                    '''
                    You are a helpful assistant that uses the following context to answer the question.
                    Conversation histort {history} 
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.
                    \n\nContext: {context}\n\nQuestion: {question}\nAnswer:
                        '''
                ) 
        self.parser = StrOutputParser()
        
    def ask(self, question):
            better_query =self.query_transformer.transform(question) 
            
            documents = self.retriever.search(query= better_query, candidate_k=10, final_k=3)   
            
            context =format_docs(documents) 
            
            history = self.history.get_history()
            
            chain = (self.prompt| self.llm| self.parser)
            
            answer = chain.invoke({
                "context": context,
                "question": question,
                "history": history
            })    
            self.history.add_user_message(question)
            self.history.add_ai_message(answer)
            
            return answer
    def stream(self, question):
            better_query =self.query_transformer.transform(question) 
                
            documents = self.retriever.search(query= better_query, candidate_k=10, final_k=3)   
                
            context =format_docs(documents) 
                
            history = self.history.get_history()
                
            chain = (self.prompt| self.llm)   
            
            for chunk in chain.stream({
                "context": context,
                "question": question,
                "history": history
            }):
                if chunk.content:
                    yield chunk.content
                    