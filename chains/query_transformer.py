from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import settings

class QueryTransformer:
    def __init__(self):
        self.llm = ChatGroq(
            model = settings.llm_model,
            temperature = 0
        )
        self.prompt = ChatPromptTemplate.from_template(
            
         '''rewrite the user query to be more specific and detailed, so that it can be used to retrieve relevant documents from a vector database.
         User Query: {question}
         rerurn only the improved query without any additional text or explanation.'''
        ) 
        self.chain = self.prompt | self.llm | StrOutputParser()
        
    def transform(self, question):
        return self.chain.invoke({"question": question})       