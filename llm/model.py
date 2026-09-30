from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from config.settings import settings

class AIModel:
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.llm_model,
            temperature=0.2)
        
    def ask(self, question):
        response = self.llm.invoke(question)
        return response.content    