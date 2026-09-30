import re
from dotenv import load_dotenv
load_dotenv()
import os
import sys
"""ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, ""))"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from api.schemas import AskRequest, AskResponse
from services.rag_service import RAGService, rag_service
from api.dependencies import get_rag_service
from contextlib import asynccontextmanager


from utils.logger import get_logger

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app : FastAPI):
    logger.info("Starting RAG application...")
    
    rag = get_rag_service()
    
    rag.initialize()
    
    logger.info("RAG service is initialized.")
    
    yield
    
    logger.info("Shutting down RAG application.")

app = FastAPI(
    title="RAG API",
    version="1.0.0",
    lifespan= lifespan)

@app.get("/")
def home():
    return{
        "message":" RAG api is running"
    }
    
    
@app.post("/ask", response_model=AskResponse)
def ask(request : AskRequest,
        rag : RAGService = Depends(get_rag_service)):
    try:
        answer = rag.ask(request.question)
        return AskResponse(
            question=request.question,
            answer=answer
        )
         
    except Exception:
        logger.exception("RAG Request Failed")    
        raise HTTPException(
            status_code=500,
            detail="Failed to generate answer"
        )
@app.post("/ask/stream")
def ask_stream(request : AskRequest,
        rag : RAGService = Depends(get_rag_service)):
    try:
        def genarate_chunks():
            for chunk in rag.stream(request.question):
                if chunk is None:
                    continue
                if isinstance(chunk, str):
                    yield chunk
                elif isinstance(chunk, list):
                    yield
                    "".join(str(item) for item in chunk)
                else:
                    yield str(chunk)

        return StreamingResponse(
            genarate_chunks(),
            media_type= "text/plain"
        )
    
         
    except Exception:
        logger.exception("Streamin Request Failed")    
        raise HTTPException(
            status_code=500,
            detail="Streaming Failed"
        )        
        