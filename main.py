import os
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any

# Add the project root to the Python path to allow imports from other files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the core chat logic
from answer import chat_pipeline

# --- API Definition ---

app = FastAPI(
    title="Chatbot RAG API",
    description="An API for the Milvus RAG chatbot.",
    version="1.0.0",
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Pydantic Models for Request and Response ---

class ChatRequest(BaseModel):
    """Request model for a user's chat query."""
    query: str
    history: List[dict] | None = None

class ChatResponse(BaseModel):
    """Response model for the chatbot's answer."""
    answer: str
    contexts: List[Any] # The context can be complex, so using List[Any] for simplicity

# --- API Endpoints ---

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Receives a user query and conversation history, processes it through the RAG pipeline,
    and returns the generated answer along with the retrieved contexts.
    """
    # Call the chat pipeline function with the query and history
    result = chat_pipeline(request.query, request.history or [])
    
    # The result from chat_pipeline is already a dictionary like:
    # {"answer": "...", "contexts": [...]}
    # FastAPI will automatically convert it to the ChatResponse model.
    return result

# To run this API, save it as main.py and run the following command in your terminal:
# uvicorn main:app --reload

# --- Static Files ---
# app.mount("/", StaticFiles(directory="static", html=True), name="static")
app.mount("/", StaticFiles(directory=".", html=True), name="static")
