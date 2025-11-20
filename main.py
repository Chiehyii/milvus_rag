import os
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any, Optional
import psycopg2

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
    log_id: int | None = None

class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    log_id: int
    feedback_type: Optional[str] = None  # "like", "dislike", or null
    feedback_text: Optional[str] = None

# --- API Endpoints ---

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Receives a user query and conversation history, processes it through the RAG pipeline,
    and returns the generated answer along with the retrieved contexts.
    """
    import logging
    print("--- [INFO] Received new chat request ---")
    try:
        print(f"--- [INFO] Processing query: '{request.query}' ---")
        # Call the chat pipeline function with the query and history
        result = chat_pipeline(request.query, request.history or [])
        
        print("--- [INFO] Successfully processed request and got result from chat_pipeline ---")
        # The result from chat_pipeline is already a dictionary like:
        # {"answer": "...", "contexts": [...], "log_id": ...}
        # FastAPI will automatically convert it to the ChatResponse model.
        return result
    except Exception as e:
        print(f"!!!!!! [ERROR] An exception occurred in chat_endpoint: {e} !!!!!!!")
        logging.error("Exception in /chat endpoint", exc_info=True)
        # When debugging on Render, it's often better to see the server crash with a full traceback.
        # FastAPI's default behavior on an unhandled exception is to log it and return a 500 error.
        # Re-raising ensures the error is not silently handled and is visible in Render's logs.
        raise

@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """
    Receives user feedback and updates the corresponding log entry in the database.
    """
    print(f"--- [INFO] Received feedback for log_id: {request.log_id} ---")
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        cursor = conn.cursor()
        
        TABLE_NAME = "qa_logs2"
        update_query = f"""UPDATE {TABLE_NAME}
                         SET feedback_type = %s, feedback_text = %s
                         WHERE id = %s;"""
        
        cursor.execute(update_query, (request.feedback_type, request.feedback_text, request.log_id))
        conn.commit()
        
        print(f"--- [INFO] Successfully updated feedback for log_id: {request.log_id} ---")
        return {"status": "success", "message": "Feedback recorded."}
        
    except psycopg2.Error as e:
        print(f"!!!!!! [ERROR] Database error in /feedback endpoint: {e} !!!!!!!")
        return {"status": "error", "message": "Failed to record feedback."}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# --- Static Files ---

# app.mount("/", StaticFiles(directory="static", html=True), name="static")
app.mount("/", StaticFiles(directory=".", html=True), name="static")
