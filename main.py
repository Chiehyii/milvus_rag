import os
import sys
from fastapi import FastAPI, HTTPException
import asyncio
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any, Optional
import psycopg2
import traceback
import json
from fastapi.responses import StreamingResponse
from answer import stream_chat_pipeline

# Add the project root to the Python path to allow imports from other files
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    log_id: int
    feedback_type: Optional[str] = None  # "like", "dislike", or null
    feedback_text: Optional[str] = None

# --- API Endpoints ---

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Receives a user query and conversation history, processes it through the RAG pipeline,
    and returns a streaming response of the generated answer.
    """
    print("--- [INFO] Received new chat stream request ---")
    
    async def event_generator():
        try:
            print(f"--- [INFO] Processing query for stream: '{request.query}' ---")
            # The pipeline now yields events (content chunks or final data)
            async for event in stream_chat_pipeline(request.query, request.history or []):
                event_type = event.get("type")
                data = event.get("data")

                if event_type == "content":
                    # Send a standard message event
                    # The data is just the text chunk
                    sse_data = json.dumps({"type": "content", "data": data})
                    yield f"data: {sse_data}\n\n"
                
                elif event_type == "final_data":
                    # Send a custom named event for the final payload
                    # The data is the dict with contexts and log_id
                    sse_data = json.dumps({"type": "final_data", "data": data})
                    yield f"event: end_stream\ndata: {sse_data}\n\n"
                    
                # Yield a small delay to ensure messages are sent separately
                # await asyncio.sleep(0.01)

        except Exception as e:
            print(f"!!!!!! [ERROR] An exception occurred in stream: {e} !!!!!!!")
            traceback.print_exc()
            # Optionally, send an error event to the client
            error_message = json.dumps({"type": "error", "data": "An error occurred on the server."})
            yield f"event: error\ndata: {error_message}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """
    Receives user feedback and updates the corresponding log entry in the database.
    """
    print(f"--- [INFO] Received feedback for log_id: {request.log_id} ---")
    # Move blocking DB operations into a sync helper and run it in a thread
    def _update_feedback(log_id: int, feedback_type: str | None, feedback_text: str | None):
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

            cursor.execute(update_query, (feedback_type, feedback_text, log_id))
            conn.commit()
            return True
        except psycopg2.Error as e:
            print(f"!!!!!! [ERROR] Database error in /feedback helper: {e} !!!!!!!")
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    # Run the DB update in a thread to avoid blocking the event loop
    ok = await asyncio.to_thread(_update_feedback, request.log_id, request.feedback_type, request.feedback_text)
    if not ok:
        # Use HTTPException to return proper status code
        raise HTTPException(status_code=500, detail="Failed to record feedback")

    print(f"--- [INFO] Successfully updated feedback for log_id: {request.log_id} ---")
    return {"status": "success", "message": "Feedback recorded."}

# --- Static Files ---

# app.mount("/", StaticFiles(directory="static", html=True), name="static")
app.mount("/", StaticFiles(directory=".", html=True), name="static")
