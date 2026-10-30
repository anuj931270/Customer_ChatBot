"""
main.py
The FastAPI app. This is the door through which users talk to the bot.
Run:  uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.database import get_recent_history, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # create tables when the server starts
    yield


app = FastAPI(title="Enterprise RAG Customer Support Chatbot", version="1.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64, examples=["user-101"])
    question: str = Field(..., min_length=2, max_length=1000, examples=["How can I track my order?"])
    category: str | None = Field(default=None, examples=["orders"])


class Source(BaseModel):
    faq_id: int | None = None
    question: str | None = None
    score: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source]
    used_fallback: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    from app.rag_pipeline import answer_question  # imported here to keep startup fast

    try:
        return answer_question(request.session_id, request.question, request.category)
    except Exception:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")


@app.get("/history/{session_id}")
def history(session_id: str):
    messages = get_recent_history(session_id, limit=50)
    return {"session_id": session_id, "messages": [{"role": r, "content": c} for r, c in messages]}
