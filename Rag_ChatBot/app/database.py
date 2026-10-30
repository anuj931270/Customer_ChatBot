"""
database.py
We use SQL (MySQL in production, SQLite for quick local testing) to store:
  1. faqs          -> a copy of the FAQ data (question, answer, category)
  2. chat_history  -> every message, so the bot can remember the conversation
"""
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, delete, select
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()


class FAQ(Base):
    __tablename__ = "faqs"

    id = Column(Integer, primary_key=True)
    category = Column(String(50), index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)


class ChatMessage(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True, nullable=False)
    role = Column(String(10), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db() -> None:
    """Create the tables if they do not exist yet."""
    Base.metadata.create_all(engine)


def replace_all_faqs(rows: list[dict]) -> None:
    """Delete old FAQs and insert the new list. Used by the ingest script."""
    with SessionLocal() as db:
        db.execute(delete(FAQ))
        for row in rows:
            db.add(FAQ(**row))
        db.commit()


def save_message(session_id: str, role: str, content: str) -> None:
    with SessionLocal() as db:
        db.add(ChatMessage(session_id=session_id, role=role, content=content))
        db.commit()


def get_recent_history(session_id: str, limit: int = 8) -> list[tuple[str, str]]:
    """Return the last `limit` messages of a session, oldest first."""
    with SessionLocal() as db:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.desc())
            .limit(limit)
        )
        rows = db.execute(stmt).scalars().all()
    rows.reverse()  # we fetched newest first, so flip to oldest first
    return [(r.role, r.content) for r in rows]
