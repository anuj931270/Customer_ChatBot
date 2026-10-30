"""
rag_pipeline.py
The heart of the project. Steps for every question:
  1. Load the chat history of this session.
  2. If there is history, rewrite the question so it can stand alone.
  3. Search Pinecone (semantic search, top-k, metadata filter).
  4. Keep only matches above the similarity threshold.
  5. If nothing good was found -> return the safe fallback (no LLM call).
  6. Otherwise build a grounded prompt and ask the LLM.
  7. Save both messages in the database and return the answer.
"""
import logging

from app import config
from app.database import get_recent_history, save_message

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """You are a polite customer support assistant.
Answer the customer's question using ONLY the context given below.

Rules:
- If the context does not contain the answer, reply exactly with: {fallback}
- Do not use outside knowledge and do not guess.
- Keep the answer short, clear and friendly.

Context:
{context}"""

CONDENSE_PROMPT = (
    "Using the chat history, rewrite the customer's last message as one "
    "standalone question. If it is already clear, return it unchanged. "
    "Return only the question."
)


# ---------- small helper (easy to unit test, no API needed) ----------
def filter_by_threshold(results: list, threshold: float) -> list:
    """Keep only (document, score) pairs whose score is >= threshold."""
    return [(doc, score) for doc, score in results if score >= threshold]


# ---------- retrieval ----------
def retrieve(question: str, category: str | None = None, apply_threshold: bool = True) -> list:
    """Return a list of (document, score). Higher score = closer meaning."""
    from app.vector_store import get_vector_store

    store = get_vector_store()
    metadata_filter = {"category": category} if category else None
    results = store.similarity_search_with_score(
        question, k=config.TOP_K, filter=metadata_filter
    )
    if apply_threshold:
        results = filter_by_threshold(results, config.SIMILARITY_THRESHOLD)
    return results


# ---------- LLM helpers ----------
def _get_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=config.CHAT_MODEL, temperature=0)  # 0 = stable answers


def _history_to_messages(history: list[tuple[str, str]]) -> list:
    from langchain_core.messages import AIMessage, HumanMessage

    return [
        HumanMessage(content=text) if role == "user" else AIMessage(content=text)
        for role, text in history
    ]


def _make_standalone_question(question: str, history_messages: list) -> str:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONDENSE_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    chain = prompt | _get_llm() | StrOutputParser()
    return chain.invoke({"history": history_messages, "question": question}).strip()


def _generate_answer(question: str, context: str, history_messages: list) -> str:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    chain = prompt | _get_llm() | StrOutputParser()
    return chain.invoke(
        {
            "fallback": config.FALLBACK_MESSAGE,
            "context": context,
            "history": history_messages,
            "question": question,
        }
    ).strip()


# ---------- main function used by the API ----------
def answer_question(session_id: str, question: str, category: str | None = None) -> dict:
    history = get_recent_history(session_id, limit=config.HISTORY_TURNS * 2)
    history_messages = _history_to_messages(history)

    # Step 2: make follow-up questions clear ("and how long?" -> full question)
    search_question = question
    if history_messages:
        search_question = _make_standalone_question(question, history_messages)

    # Steps 3-4: search and keep only good matches
    hits = retrieve(search_question, category)

    # Step 5: no good match -> safe reply, no LLM call
    if not hits:
        answer = config.FALLBACK_MESSAGE
        sources: list = []
        used_fallback = True
    else:
        # Step 6: grounded answer
        context = "\n\n".join(doc.page_content for doc, _ in hits)
        answer = _generate_answer(search_question, context, history_messages)
        sources = [
            {
                "faq_id": doc.metadata.get("faq_id"),
                "question": doc.metadata.get("question"),
                "score": round(float(score), 3),
            }
            for doc, score in hits
        ]
        used_fallback = answer.strip() == config.FALLBACK_MESSAGE

    # Step 7: remember this turn
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", answer)

    logger.info("session=%s fallback=%s hits=%d", session_id, used_fallback, len(hits))
    return {
        "session_id": session_id,
        "answer": answer,
        "sources": sources,
        "used_fallback": used_fallback,
    }
