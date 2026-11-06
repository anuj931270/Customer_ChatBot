"""
ingest.py  -  Prepare the knowledge base (run this once, and again when FAQs change).

What it does:
  1. Reads data/faqs.csv
  2. Saves a copy of the FAQs in the SQL database
  3. Turns each FAQ into a Document (text + metadata)
  4. Splits long answers into small chunks
  5. Creates embeddings with OpenAI and stores them in Pinecone

Run:  python -m scripts.ingest
Updating the FAQ file only needs this script again. No model training is needed.
"""
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
from app.database import init_db, replace_all_faqs
from app.vector_store import clear_index, ensure_index, get_vector_store

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "faqs.csv"


def load_faqs(path: Path) -> list[dict]:
    """Read the CSV, clean it, and return a list of rows with ids."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["question", "answer"]).drop_duplicates(subset=["question"])
    for col in ["category", "question", "answer"]:
        df[col] = df[col].astype(str).str.strip()
    df["category"] = df["category"].str.lower()

    rows = df.to_dict(orient="records")
    for i, row in enumerate(rows, start=1):
        row["id"] = i
    return rows


def build_documents(rows: list[dict]) -> tuple[list[Document], list[str]]:
    """Make one Document per FAQ. Long answers are split into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs, ids = [], []

    for row in rows:
        text = f"Question: {row['question']}\nAnswer: {row['answer']}"
        metadata = {
            "faq_id": row["id"],
            "category": row["category"],
            "question": row["question"],
        }
        for n, chunk in enumerate(splitter.split_text(text)):
            docs.append(Document(page_content=chunk, metadata=metadata))
            ids.append(f"faq-{row['id']}-{n}")
    return docs, ids


def main() -> None:
    rows = load_faqs(CSV_PATH)
    print(f"Loaded {len(rows)} FAQs from {CSV_PATH.name}")

    init_db()
    replace_all_faqs(rows)
    print("Saved FAQs in the SQL database")

    docs, ids = build_documents(rows)
    ensure_index()
    clear_index()  # remove old vectors, so deleted FAQs disappear
    get_vector_store().add_documents(docs, ids=ids)
    print(f"Stored {len(docs)} chunks in Pinecone index '{config.PINECONE_INDEX_NAME}'")
    print("Done. The knowledge base is ready.")


if __name__ == "__main__":
    main()
