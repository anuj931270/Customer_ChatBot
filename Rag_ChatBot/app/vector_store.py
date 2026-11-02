"""
vector_store.py
Everything about Pinecone (our vector database) and embeddings.
Heavy imports are inside the functions, so simple tests run fast.
"""
import time
from functools import lru_cache

from app import config


def ensure_index() -> None:
    """Create the Pinecone index if it does not exist."""
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    if config.PINECONE_INDEX_NAME in pc.list_indexes().names():
        return

    pc.create_index(
        name=config.PINECONE_INDEX_NAME,
        dimension=config.EMBEDDING_DIMENSION,
        metric="cosine",  # we compare vectors with cosine similarity
        spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
    )
    # wait until the index is ready to use
    while not pc.describe_index(config.PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)


def clear_index() -> None:
    """Remove all old vectors so deleted FAQs do not stay in the search."""
    from pinecone import Pinecone

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX_NAME)
    try:
        index.delete(delete_all=True)
    except Exception:
        pass  # a brand new index has nothing to delete


@lru_cache(maxsize=1)
def get_vector_store():
    """LangChain wrapper: embeddings + Pinecone index in one object."""
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore

    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    return PineconeVectorStore(index_name=config.PINECONE_INDEX_NAME, embedding=embeddings)
