"""
config.py
All settings live here. Values come from the .env file,
so we never write secret keys inside the code.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file and puts the values in os.environ

# --- API keys (LangChain reads these from the environment by itself) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# --- Pinecone ---
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "support-faq")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# --- Models ---
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small gives 1536 numbers per text
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# --- Retrieval settings ---
TOP_K = int(os.getenv("TOP_K", "4"))                              # how many matches to fetch
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))  # minimum score
HISTORY_TURNS = 4                                                 # last 4 Q&A pairs used as memory

# --- Database ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chatbot.db")

# --- Safe reply when we have no good answer ---
FALLBACK_MESSAGE = (
    "Sorry, I could not find this in our help information. "
    "Please contact our support team and they will be happy to help you."
)
