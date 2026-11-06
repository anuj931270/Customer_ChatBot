import os

# Use a throw-away SQLite file for tests (must be set before the app is imported)
os.environ["DATABASE_URL"] = "sqlite:///./test_chatbot.db"
