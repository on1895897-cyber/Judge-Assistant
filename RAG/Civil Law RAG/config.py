"""
config.py

Central configuration module for the Legal AI system.

Purpose:
---------
This file contains all environment-level and system-level constants
used across the application, including:

- File paths (documents, vector database)
- Embedding model configuration
- Indexing parameters (batch size)

Why this exists:
----------------
To avoid hardcoding paths and model names inside logic modules.
This improves maintainability, portability, and production readiness.

Design Principle:
-----------------
Configuration must be isolated from business logic.
Changing paths or models should not require editing core system code.
"""
import os
from dotenv import load_dotenv

# -----------------------------
# Load .env 
# -----------------------------
load_dotenv()

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.getcwd()
DOCS_PATH = os.path.join(BASE_DIR, "docs", "civil_law_clean.txt")
DB_DIR = os.path.join(BASE_DIR, "db", "chroma_db")

# -----------------------------
# Initializations
# -----------------------------
EMBEDDING_MODEL = "sayed0am/arabic-english-bge-m3"
BATCH_SIZE = 50
LLM_MODEL = "llama-3.3-70b-versatile"

# -----------------------------
# Default State Template
# -----------------------------
default_state_template = {
    "last_query": None,
    "last_results": [],
    "last_answer": None,
    "current_book": None,
    "current_part": None,
    "current_chapter": None,
    "current_article": None,
    "filter_type": "",
    "k": 8,
    "books_in_scope": [],
    "query_history": [],
    "retrieval_history": [],
    "retry_count": 0,
    "max_retries": 2,
    "answer_history": [],
    "db_initialized": True,
    "db": None,  # to be initialized in main or graph
    "split_config": {},
    "rewritten_question": None,
    "classification": None,
    "retrieval_confidence": None,
    "refined_query": None,
    "grade": None,
    "llm_pass": None,
    "failure_reason": None,
    "proceedToGenerate": None,
    "retrieval_attempts": 0,
    "final_answer": None
}

# -----------------------------
# Graph Constants
# -----------------------------
START = "__start__"
END = "__end__"