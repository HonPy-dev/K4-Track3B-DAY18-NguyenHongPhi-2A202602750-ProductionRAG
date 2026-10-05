"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- LLM ---
# Model sinh câu trả lời / judging cho RAGAS. Có thể đổi qua env để dùng
# gateway OpenAI-compatible (VD: OPENAI_BASE_URL + LLM_MODEL=openai/gpt-4o-mini).
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Optional: bản model local (tránh re-download + ổn định trên máy ít bộ nhớ) ---
# Đặt biến môi trường LOCAL_MODEL_DIR trỏ tới thư mục chứa bản copy
# (VD: D:\hf_models với subfolder "bge-m3-local", "bge-reranker-local").
# Không set / không tồn tại → dùng tên model trên HF hub như bình thường.
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "")


def resolve_model_dir(hf_name: str, local_name: str) -> str:
    """Ưu tiên bản fp16 local nếu có, ngược lại dùng tên model trên HF hub."""
    if LOCAL_MODEL_DIR:
        local = os.path.join(LOCAL_MODEL_DIR, local_name)
        if os.path.isdir(local):
            return local
    return hf_name

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
