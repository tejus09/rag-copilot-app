"""Central, env-driven configuration for rag-copilot-app."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class Settings:
    ollama_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retriever_k: int
    data_dir: Path
    vector_db_dir: Path
    local_models_dir: Path


def _resolve(env_var: str, default_name: str) -> Path:
    raw = os.environ.get(env_var, default_name)
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings() -> Settings:
    return Settings(
        ollama_model=os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:1.5b"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        chunk_size=int(os.environ.get("CHUNK_SIZE", "1500")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "200")),
        retriever_k=int(os.environ.get("RETRIEVER_K", "5")),
        data_dir=_resolve("DATA_DIR", "data"),
        vector_db_dir=_resolve("VECTOR_DB_DIR", "vector_db"),
        local_models_dir=_resolve("LOCAL_MODELS_DIR", "local_models"),
    )


settings = load_settings()
