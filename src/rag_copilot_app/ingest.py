"""Ingest a repo (git URL or local path) into a per-repo Chroma collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag_copilot_app.config import Settings, load_settings
from rag_copilot_app.loaders import (
    clone_or_locate,
    collection_name_for,
    discover_python_files,
    load_documents,
)
from rag_copilot_app.splitter import split_documents

ProgressFn = Callable[[str], None]


@dataclass
class IngestResult:
    collection_name: str
    repo_path: Path
    num_files: int
    num_chunks: int


def ingest_repo(
    source: str,
    cfg: Settings | None = None,
    on_progress: ProgressFn | None = None,
) -> IngestResult:
    cfg = cfg or load_settings()
    notify = on_progress or (lambda _msg: None)

    notify(f"Resolving {source}...")
    repo_path = clone_or_locate(source, cfg.data_dir)
    collection_name = collection_name_for(repo_path, source)

    notify("Discovering Python files...")
    files = discover_python_files(repo_path)
    if not files:
        raise ValueError(f"No Python files found in {repo_path}")

    notify(f"Loading {len(files)} files...")
    docs = load_documents(repo_path, files)

    notify("Splitting into chunks...")
    split_docs = split_documents(docs, cfg.chunk_size, cfg.chunk_overlap)

    notify(f"Embedding {len(split_docs)} chunks (first run downloads the model)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.embedding_model, cache_folder=str(cfg.local_models_dir)
    )

    cfg.vector_db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(cfg.vector_db_dir))
    existing = {c.name for c in client.list_collections()}
    if collection_name in existing:
        notify(f"Replacing existing index for '{collection_name}'...")
        client.delete_collection(collection_name)

    Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(cfg.vector_db_dir),
    )
    notify("Done.")

    return IngestResult(
        collection_name=collection_name,
        repo_path=repo_path,
        num_files=len(files),
        num_chunks=len(split_docs),
    )


def list_collections(cfg: Settings | None = None) -> list[str]:
    cfg = cfg or load_settings()
    if not cfg.vector_db_dir.exists():
        return []
    client = chromadb.PersistentClient(path=str(cfg.vector_db_dir))
    return sorted(c.name for c in client.list_collections())
