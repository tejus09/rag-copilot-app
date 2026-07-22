"""Resolve a repo source (git URL or local path) into a local directory of
Python source files, ready for splitting and embedding."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from langchain_core.documents import Document

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "site-packages",
    "egg-info",
}


def is_git_url(source: str) -> bool:
    return source.startswith(("http://", "https://", "git@")) or source.endswith(".git")


def derive_repo_name(source: str) -> str:
    name = source.rstrip("/").split("/")[-1]
    return name[:-4] if name.endswith(".git") else name


def collection_name_for(repo_path: Path, source: str) -> str:
    name = derive_repo_name(source) if is_git_url(source) else repo_path.name
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    return slug or "repo"


def clone_or_locate(source: str, data_dir: Path) -> Path:
    """Return a local directory for `source`, cloning it first if it's a git URL."""
    if is_git_url(source):
        dest = data_dir / derive_repo_name(source)
        if not dest.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["git", "clone", "--depth", "1", source, str(dest)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"git clone failed for {source}:\n{result.stderr}")
        return dest

    path = Path(source).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Local path does not exist or is not a directory: {path}")
    return path


def discover_python_files(root: Path) -> list[Path]:
    """Find *.py files under root, skipping vendored/build/VCS directories."""
    files = []
    for path in root.rglob("*.py"):
        if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts):
            continue
        files.append(path)
    return sorted(files)


def load_documents(repo_path: Path, files: list[Path]) -> list[Document]:
    """Read each file into a Document, tagging it with its path relative to the repo."""
    docs = []
    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(file_path.relative_to(repo_path))},
            )
        )
    return docs
