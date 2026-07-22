"""CLI entrypoints mirroring the Streamlit app's core functions."""

from __future__ import annotations

import typer
from rich.console import Console

from rag_copilot_app.config import load_settings
from rag_copilot_app.ingest import ingest_repo, list_collections
from rag_copilot_app.ollama_utils import is_model_pulled, is_ollama_running
from rag_copilot_app.rag_chain import RagSession

app = typer.Typer(help="Chat with a codebase, fully local.")
console = Console()


@app.command()
def ingest(source: str) -> None:
    """Clone/locate SOURCE and index its Python files into a local Chroma collection."""
    cfg = load_settings()
    result = ingest_repo(
        source, cfg, on_progress=lambda msg: console.print(f"[dim]{msg}[/dim]")
    )
    console.print(
        f"[green]Indexed[/green] {result.num_files} files -> {result.num_chunks} chunks "
        f"as '{result.collection_name}'"
    )


@app.command(name="list")
def list_repos() -> None:
    """List repos already indexed."""
    collections = list_collections(load_settings())
    if not collections:
        console.print("No repos indexed yet. Run `rag-copilot ingest <source>` first.")
        return
    for name in collections:
        console.print(f"- {name}")


@app.command()
def chat(collection: str) -> None:
    """Start an interactive chat against an already-indexed COLLECTION."""
    cfg = load_settings()
    if not is_ollama_running():
        console.print("[red]Ollama isn't reachable. Run `ollama serve` first.[/red]")
        raise typer.Exit(1)
    if not is_model_pulled(cfg.ollama_model):
        console.print(
            f"[red]Model not pulled. Run `ollama pull {cfg.ollama_model}`[/red]"
        )
        raise typer.Exit(1)

    session = RagSession(collection, cfg)
    console.print(
        f"Chatting with [bold]{collection}[/bold] (model: {cfg.ollama_model}). "
        "Ctrl+C to exit."
    )
    while True:
        try:
            question = typer.prompt("\nYou")
        except (KeyboardInterrupt, EOFError):
            break
        docs = session.retrieve(question)
        console.print("[bold cyan]Assistant:[/bold cyan] ", end="")
        for chunk in session.stream_answer(question, docs=docs):
            console.print(chunk, end="")
        console.print()
        sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})
        if sources:
            console.print(f"[dim]Sources: {', '.join(sources)}[/dim]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
