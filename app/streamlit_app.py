"""Streamlit UI: point at any repo, index it locally, chat with it via Ollama."""

import sys
from dataclasses import replace
from pathlib import Path

# `streamlit run` executes this file as a standalone script (not via a package
# install), so make src/ importable directly rather than depending on the
# project's editable install being on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st

from rag_copilot_app.config import load_settings
from rag_copilot_app.ingest import ingest_repo, list_collections
from rag_copilot_app.ollama_utils import is_model_pulled, is_ollama_running
from rag_copilot_app.rag_chain import RagSession

st.set_page_config(page_title="RAG Copilot", page_icon="🧑‍💻", layout="wide")

cfg = load_settings()

st.session_state.setdefault("messages", {})
st.session_state.setdefault("active_collection", None)
st.session_state.setdefault("sessions", {})

with st.sidebar:
    st.title("🧑‍💻 RAG Copilot")
    st.caption("Chat with any Python codebase, fully local.")

    st.subheader("Ollama status")
    if is_ollama_running():
        st.success("Ollama is running")
        if is_model_pulled(cfg.ollama_model):
            st.success(f"Model ready: `{cfg.ollama_model}`")
        else:
            st.error(f"Model not pulled: `{cfg.ollama_model}`")
            st.code(f"ollama pull {cfg.ollama_model}")
    else:
        st.error("Ollama isn't reachable")
        st.code("ollama serve")

    st.divider()
    st.subheader("Index a repo")
    repo_source = st.text_input(
        "GitHub URL or local path",
        placeholder="https://github.com/pallets/flask",
    )
    with st.expander("Chunking settings"):
        chunk_size = st.number_input(
            "Chunk size (characters)",
            min_value=200,
            max_value=8000,
            value=cfg.chunk_size,
            step=100,
            help="Max characters per chunk. Larger chunks give the model more "
            "surrounding context per retrieved hit, but reduce precision.",
        )
        chunk_overlap = st.number_input(
            "Chunk overlap (characters)",
            min_value=0,
            max_value=chunk_size - 1,
            value=min(cfg.chunk_overlap, chunk_size - 1),
            step=50,
            help="Characters shared between consecutive chunks, to avoid "
            "cutting a function/class definition in half.",
        )
    if st.button("Ingest", disabled=not repo_source, use_container_width=True):
        status_box = st.empty()
        ingest_cfg = replace(cfg, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        try:
            with st.spinner("Ingesting..."):
                result = ingest_repo(
                    repo_source,
                    ingest_cfg,
                    on_progress=lambda msg: status_box.info(msg),
                )
            status_box.success(
                f"Indexed {result.num_files} files -> {result.num_chunks} chunks "
                f"as '{result.collection_name}'"
            )
            st.session_state.active_collection = result.collection_name
            st.session_state.sessions.pop(result.collection_name, None)
            st.rerun()
        except Exception as exc:  # noqa: BLE001 - surface any ingestion failure in the UI
            status_box.error(f"Ingestion failed: {exc}")

    st.divider()
    st.subheader("Chat with")
    collections = list_collections(cfg)
    if collections:
        default_index = (
            collections.index(st.session_state.active_collection)
            if st.session_state.active_collection in collections
            else 0
        )
        st.session_state.active_collection = st.selectbox(
            "Indexed repos", collections, index=default_index
        )
    else:
        st.info("No repos indexed yet. Ingest one above to get started.")

    if st.session_state.active_collection and st.button(
        "Clear chat", use_container_width=True
    ):
        st.session_state.messages[st.session_state.active_collection] = []
        st.rerun()

    st.divider()
    st.subheader("Retrieval settings")
    top_k = st.slider(
        "Top K (chunks retrieved per question)",
        min_value=1,
        max_value=20,
        value=cfg.retriever_k,
        help="How many chunks the retriever pulls from the vector store to "
        "answer each question. Higher = more context, slower/noisier answers.",
    )

active = st.session_state.active_collection

st.title("Chat")
if not active:
    st.info("Index a repo from the sidebar to start chatting.")
    st.stop()

st.caption(f"Talking to: **{active}** · model: `{cfg.ollama_model}` · top-k: {top_k}")

if active not in st.session_state.sessions:
    st.session_state.sessions[active] = RagSession(active, cfg)
session = st.session_state.sessions[active]

history = st.session_state.messages.setdefault(active, [])

for message in history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- `{source}`")

question = st.chat_input("Ask about this codebase...")
if question:
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        docs = session.retrieve(question, k=top_k)
        answer = st.write_stream(session.stream_answer(question, docs=docs))
        sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.markdown(f"- `{source}`")

    history.append({"role": "assistant", "content": answer, "sources": sources})
