"""Build a retriever + local LLM chain for a previously ingested collection."""

from __future__ import annotations

from collections.abc import Iterator

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from rag_copilot_app.config import Settings, load_settings

SYSTEM_PROMPT = (
    "You are an expert software architecture assistant.\n"
    "Use the following pieces of retrieved code context to answer the question.\n"
    "Always reference the specific file names or functions from the context.\n"
    "If the context doesn't contain the answer, say so instead of guessing.\n\n"
    "Context:\n{context}"
)


def format_docs(docs: list[Document]) -> str:
    formatted_chunks = []
    for doc in docs:
        file_name = doc.metadata.get("source", "Unknown file")
        formatted_chunks.append(f"--- File: {file_name} ---\n{doc.page_content}")
    return "\n\n".join(formatted_chunks)


class RagSession:
    """Retriever + LLM bound to one Chroma collection, for one chat session."""

    def __init__(
        self,
        collection_name: str,
        cfg: Settings | None = None,
        llm: BaseChatModel | None = None,
        retriever: BaseRetriever | None = None,
    ) -> None:
        self.cfg = cfg or load_settings()
        self._fixed_retriever = retriever
        self.vectorstore: Chroma | None = None
        if retriever is None:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.cfg.embedding_model,
                cache_folder=str(self.cfg.local_models_dir),
            )
            self.vectorstore = Chroma(
                collection_name=collection_name,
                persist_directory=str(self.cfg.vector_db_dir),
                embedding_function=embeddings,
            )
        self.llm = llm or ChatOllama(model=self.cfg.ollama_model, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{question}")]
        )

    def retrieve(self, question: str, k: int | None = None) -> list[Document]:
        if self._fixed_retriever is not None:
            return self._fixed_retriever.invoke(question)
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k or self.cfg.retriever_k}
        )
        return retriever.invoke(question)

    def stream_answer(
        self, question: str, docs: list[Document] | None = None
    ) -> Iterator[str]:
        docs = docs if docs is not None else self.retrieve(question)
        messages = self.prompt.invoke(
            {"context": format_docs(docs), "question": question}
        )
        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content
