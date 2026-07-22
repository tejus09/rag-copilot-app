"""Chain wiring tests using LangChain's built-in fakes — no real Ollama call,
no embedding model download, so these run fine in CI."""

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever

from rag_copilot_app.rag_chain import RagSession, format_docs


class FakeRetriever(BaseRetriever):
    docs: list[Document]

    def _get_relevant_documents(self, query, *, run_manager=None):  # noqa: ANN001
        return self.docs


def test_format_docs_labels_each_chunk_with_its_source() -> None:
    docs = [
        Document(page_content="def a(): pass", metadata={"source": "a.py"}),
        Document(page_content="def b(): pass", metadata={"source": "b.py"}),
    ]

    formatted = format_docs(docs)

    assert "--- File: a.py ---" in formatted
    assert "--- File: b.py ---" in formatted
    assert "def a(): pass" in formatted


def test_format_docs_handles_missing_source_metadata() -> None:
    formatted = format_docs([Document(page_content="x = 1", metadata={})])
    assert "Unknown file" in formatted


def test_rag_session_retrieve_delegates_to_retriever() -> None:
    docs = [Document(page_content="code", metadata={"source": "m.py"})]
    session = RagSession(
        "fake_collection",
        retriever=FakeRetriever(docs=docs),
        llm=FakeListChatModel(responses=["irrelevant"]),
    )

    assert session.retrieve("what does this do?") == docs


def test_rag_session_stream_answer_yields_llm_output() -> None:
    docs = [Document(page_content="code", metadata={"source": "m.py"})]
    session = RagSession(
        "fake_collection",
        retriever=FakeRetriever(docs=docs),
        llm=FakeListChatModel(responses=["hello world"]),
    )

    answer = "".join(session.stream_answer("question", docs=docs))

    assert answer == "hello world"
