from langchain_core.documents import Document

from rag_copilot_app.splitter import split_documents


def test_split_documents_single_short_doc_stays_one_chunk() -> None:
    doc = Document(page_content="def foo():\n    return 1\n", metadata={"source": "a.py"})

    chunks = split_documents([doc], chunk_size=1500, chunk_overlap=200)

    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "a.py"


def test_split_documents_long_doc_produces_multiple_chunks() -> None:
    function_block = "def f_{i}():\n    return {i}\n\n\n"
    long_source = "".join(function_block.format(i=i) for i in range(200))
    doc = Document(page_content=long_source, metadata={"source": "big.py"})

    chunks = split_documents([doc], chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "big.py" for chunk in chunks)
    assert all(len(chunk.page_content) <= 200 + 20 for chunk in chunks)


def test_split_documents_preserves_content() -> None:
    doc = Document(page_content="x = 1\ny = 2\n", metadata={"source": "vars.py"})

    chunks = split_documents([doc], chunk_size=1500, chunk_overlap=0)

    assert "x = 1" in chunks[0].page_content
    assert "y = 2" in chunks[0].page_content
