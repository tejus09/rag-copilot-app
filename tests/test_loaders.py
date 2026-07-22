from pathlib import Path

import pytest

from rag_copilot_app.loaders import (
    clone_or_locate,
    collection_name_for,
    derive_repo_name,
    discover_python_files,
    is_git_url,
    load_documents,
)


@pytest.mark.parametrize(
    "source,expected",
    [
        ("https://github.com/pallets/flask", True),
        ("https://github.com/pallets/flask.git", True),
        ("git@github.com:pallets/flask.git", True),
        ("/Users/tejus/code/flask", False),
        ("./relative/path", False),
    ],
)
def test_is_git_url(source: str, expected: bool) -> None:
    assert is_git_url(source) == expected


@pytest.mark.parametrize(
    "source,expected",
    [
        ("https://github.com/pallets/flask", "flask"),
        ("https://github.com/pallets/flask.git", "flask"),
        ("https://github.com/pallets/flask/", "flask"),
    ],
)
def test_derive_repo_name(source: str, expected: str) -> None:
    assert derive_repo_name(source) == expected


def test_collection_name_for_git_url(tmp_path: Path) -> None:
    repo_path = tmp_path / "flask"
    repo_path.mkdir()
    assert collection_name_for(repo_path, "https://github.com/pallets/flask.git") == "flask"


def test_collection_name_for_local_path_sanitizes(tmp_path: Path) -> None:
    repo_path = tmp_path / "My Cool Repo!"
    repo_path.mkdir()
    assert collection_name_for(repo_path, str(repo_path)) == "my_cool_repo"


def test_clone_or_locate_missing_local_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        clone_or_locate(str(tmp_path / "does_not_exist"), tmp_path)


def test_clone_or_locate_existing_local_path(tmp_path: Path) -> None:
    repo_path = tmp_path / "myrepo"
    repo_path.mkdir()
    assert clone_or_locate(str(repo_path), tmp_path) == repo_path.resolve()


def test_discover_python_files_skips_excluded_dirs(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')")
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "vendored.py").write_text("print('nope')")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "script.py").write_text("print('nope')")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cache.py").write_text("print('nope')")

    found = discover_python_files(tmp_path)

    assert found == [tmp_path / "src" / "app.py"]


def test_discover_python_files_sorted(tmp_path: Path) -> None:
    (tmp_path / "b.py").write_text("pass")
    (tmp_path / "a.py").write_text("pass")

    found = discover_python_files(tmp_path)

    assert [f.name for f in found] == ["a.py", "b.py"]


def test_load_documents_sets_relative_source_metadata(tmp_path: Path) -> None:
    nested = tmp_path / "pkg"
    nested.mkdir()
    file_path = nested / "mod.py"
    file_path.write_text("def foo():\n    return 1\n")

    docs = load_documents(tmp_path, [file_path])

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "pkg/mod.py"
    assert "def foo" in docs[0].page_content


def test_load_documents_skips_undecodable_files(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.py"
    bad_file.write_bytes(b"\xff\xfe\x00\x01")

    docs = load_documents(tmp_path, [bad_file])

    assert docs == []
